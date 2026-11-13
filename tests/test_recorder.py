import gzip
import os

from tradebot.strategies.mtf_entry import recorder as R


def test_side_depth_counts_only_levels_within_pct():
    bids = [[99.5, 3, 10], [99.0, 1, 10], [97.0, 2, 10]]          # mid 100, 1% => >= 99.0
    assert R.side_depth(bids, 100.0, 0.1, 0.01, True) == 99.5 * 10 * 0.1 + 99.0 * 10 * 0.1
    asks = [[100.5, 1, 5], [101.5, 1, 5]]
    assert R.side_depth(asks, 100.0, 1.0, 0.01, False) == 100.5 * 5
    assert R.side_depth(None, 100.0, 1.0, 0.01, True) == 0.0


class FakeClient:
    def _request(self, method, path, params=None, **k):
        if path.endswith("/detail"):
            return [{"symbol": "AAA_USDT", "contractSize": 1}, {"symbol": "BBB_USDT", "contractSize": 10}]
        if "/depth/" in path:
            return {"bids": [[9.95, 1, 100]], "asks": [[10.05, 1, 100]]}
        raise AssertionError(path)

    def tickers(self):
        return [{"symbol": "AAA_USDT", "lastPrice": 10.0, "amount24": 5_000_000, "holdVol": 123, "fundingRate": 0.0001},
                {"symbol": "BBB_USDT", "lastPrice": 1.0, "amount24": 1000, "holdVol": 5, "fundingRate": 0},
                {"symbol": "XXX_USD", "lastPrice": 1}]


def test_snapshot_records_oi_funding_and_depth_only_for_liquid(tmp_path):
    rows = R.snapshot(FakeClient(), now=1_700_000_000)
    assert [r["symbol"] for r in rows] == ["AAA_USDT", "BBB_USDT"]
    a, b = rows
    assert a["oi"] == 123 and a["funding"] == 0.0001 and a["bid_d1"] == 995.0 and a["ask_d1"] == 1005.0
    assert b["bid_d1"] == ""                                       # illiquid: no depth call
    p = R.write_snapshot(rows, str(tmp_path), now=1_700_000_000)
    assert os.path.basename(p) == "snap_20231114_2213.csv.gz"
    with gzip.open(p, "rt") as f:
        assert f.readline().strip().split(",") == R.FIELDS


def test_consolidate_merges_past_days_only(tmp_path):
    d = str(tmp_path)
    for n in ("snap_20231113_0000.csv.gz", "snap_20231113_0005.csv.gz", "snap_20231114_0000.csv.gz"):
        with gzip.open(os.path.join(d, n), "wt") as f:
            f.write("ts,symbol\n1,AAA\n")
    made = R.consolidate(d, today="20231114")
    assert [os.path.basename(m) for m in made] == ["day_20231113.csv.gz"]
    with gzip.open(made[0], "rt") as f:
        assert f.read().splitlines() == ["ts,symbol", "1,AAA", "1,AAA"]           # one header, two data rows
    assert sorted(os.listdir(d)) == ["day_20231113.csv.gz", "snap_20231114_0000.csv.gz"]   # today's file untouched
