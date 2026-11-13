import gzip
import os

from tradebot.strategies.mtf_entry import recorder as R


def test_side_depth_counts_only_levels_within_pct():
    bids = [[99.5, 10, 3], [99.0, 10, 1], [97.0, 10, 2]]          # mid 100, 1% => >= 99.0
    assert R.side_depth(bids, 100.0, 0.1, 0.01, True) == 99.5 * 10 * 0.1 + 99.0 * 10 * 0.1
    asks = [[100.5, 5, 1], [101.5, 5, 1]]
    assert R.side_depth(asks, 100.0, 1.0, 0.01, False) == 100.5 * 5
    assert R.side_depth(None, 100.0, 1.0, 0.01, True) == 0.0


class FakeClient:
    def _request(self, method, path, params=None, **k):
        if path.endswith("/detail"):
            return [{"symbol": "AAA_USDT", "contractSize": 1}, {"symbol": "BBB_USDT", "contractSize": 10}]
        if "/depth/" in path:
            return {"bids": [[9.95, 100, 1]], "asks": [[10.05, 100, 1]]}
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


def test_coverage_flags_truncated_books():
    bids = [[99.9, 1, 1], [99.5, 1, 1]]                # reaches only 0.5% below mid
    asks = [[100.1, 1, 1], [103.0, 1, 1]]
    assert abs(R.coverage(bids, asks, 100.0) - 0.005) < 1e-12
    assert R.coverage(None, asks, 100.0) == 0.0


def test_flow_aggregator_minutes_flags_and_flush():
    from tradebot.strategies.mtf_entry.flow_recorder import Aggregator, write_part
    a = Aggregator({"AAA_USDT": 0.01})
    t0 = 1_700_000_000_000                                        # ms, some minute M
    a.add("AAA_USDT", 100.0, 5, 1, 1, t0)                         # buy opening: 100*5*0.01 = 5 USDT
    a.add("AAA_USDT", 101.0, 10, 2, 3, t0 + 1000)                 # sell, unchanged: 10.1 USDT
    a.add("AAA_USDT", 102.0, 1, 1, 1, t0 + 61_000)                # next minute
    m = int(t0 // 60000 * 60)
    done = a.flush(m + 70)                                         # minute M finished, M+1 still open
    assert len(done) == 1 and done[0]["minute"] == m and done[0]["n"] == 2
    assert done[0]["v_T1_O1"] == 5.0 and done[0]["v_T2_O3"] == 10.1 and done[0]["p_open"] == 100.0 and done[0]["p_close"] == 101.0
    rest = a.flush(m + 200, keep_current=False)
    assert len(rest) == 1 and rest[0]["minute"] == m + 60
    import tempfile
    p = write_part(done, tempfile.mkdtemp(), "t")
    assert p and p.endswith(".csv.gz") and write_part([], "/tmp/x", "t") is None
