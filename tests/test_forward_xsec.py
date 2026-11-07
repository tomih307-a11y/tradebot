import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import forward_xsec as fx   # noqa: E402

DAY = 86400


def test_select_portfolio_takes_top_and_bottom_quintiles_deterministically():
    sc = {f"S{i:02d}_USDT": float(i) for i in range(50)}
    longs, shorts = fx.select_portfolio(sc)
    assert len(longs) == len(shorts) == 10 and longs[-1] == "S49_USDT" and shorts[0] == "S00_USDT"
    assert fx.select_portfolio({"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0})[0] == ["E"]          # ties broken by symbol


def test_settle_net_return_matches_the_registered_formula_and_rejects_missing_prices():
    tr = {"longs": {"A": 100.0, "B": 100.0}, "shorts": {"C": 100.0, "D": 100.0}, "status": "open"}
    out = fx.settle(dict(tr), {"A": 110.0, "B": 110.0, "C": 90.0, "D": 90.0})
    assert abs(out["gross"] - 0.10) < 1e-12 and abs(out["net"] - (0.10 - 0.0018)) < 1e-12 and out["status"] == "settled"
    bad = fx.settle(dict(tr), {"A": 110.0})                      # most prices missing at exit
    assert bad["status"] == "failed"


class FakeClient:
    page_limit = 1000

    def __init__(self, n=60, seed=0):
        self.n, self.seed = n, seed

    def tickers(self):
        return [{"symbol": f"S{i}_USDT", "lastPrice": 100.0 + i, "amount24": 1e9 - i} for i in range(80)]

    def klines(self, sym, tf, a, b):
        i = int(sym[1:-5])
        rng = np.random.default_rng(i)
        c = 100 * np.exp(np.cumsum(rng.normal(0.001 * (i % 7 - 3), 0.03, self.n)))
        t0 = (b // DAY - self.n) * DAY
        t = t0 + np.arange(self.n) * DAY
        return {"time": t.tolist(), "open": c.tolist(), "high": c.tolist(), "low": c.tolist(), "close": c.tolist(),
                "vol": [1.0] * self.n, "amount": [1e7 + 1e5 * i] * self.n}


def test_open_is_idempotent_and_creates_three_tranches():
    st = {"start": None, "tranches": []}
    n, why = fx.open_tranches(FakeClient(), st, 1_800_000_000, 80, 60, "2026-11-07")
    assert n == 3 and why == "ok" and {t["signal"] for t in st["tranches"]} == {"mom30", "mom7", "random"}
    t = st["tranches"][0]
    assert len(t["longs"]) == len(t["shorts"]) == 12 and not set(t["longs"]) & set(t["shorts"])
    assert fx.open_tranches(FakeClient(), st, 1_800_000_000 + 600, 80, 60, "2026-11-07")[0] == 0          # same day again: nothing


def test_verdict_uses_the_registered_thresholds_and_minimum_sample():
    nets = np.random.default_rng(1).normal(0.004, 0.01, 120)
    ctrl = np.random.default_rng(2).normal(-0.002, 0.01, 120)
    st = fx.summarize_signal(nets, ctrl)
    assert st["n"] == 120 and "نجحت" in fx.verdict("mom30", {**st, "t_hac": 3.0, "ci_lo": 0.1, "exp_pct": 0.4, "control_mean_pct": -0.2})
    assert "لم تتحقق" in fx.verdict("mom7", {**st, "t_hac": 2.0})                      # H2 needs t >= 2.24
    assert "لم يبلغ" in fx.verdict("mom30", {**st, "n": 40})


def test_write_outputs_produces_summary_with_control_row():
    state = {"start": 0, "tranches": []}
    rng = np.random.default_rng(3)
    for sig in fx.SIGNALS:
        for d in range(100):
            state["tranches"].append({"date": f"d{d}", "signal": sig, "entry_ts": d * DAY, "exit_ts": (d + 7) * DAY, "status": "settled",
                                      "gross": float(rng.normal(0, 0.01)), "net": float(rng.normal(-0.002, 0.01)), "n_long": 20, "n_short": 20, "longs": {}, "shorts": {}})
    with tempfile.TemporaryDirectory() as td:
        md = fx.write_outputs(Path(td), state)
        assert (Path(td) / "results.csv").exists() and "mom30 (H1)" in md and "شاهد" in md
