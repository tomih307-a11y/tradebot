import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from smc_study import simulate_trade, summarize   # noqa: E402


def c15(rows):
    a = np.array(rows, float)                      # open, high, low, close
    return {"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3]}


def run(rows, s=1, entry=100.0, stop=98.0, tps=(102.0, 104.0, 106.0), mode="limit", be=False, fee=0.0, slip=0.0, j0=0):
    return simulate_trade(s, entry, stop, list(tps), c15(rows), j0, mode, 4, 10, be, fee, slip)


def test_limit_never_touched_is_not_filled():
    assert not run([[105, 106, 104, 105]] * 5)["filled"]


def test_all_targets_hit_gives_mean_of_r_multiples():
    # risk 2: TP1/2/3 are 1R, 2R, 3R -> average 2R
    r = run([[101, 101, 99.5, 100], [100, 102.5, 100, 102], [102, 104.5, 102, 104], [104, 106.5, 104, 106]])
    assert r["filled"] and r["reason"] == "all_tp" and abs(r["R_gross"] - 2.0) < 1e-9 and r["tp_hit"] == 3


def test_stop_is_minus_one_r_and_adverse_first_in_fill_candle():
    # the fill candle also dips to the stop: counted as a full loss, favourable high ignored
    r = run([[100.5, 103, 97.5, 99]])
    assert r["reason"] == "stop" and abs(r["R_gross"] + 1.0) < 1e-9


def test_breakeven_after_tp1_turns_loss_into_partial_win():
    rows = [[101, 101, 99.5, 100], [100, 102.2, 100, 102], [102, 102, 99.0, 99.5]]      # TP1, then back to entry
    plain = run(rows, be=False)
    be = run(rows, be=True)
    assert be["R_gross"] > plain["R_gross"] and abs(be["R_gross"] - (1 / 3) * 1.0) < 1e-9   # 1/3 at +1R, rest flat


def test_short_mirror_and_fees_reduce_r():
    rows = [[99, 100.5, 99, 100], [100, 100, 97.8, 98], [98, 98, 95.8, 96], [96, 96, 93.8, 94]]
    g = run(rows, s=-1, entry=100.0, stop=102.0, tps=(98.0, 96.0, 94.0))
    f = run(rows, s=-1, entry=100.0, stop=102.0, tps=(98.0, 96.0, 94.0), fee=0.04)
    assert abs(g["R_gross"] - 2.0) < 1e-9 and f["R_net"] < g["R_net"]


def test_market_mode_enters_at_open_and_summary():
    r = run([[100, 100.2, 99.9, 100], [100, 102.1, 100, 102]], mode="market")
    assert r["filled"] and r["fill_j"] == 0
    s = summarize([1.0, -1.0, 2.0])
    assert s["n"] == 3 and abs(s["total"] - 2.0) < 1e-9 and s["win"] > 60


def test_tier_threshold_is_float_safe_and_ci_brackets_mean():
    from smc_study2 import TIERS, boot_ci, tier_min
    assert 4 / 6 >= tier_min(4) and 5 / 6 >= tier_min(5) and 3 / 6 >= tier_min(3)
    assert 3 / 6 < tier_min(4) and 4 / 6 < tier_min(5)                 # tiers are really distinct
    rng = np.random.default_rng(0)
    tr = [{"T": 86400 * (i // 3), "R_net": float(x)} for i, x in enumerate(rng.normal(0.2, 1.0, 600))]
    lo, hi = boot_ci(tr)
    assert lo < np.mean([t["R_net"] for t in tr]) < hi and hi - lo < 0.5
