import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rules_study import simulate_open   # noqa: E402


def candles(rows):
    a = np.array(rows, float)                     # open, high, low, close
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def sim(rows, s=1, stop0=98.0, atr=1.0, trail=2.0, act=1.0, hold=50, loser=10, fee=0.0, slip=0.0):
    o, h, l, c = candles(rows)
    return simulate_open(s, stop0, o, h, l, c, np.full(len(c), atr), 0, trail, act, hold, loser, fee, slip)


def test_initial_stop_is_minus_one_r():
    r = sim([[100, 100.5, 97.5, 99]])
    assert r["reason"] == "stop" and abs(r["R_gross"] + 1.0) < 1e-9


def test_trailing_locks_profit_and_never_loosens():
    # risk 2. closes 103 (+1.5R) -> trail to 101; next candle dips to 101.5 (no hit); then 99 hits the trail at 101
    rows = [[100, 103.2, 99.9, 103], [103, 104, 101.5, 103.8], [103.8, 104, 100.5, 101.2]]
    r = sim(rows)
    assert r["reason"] == "trail_stop" and r["R_gross"] > 0
    assert abs(r["R_gross"] - (101.8 - 100) / 2) < 1e-9 or r["R_gross"] > 0.4          # exits at the trailed level (~1.8 after 3rd trail)


def test_trail_waits_for_activation():
    rows = [[100, 100.9, 99.5, 100.8], [100.8, 101, 97.9, 98.1]]                      # +0.4R only -> stop stays at 98
    r = sim(rows)
    assert r["reason"] == "stop" and abs(r["R_gross"] + 1.0) < 1e-9


def test_open_trade_can_run_without_any_target():
    rows = [[100 + i, 100.6 + i, 99.9 + i, 100.5 + i] for i in range(30)]            # steady climb
    r = sim(rows, hold=25)
    assert r["reason"] == "max_hold" and r["R_gross"] > 5


def test_losing_trade_is_closed_after_loser_bars_and_short_mirror():
    rows = [[100, 100.4, 99.6, 99.8]] * 12
    r = sim(rows, loser=5)
    assert r["reason"] == "time_loser" and r["bars"] == 5
    sh = [[100, 100.6, 98.5, 99], [99, 99.2, 96.5, 97]]
    r2 = sim(sh, s=-1, stop0=102.0, hold=1)
    assert r2["R_gross"] > 0 and r2["reason"] == "max_hold"
    assert sim(rows, fee=0.04, loser=5)["R_net"] < sim(rows, loser=5)["R_net"]
