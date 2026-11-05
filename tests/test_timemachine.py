import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from timemachine import find_signals, mc_paths, slot_candidates, swing_stop, verdict   # noqa: E402
from rules_study import simulate_open                                                   # noqa: E402


def test_find_signals_needs_cross_above_zero_and_htf_agreement():
    nan = np.nan
    p = np.array([nan, 0.5, 0.8, 1.2, 1.1], float)
    g = np.array([nan, 0.6, 0.9, 1.0, 1.15], float)             # PMO crosses above signal at bar 3, both > 0
    ph = np.array([nan, 1, 1, 1, 1], float)
    assert list(find_signals(p, g, ph, 1)) == [3]
    assert list(find_signals(p, g, -ph, 1)) == []                 # higher timeframe disagrees
    p2, g2 = np.array([nan, -0.5, -0.3, 0.05, 0.1]), np.array([nan, -0.4, -0.2, -0.1, 0.0])
    assert list(find_signals(p2, g2, ph, 1)) == []                # the cross happened with the signal still below zero
    assert list(find_signals(-p, -g, -ph, -1)) == [3]             # shorts are the exact mirror


def test_swing_stop_and_minimum_distance():
    low = np.array([10, 9.5, 9.8, 10.2, 10.4]); high = low + 0.3
    assert abs(swing_stop(1, low, high, 10.4, 4, 0.1, 4, 0.0, 0.0) - 9.5) < 1e-9
    assert abs(swing_stop(1, low, high, 10.4, 4, 0.1, 4, 1.0, 0.0) - 9.4) < 1e-9          # buffer in ATRs
    tight = swing_stop(1, np.array([10.39, 10.4]), np.array([10.5, 10.5]), 10.4, 1, 0.0, 2, 0.0, 1.0)
    assert abs((10.4 - tight) / 10.4 - 0.01) < 1e-9                                         # never closer than min_risk_pct


def _candles(entry, n, ratios):
    o, h, l, c = [], [], [], []
    pc = entry
    for _ in range(n):
        ro, rh, rl, rc = ratios
        o.append(pc * ro); h.append(pc * rh); l.append(pc * rl); c.append(pc * rc); pc = pc * rc
    return np.array(o), np.array(h), np.array(l), np.array(c)


def _cmp(ratios, s, stop0, **kw):
    entry, T, atr_pct = 100.0, 120, 0.01
    o, h, l, c = _candles(entry, T, ratios)
    ref = simulate_open(s, stop0, o, h, l, c, atr_pct * c, 0, kw.get("trail", 2.0), kw.get("act", 1.0), T - 1, kw.get("loser", 96), 0.04, 0.0)
    Ro, Rh, Rl, Rc = (np.array([x]) for x in ratios)
    cand = [np.array([0])] * 96
    R, G, rsn = mc_paths(s, entry, stop0, (Ro, Rh, Rl, Rc), cand, 0, 50, T, kw.get("trail", 2.0), kw.get("act", 1.0), atr_pct, kw.get("loser", 96),
                         0.04, 0.0, np.random.default_rng(1))
    return ref, R, rsn


def test_monte_carlo_matches_the_single_path_simulator_on_deterministic_paths():
    ref, R, rsn = _cmp((1.0, 1.006, 0.999, 1.005), 1, 98.0)                 # steady climb: trailing stop eventually / max hold
    assert np.allclose(R, ref["R_net"], atol=1e-9)
    ref2, R2, rsn2 = _cmp((1.0, 1.001, 0.985, 0.99), 1, 98.0)               # falling: the initial stop is hit on the first bar
    assert np.allclose(R2, ref2["R_net"], atol=1e-9) and ref2["reason"] == "stop" and (rsn2 == 1).all()
    ref3, R3, _ = _cmp((1.0, 1.004, 0.996, 0.9995), 1, 90.0, loser=10)      # drifting slightly down: cut as an unprofitable trade
    assert np.allclose(R3, ref3["R_net"], atol=1e-9) and ref3["reason"] == "time_loser"
    ref4, R4, _ = _cmp((1.0, 1.001, 0.994, 1.004), -1, 102.0)               # short mirror
    assert np.allclose(R4, ref4["R_net"], atol=1e-9)


def test_slots_wrap_around_midnight():
    t = np.arange(0, 96 * 3) * 900
    cand = slot_candidates(t)
    assert len(cand) == 96 and set((np.asarray(t)[cand[0]] // 900) % 96) == {94, 95, 0, 1, 2}


def test_verdict_thresholds():
    assert "قليل" in verdict(10, 0.5, 0.1, 0.9, 0.01)
    assert "تتفوق" in verdict(200, 0.2, 0.05, 0.35, 0.01)
    assert "خسرت" in verdict(200, -0.2, -0.35, -0.05, 0.9)
    assert "رمية عملة" in verdict(200, 0.01, -0.1, 0.12, 0.4)
    assert "غير حاسمة" in verdict(200, 0.05, -0.05, 0.15, 0.04)
