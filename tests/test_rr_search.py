import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rr_search import BARS, bracket, breakout_signals, cooldown, day_boot, rsi_signals, touch_signals   # noqa: E402


def scalar(s, j, o, h, l, c, atr, k, rr, fee, slip):
    j0 = j + 1
    px = o[j0] * (1 + s * slip / 100)
    d = max(k * atr, 0.003 * px)
    stop, tgt = px - s * d, px + s * rr * d
    for t in range(BARS):
        i = j0 + t
        adverse, favour = (l[i], h[i]) if s == 1 else (h[i], l[i])
        if s * (adverse - stop) <= 0:                                  # adverse first (also when the target is touched in the same candle)
            ex = o[i] if (t > 0 and s * (o[i] - stop) <= 0) else stop
            ex *= (1 - s * slip / 100)
            break
        if s * (favour - tgt) >= 0:
            ex = tgt
            break
    else:
        ex = c[j0 + BARS - 1] * (1 - s * slip / 100)
    return s * (ex - px) / d - fee / 100 * (px + ex) / d


def test_vectorised_bracket_equals_the_scalar_reference():
    rng = np.random.default_rng(5)
    n = 1500
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    h, l = np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998
    j = np.arange(50, n - BARS - 5, 37)
    atr = np.full(len(j), 0.8)
    for s in (1, -1):
        for k, rr in ((1.0, 0.5), (2.0, 1.5), (1.0, 3.0)):
            R, out, fees = bracket(s, j, o, h, l, c, atr, k, rr, 0.04, 0.05)
            ref = np.array([scalar(s, int(jj), o, h, l, c, 0.8, k, rr, 0.04, 0.05) for jj in j])
            assert np.allclose(R, ref, atol=1e-9), (s, k, rr, np.abs(R - ref).max())
            assert set(np.unique(out)) <= {0, 1, 2}


def test_breakeven_identity_for_a_driftless_market_after_costs_is_negative():
    rng = np.random.default_rng(8)
    n = 60000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    h, l = np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999
    j = np.arange(100, n - BARS - 5, 40)
    atr = np.full(len(j), 100 * 0.004 * 4)
    R, out, fees = bracket(1, j, o, h, l, c, atr, 1.0, 1.0, 0.04, 0.05)
    assert R.mean() < 0.02 and fees.mean() > 0          # no edge on a random walk: expectancy cannot be clearly positive


def test_signal_helpers_and_cooldown():
    p = np.array([np.nan, -1, 0.5, 1, -0.2, 0.1], float)
    assert list(touch_signals(p, 1)) == [2, 5] and list(touch_signals(p, -1)) == [4]
    assert list(cooldown(np.array([5, 50, 120, 130, 300]), 96)) == [5, 120, 300]
    close = np.concatenate([np.linspace(100, 101, BARS), [102.0]])
    high, low = close + 0.1, close - 0.1
    assert list(breakout_signals(close, high, low, 1)) == [BARS]
    r = np.array([np.nan, 30, 24, 26, 40], float)
    assert list(rsi_signals(r, 1)) == [3]


def test_day_boot_brackets_the_mean():
    rng = np.random.default_rng(2)
    day = np.repeat(np.arange(60), 20)
    R = rng.normal(0.1, 1.0, len(day))
    lo, hi = day_boot(day, R)
    assert lo < R.mean() < hi


def test_required_target_rate_zeroes_the_expectancy():
    from rr_search import stat_block
    rng = np.random.default_rng(3)
    out = rng.choice([0, 1, 2], size=4000, p=[0.2, 0.5, 0.3])
    R = np.where(out == 2, 1.9, np.where(out == 1, -1.05, rng.normal(-0.1, 0.4, 4000)))
    T = np.arange(4000) * 3600
    b = stat_block(T, R, out)
    # plug the 'required' hit rate back in: the mixture expectancy must be zero
    exp0 = b["required_target"] * b["r_target"] + (1 - b["required_target"]) * b["r_other"]
    assert abs(exp0) < 1e-9 and 0 < b["required_target"] < 1
    assert abs(b["p_target"] * b["r_target"] + (1 - b["p_target"]) * b["r_other"] - b["exp"]) < 1e-9


def test_degenerate_atr_cannot_blow_up_r():
    rng = np.random.default_rng(4)
    n = 600
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    h, l = np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999
    j = np.arange(20, n - BARS - 5, 11)
    R, out, fees = bracket(1, j, o, h, l, c, np.full(len(j), 1e-12), 1.0, 1.0, 0.04, 0.05)
    assert np.isfinite(R).all() and np.abs(R).max() < 50          # the floor keeps the stop distance at >= 0.3% of price
