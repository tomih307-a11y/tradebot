import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from xsec_funding import DAY, funding_drag, funding_sums   # noqa: E402


def test_funding_sums_window_and_coverage_rules():
    ts = np.arange(0, 20 * DAY, DAY // 3, dtype=float)
    rates = {0: (ts, np.full(len(ts), 0.0001))}
    s = funding_sums(None, rates, [0, 1], 5 * DAY, 12 * DAY)            # 7 days x 3 settlements per day
    assert abs(s[0] - 21 * 0.0001) < 1e-12 and np.isnan(s[1])         # contract 1 has no funding history -> NaN, never silently zero
    assert np.isnan(funding_sums(None, rates, [0], 25 * DAY, 30 * DAY)[0])    # history does not cover the window


def test_longs_pay_and_shorts_receive_funding():
    n_days, n = 60, 40
    dates = np.arange(n_days) * DAY
    rng = np.random.default_rng(1)
    C = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n)), axis=0))
    O = np.vstack([C[:1], C[:-1]])
    turn = np.full_like(C, 1e7)
    score = np.tile(np.arange(n, dtype=float), (n_days, 1))              # asset j always ranks j: top = last columns (long), bottom = first (short)
    ts = np.arange(0, 70 * DAY, DAY // 3, dtype=float)
    # every contract pays the same positive rate -> longs and shorts cancel exactly
    flat = {j: (ts, np.full(len(ts), 0.0002)) for j in range(n)}
    d0 = funding_drag(score, turn, O, dates, flat, 7, n, 5, 40)
    assert abs(d0["funding_pct"]) < 1e-9
    # winners (long leg) pay a higher rate than losers: funding must be a COST (negative PnL)
    skew = {j: (ts, np.full(len(ts), 0.0001 + 0.00001 * j)) for j in range(n)}
    d1 = funding_drag(score, turn, O, dates, skew, 7, n, 5, 40)
    assert d1["funding_pct"] < 0
