import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from xsec import block, build_matrices, features, period_returns   # noqa: E402

DAY = 86400


def panel(mu, n_days=420, n=60, vol=0.04, seed=0):
    rng = np.random.default_rng(seed)
    ret = mu[None, :] + rng.normal(0, vol, (n_days, n))
    C = 100 * np.exp(np.cumsum(ret, axis=0))
    O = np.vstack([C[:1], C[:-1]])                       # open = previous close
    A = np.full_like(C, 1e7)
    dates = np.arange(n_days) * DAY
    return dates, O, C, A


def run(mu, H=3, cost=0.0, sig="mom30", seed=0):
    dates, O, C, A = panel(mu, seed=seed)
    f, t30 = features(C, A)
    _, r = period_returns(f[sig], t30, O, dates, H, 60, cost, 40, len(dates))
    return r


def test_no_edge_gives_small_t_and_planted_momentum_is_detected():
    n = 60
    null = run(np.zeros(n), seed=1)
    assert abs(block(null)["t"]) < 3.0
    rng = np.random.default_rng(9)
    mu = rng.normal(0, 0.004, n)                          # persistent cross-sectional drift: momentum must pick it up
    real = run(mu, seed=1)
    assert block(real)["exp_pct"] > 0 and block(real)["t"] > 4.0


def test_costs_reduce_the_return_one_for_one_and_required_win_rate_formula():
    n = 60
    r0 = run(np.random.default_rng(2).normal(0, 0.003, n), cost=0.0)
    r1 = run(np.random.default_rng(2).normal(0, 0.003, n), cost=0.0018)
    assert np.allclose(r0 - r1, 0.0018)
    b = block(r1)
    assert abs(b["required_win"] - 1 / (1 + b["payoff"])) < 1e-12


def test_rev1_is_the_negative_of_the_one_day_return_and_matrices_align():
    t0 = 100 * DAY
    series = {"A": {"time": np.array([t0, t0 + DAY, t0 + 2 * DAY]), "open": np.array([1., 2, 3]), "close": np.array([2., 3, 4]), "amount": np.ones(3)},
              "B": {"time": np.array([t0 + DAY, t0 + 2 * DAY]), "open": np.array([5., 6]), "close": np.array([6., 7]), "amount": np.ones(2)}}
    dates, O, C, A = build_matrices(series, t0, t0 + 2 * DAY)
    assert O.shape == (3, 2) and np.isnan(O[0, 1]) and O[2, 1] == 6.0
    f, _ = features(np.array([[1.0], [1.1], [0.99]]), np.ones((3, 1)))
    assert np.allclose(f["rev1"][1:, 0], [-(0.1), -(0.99 / 1.1 - 1)])
