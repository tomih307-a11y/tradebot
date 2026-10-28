import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gallery import boot, touches   # noqa: E402


def test_touches_are_first_bars_reaching_zero():
    p = np.array([np.nan, -2, -1, 0.0, 1, 2, 1, 0.0, -1, -2, -1, 0.5], float)
    up, dn = touches(p)
    assert list(up) == [3, 11] and list(dn) == [7]


def test_day_block_bootstrap_brackets_the_mean_and_is_symmetric_under_noise():
    rng = np.random.default_rng(1)
    by_day = {d: list(rng.normal(0.01, 0.05, 20)) for d in range(40)}
    lo, hi = boot(by_day)
    m = np.mean([v for x in by_day.values() for v in x])
    assert lo < m < hi
