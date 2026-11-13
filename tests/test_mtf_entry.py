import numpy as np

from tradebot.strategies.mtf_entry import model as M
from tradebot.strategies.mtf_entry import paper as P


def test_uptrend_state_and_direction():
    c = 100 * np.exp(np.cumsum(np.linspace(0.001, 0.01, 300)))  # accelerating => PMO and MACD rising
    assert M.tf_state(c) == 1
    assert M.tf_state(1000 - np.cumsum(np.linspace(0.5, 5, 300))) == -1
    assert M.tf_state(c[:50]) == 0
    assert M.direction({tf: 1 for tf, _ in M.FRAMES})[0] == 1
    assert M.direction({tf: -1 for tf, _ in M.FRAMES})[0] == -1
    assert M.direction({"Day1": 1, "Hour4": -1})[0] == 0


def test_zone_mirrors_and_touch():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 0.3, 300)); h, l = c + 0.3, c - 0.3
    zl, zs = M.entry_zone(h, l, c, 1), M.entry_zone(h, l, c, -1)
    assert zl[0] < zl[1] and zs[0] < zs[1] and M.entry_zone(h, l, c, 0) is None
    assert M.touches(zl, zl[0] - 1, zl[0] + 0.01) and not M.touches(zl, zl[1] + 1, zl[1] + 2)


def test_paper_open_close_roundtrip():
    st = {"balance": 25.0, "open": [], "closed": [], "seen": {}}
    p = P.open_pos(st, "X_USDT", 1, 100.0, 0)
    assert abs(p["margin"] - 2.5) < 1e-9
    pnl = P.close_pos(st, p, 110.0, 1)
    assert pnl > 0 and not st["open"] and len(st["closed"]) == 1


def test_trailing_stop_only_ratchets_in_favour():
    assert M.trail_stop(1, 110.0, 2.0) == 104.0 and M.trail_stop(-1, 90.0, 2.0) == 96.0
    assert M.ratchet(1, 104.0, 102.0) == 104.0 and M.ratchet(1, 104.0, 106.0) == 106.0
    assert M.ratchet(-1, 96.0, 98.0) == 96.0 and M.ratchet(-1, 96.0, 94.0) == 94.0
    assert M.ratchet(1, None, 100.0) == 100.0


def test_turn_state_imminent_and_crossed():
    g = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5])            # closing at 0.5/bar -> crosses in ~1 bar
    assert M.turn_state(g, 1)[0] == "imminent" and M.turn_state(g, -1) is None
    assert M.turn_state(np.array([-1.0, -0.5, -0.2, 0.1, 0.4, 0.8]), 1) is None   # crossed 3 bars ago: late
    assert M.turn_state(np.array([-1.0, -0.5, -0.2, -0.1, 0.1, 0.4]), 1) == ("crossed", 1)
    assert M.turn_state(-g, -1)[0] == "imminent"


def _flat(n=120, price=100.0):
    rng = np.random.default_rng(0)
    c = price + rng.normal(0, 0.1, n); o = np.r_[c[0], c[:-1]]
    return o, c + 0.15, c - 0.15, c, np.full(n, 1000.0)


def test_breakout_up_and_down_with_volume():
    o, h, l, c, v = _flat()
    o[-1], c[-1], h[-1], l[-1], v[-1] = 100.0, 102.0, 102.1, 99.95, 6000.0     # strong up break on 6x volume
    r = M.breakout(o, h, l, c, v)
    assert r and r["dir"] == 1 and r["age"] == 0 and r["vol_x"] >= 4
    o, h, l, c, v = _flat()
    o[-1], c[-1], h[-1], l[-1], v[-1] = 100.0, 98.0, 100.05, 97.9, 6000.0
    assert M.breakout(o, h, l, c, v)["dir"] == -1


def test_breakout_needs_volume_range_and_not_late():
    o, h, l, c, v = _flat()
    o[-1], c[-1], h[-1], l[-1], v[-1] = 100.0, 102.0, 102.1, 99.95, 1500.0     # no volume spike
    assert M.breakout(o, h, l, c, v) is None
    o, h, l, c, v = _flat()
    o[-1], c[-1], h[-1], l[-1], v[-1] = 100.0, 102.0, 102.1, 99.95, 6000.0
    h[:60] += 20.0                                                               # range far too wide to be "quiet"
    assert M.breakout(o, h, l, c, v) is None
