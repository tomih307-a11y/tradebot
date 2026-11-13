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


def _squeezed(n=300, sign=1):
    rng = np.random.default_rng(3)
    sd = np.r_[np.full(n - 30, 1.0), np.full(30, 0.02)]
    c = 100 + rng.normal(0, 1, n) * sd
    o = np.r_[c[0], c[:-1]]
    h, l, v = np.maximum(o, c) + 0.02, np.minimum(o, c) - 0.02, np.full(n, 1000.0)
    o[-1], c[-1] = 100.0, 100.0 + 2.0 * sign
    h[-1], l[-1], v[-1] = max(o[-1], c[-1]) + 0.05, min(o[-1], c[-1]) - 0.05, 5000.0
    return o, h, l, c, v


def test_squeeze_breakout_both_sides_and_filters():
    r = M.squeeze_breakout(*_squeezed(sign=1))
    assert r and r["dir"] == 1 and r["age"] == 0 and r["squeeze_pct"] <= 20 and r["vol_x"] >= 3
    assert M.squeeze_breakout(*_squeezed(sign=-1))["dir"] == -1
    o, h, l, c, v = _squeezed()
    v[-1] = 1500.0                                              # no volume spike -> no signal
    assert M.squeeze_breakout(o, h, l, c, v) is None
    o, h, l, c, v = _squeezed()
    assert M.squeeze_breakout(o[:100], h[:100], l[:100], c[:100], v[:100]) is None   # too little history


def test_squeeze_not_triggered_when_range_was_not_narrow():
    rng = np.random.default_rng(4)
    n = 300
    c = 100 + rng.normal(0, 1, n); o = np.r_[c[0], c[:-1]]
    h, l, v = np.maximum(o, c) + 0.02, np.minimum(o, c) - 0.02, np.full(n, 1000.0)
    o[-1], c[-1], h[-1], l[-1], v[-1] = 100.0, 106.0, 106.1, 99.95, 6000.0
    assert M.squeeze_breakout(o, h, l, c, v) is None


def test_precomputed_squeeze_matches_single_call():
    o, h, l, c, v = _squeezed()
    pre = M.squeeze_precompute(o, h, l, c, v)
    s = M.squeeze_at(pre, o, h, l, c, v, len(c) - 1)
    r = M.squeeze_breakout(o, h, l, c, v)
    assert s and r and s["dir"] == r["dir"] and s["trend"] == r["trend"] and abs(s["vol_x"] - r["vol_x"]) < 1e-9


def test_event_merge_flags_and_label_no_lookahead():
    from tradebot.strategies.mtf_entry import boom_watch as B
    n = 3000
    c = np.full(n, 100.0); o = c.copy()
    for j in (500, 505, 2000):                                  # 500 & 505 same direction within 12h -> one event
        c[j] = 112.0
    ev = B.event_starts(o, c)
    assert [j for j, _ in ev] == [500, 2000]
    ev_idx = np.array([j for j, _ in ev])
    bw = np.full(n, 0.01)
    assert B.flags(ev_idx, 499, bw)["R3"] is False and B.flags(ev_idx, 520, bw)["R3"] is True       # event unknown before it starts
    assert B.label_and_lead(ev, 450)[0] is True and B.label_and_lead(ev, 450)[1] == 50
    assert B.label_and_lead(ev, 520)[0] is False


def test_early_signs_features_no_lookahead_and_labels():
    from tradebot.strategies.mtf_entry import early_signs as E
    n = 1500
    rng = np.random.default_rng(7)
    c = 100 + np.cumsum(rng.normal(0, 0.05, n)); o = np.r_[c[0], c[:-1]]
    h, l, v = np.maximum(o, c) + 0.02, np.minimum(o, c) - 0.02, np.full(n, 1000.0)
    t = 1_700_000_000 + 900 * np.arange(n)
    c[1200] = c[1199] * 1.15; o[1200] = c[1199]; h[1200] = c[1200] + 0.02; v[1200] = 9000.0     # explosive candle at 1200
    ft, fr = np.array([t[0] - 10.0]), np.array([-0.0002])
    basis = np.full(n, -0.002)
    f, fmax, valid = E.features(o, h, l, c, v, t, ft, fr, basis)
    assert not f["F3"][1199] and f["F3"][1200]                       # volume spike only AT the candle
    assert not f["F1"][1199] and f["F1"][1200] and f["F1"][1200 + 191] and not f["F1"][1200 + 192]   # recurrence flag lasts 48h exactly
    assert f["F6"][900] and f["F7"][900] and fmax[1199] > 14.9        # label sees the +15% next candle
    assert np.isnan(fmax[-1]) and valid[1100]


def test_tardis_day_arrays_and_signals_no_lookahead():
    from tradebot.strategies.mtf_entry import tardis_free as T
    day0 = 1_700_000_000_000_000 // 86_400_000_000 * 86_400_000_000
    rng = np.random.default_rng(11)
    trades, tick = [], []
    p = 10.0
    for m in range(1440):
        for s in range(6):
            ts = day0 + m * 60_000_000 + s * 9_000_000
            p = p * (1 + rng.normal(0, 0.0002))
            if 800 <= m < 830:                                    # a pump: price +, heavy buying
                p *= 1.004
            trades.append((ts, "buy" if (800 <= m < 830 or rng.random() < 0.5) else "sell", p, 100.0))
        tick.append((day0 + m * 60_000_000, 1_000_000.0 * (1.03 if m >= 700 else 1.0), -0.0001, 10.0, 10.0))
    d = T.day_arrays(trades, tick, 1.0, day0)
    assert d["cnt"][0] == 6 and d["buy"].sum() > 0 and not np.isnan(d["c"]).any()
    flags, fmax = T.signals_and_labels(d)
    n = len(fmax)
    assert n > 100 and all(len(v) == n for v in flags.values())
    assert flags["S5"].all()                                      # funding is -0.01% all day
    assert flags["S3"].any() and not flags["S3"][:5].any()        # OI jumped at minute 700 only
    assert (fmax >= 10).any() and np.nanmax(fmax) > 10            # pump creates +10% labels before it
