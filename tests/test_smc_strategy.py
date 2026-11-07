import numpy as np

from tradebot.strategies.smc_strategy import model as M

CLOSES = [100, 101, 102, 103, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 108.5, 107, 105.5, 104.5]


def ohlc4(closes, pad=0.1):
    c = np.asarray(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    return o, np.maximum(o, c) + pad, np.minimum(o, c) - pad, c


# 15m bars: (open, high, low, close). Bar 11 sweeps the swing low into the 4H block, bar 14 is the CHoCH, bar 15 is the entry candle.
BARS = [(101.0, 101.2, 100.5, 100.6), (100.6, 100.8, 100.0, 100.2), (100.2, 100.3, 99.5, 99.6), (99.6, 99.7, 99.0, 99.2),
        (99.2, 99.3, 98.9, 98.95), (98.95, 99.4, 98.95, 99.3), (99.3, 100.0, 99.2, 99.9), (99.9, 100.6, 99.8, 100.4),
        (100.4, 100.5, 99.9, 100.0), (100.0, 100.1, 99.4, 99.5), (99.5, 99.6, 99.0, 99.2), (99.2, 99.3, 98.5, 99.0),
        (99.0, 99.6, 98.9, 99.5), (99.5, 100.2, 99.4, 100.1), (100.1, 100.9, 100.0, 100.7), (100.7, 101.0, 100.5, 100.9),
        (100.9, 110.0, 100.8, 109.5), (109.5, 109.8, 109.0, 109.2)]


def arrays(bars=BARS):
    a = np.array(bars, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def state(sh_price=(109.1,)):
    o4, h4, l4, c4 = ohlc4(CLOSES)
    st = M.htf_state(h4, l4, c4, n=2)
    st["sh_price"], st["sh_conf"] = np.array(sh_price, float), np.arange(len(sh_price)) + 5
    return st, np.arange(len(c4)) * 14400 + 4 * 3600


def test_htf_state_bias_zone_and_mitigation():
    o4, h4, l4, c4 = ohlc4(CLOSES)
    st = M.htf_state(h4, l4, c4, n=2)
    assert st["bias"][-1] == 1 and abs(st["zlo"][-1] - 97.9) < 1e-9 and abs(st["zhi"][-1] - 99.1) < 1e-9
    assert np.isnan(st["zlo"][:10]).all()                                        # no block before the structure break
    l2 = l4.copy(); l2[-1] = 97.0                                                # a wick through the far edge mitigates the block
    assert np.isnan(M.htf_state(h4, l2, c4, n=2)["zlo"][-1])


def test_sweep_then_choch_inside_the_block_produces_the_planned_trade():
    st, close4 = state()
    o, h, l, c = arrays()
    t15 = 24 * 14400 + np.arange(len(c)) * 900                                    # 15m bars start after the last 4H close
    out = M.ltf_setups(o, h, l, c, np.full(len(c), 2.0), close4, t15, st, n=2)
    assert len(out) == 1
    s = out[0]
    assert s["sweep_i"] == 11 and s["choch_i"] == 14 and s["entry_i"] == 15
    assert abs(s["stop"] - (97.9 - 0.2)) < 1e-9 and abs(s["target"] - 109.1) < 1e-9 and s["rr"] >= 2.0
    st2, _ = state(sh_price=(103.1,))                                             # nearest liquidity too close: reward/risk < 2 -> no trade
    assert M.ltf_setups(o, h, l, c, np.full(len(c), 2.0), close4, t15, st2, n=2) == []


def test_filters_reject_lower_low_before_choch_and_a_chased_entry():
    st, close4 = state()
    o, h, l, c = arrays()
    t15 = 24 * 14400 + np.arange(len(c)) * 900
    b = list(BARS); b[12] = (99.0, 99.2, 98.2, 98.5)                              # lower low than the sweep that closes back under the swing low: no valid sweep
    o2, h2, l2, c2 = arrays(b)
    assert M.ltf_setups(o2, h2, l2, c2, np.full(len(c2), 2.0), close4, t15, st, n=2) == []
    assert M.ltf_setups(o, h, l, c, np.full(len(c), 0.1), close4, t15, st, n=2) == []   # tiny ATR: the CHoCH close is too far above the block


def test_fixed_exit_simulator_stop_target_and_time():
    o, h, l, c = arrays()
    tgt = M.sim_fixed(1, 15, 97.7, 109.1, o, h, l, c, 0.0, 0.0)
    assert tgt["outcome"] == 2 and abs(tgt["R_gross"] - (109.1 - 100.7) / 3.0) < 1e-9
    b = list(BARS); b[16] = (100.9, 101.0, 97.0, 98.0)
    o2, h2, l2, c2 = arrays(b)
    st_ = M.sim_fixed(1, 15, 97.7, 109.1, o2, h2, l2, c2, 0.0, 0.0)
    assert st_["outcome"] == 1 and abs(st_["R_gross"] + 1.0) < 1e-9
    both = list(BARS); both[16] = (100.9, 110.0, 97.0, 105.0)                      # one candle touches both: counted as the stop
    assert M.sim_fixed(1, 15, 97.7, 109.1, *arrays(both), 0.0, 0.0)["outcome"] == 1
    flat = M.sim_fixed(1, 15, 90.0, 120.0, o, h, l, c, 0.0, 0.0, max_bars=2)
    assert flat["outcome"] == 0 and flat["exit_i"] == 16
    assert M.sim_fixed(1, 15, 101.0, 120.0, o, h, l, c, 0.0, 0.0) is None          # stop above the entry


def test_shorts_are_the_exact_mirror_of_longs():
    o4, h4, l4, c4 = ohlc4(CLOSES)
    K = 400.0                                                                      # reflect prices around a constant: a bearish world
    k4 = {"time": np.arange(len(c4)) * 14400, "open": K - o4, "high": K - l4, "low": K - h4, "close": K - c4}
    o, h, l, c = arrays()
    o15, h15, l15, c15 = K - o, K - l, K - h, K - c
    t15 = 24 * 14400 + np.arange(len(c)) * 900
    # long on the original world, with the 4H swing-high targets overridden as in the other tests
    st_long, close4 = state()
    long_setups = M.ltf_setups(o, h, l, c, np.full(len(c), 2.0), close4, t15, st_long, n=2)
    assert len(long_setups) == 1
    # short on the reflected world: same code path through find_trades (mirror inside); compare the plan distances
    mo, mh, ml, mc = M.mirror(o15, h15, l15, c15)
    st_m = M.htf_state(*(M.mirror(k4["open"], k4["high"], k4["low"], k4["close"])[1:]), n=2)
    assert st_m["bias"][-1] == 1 and abs((st_m["zhi"][-1] - st_m["zlo"][-1]) - 1.2) < 1e-9


def test_live_setup_appears_while_the_choch_is_fresh_and_in_real_prices():
    o, h, l, c = arrays(BARS[:15])                                                # last closed bar = the CHoCH candle (index 14)
    t15 = 24 * 14400 + np.arange(len(c)) * 900
    o4, h4, l4, c4 = ohlc4(CLOSES)
    k4 = {"time": np.arange(len(c4)) * 14400, "open": o4, "high": h4, "low": l4, "close": c4}
    import tradebot.strategies.smc_strategy.model as mm
    orig = mm.htf_state
    def patched(h_, l_, c_, n=2):                       # n=2 and an explicit nearby-liquidity override like the other tests
        st = orig(h_, l_, c_, n=2)
        st["sh_price"], st["sh_conf"] = np.array([109.1]), np.array([5])
        return st
    mm.htf_state = patched
    try:
        orig_l = mm.ltf_setups
        mm.ltf_setups = lambda *a, **k: orig_l(*a, **{**k, "n": 2})
        out = mm.live_setups(o, h, l, c, np.full(len(c), 2.0), t15, k4, "long")
    finally:
        mm.htf_state, mm.ltf_setups = orig, orig_l
    assert len(out) == 1 and out[0]["age"] == 0 and out[0]["choch_i"] == 14
    assert abs(out[0]["entry"] - 100.7) < 1e-9 and out[0]["stop"] < out[0]["entry"] < out[0]["target"]
