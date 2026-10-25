import numpy as np

from tradebot.config import load_config
from tradebot.strategies.multi_agent.smc import SMCAgent, fvgs, order_block, pivots, structure, untapped
from tradebot.strategies.multi_agent.types import Candidate

CFG = load_config("config.toml")


def ohlc(closes, pad=0.1):
    c = np.asarray(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    return o, np.maximum(o, c) + pad, np.minimum(o, c) - pad, c


def test_pivots_need_n_bars_each_side():
    h = np.array([1, 2, 5, 2, 1, 2, 3, 2, 1.0])
    l = h - 0.5
    hs, _ = pivots(h, l, 2)
    assert [i for i, _ in hs] == [2, 6]
    hs2, _ = pivots(h[:5], l[:5], 2)                # the swing at 2 is confirmed only because bars 3,4 exist
    assert [i for i, _ in hs2] == [2]
    hs3, _ = pivots(h[:4], l[:4], 2)                # one bar short -> unknown yet (no look-ahead)
    assert hs3 == []


def test_bos_then_choch():
    # up-leg breaks 103 (BOS up), then a drop closes below the swing low (CHoCH down)
    closes = [100, 101, 102, 103, 102, 101, 100, 99, 100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96]
    o, h, l, c = ohlc(closes)
    trend, ev = structure(c, h, l, 2)
    kinds = [(e["kind"], e["dir"]) for e in ev]
    assert kinds[0][1] == 1 and ("CHoCH", -1) in kinds and trend == -1


def test_order_block_is_last_down_candle_before_break():
    closes = [100, 101, 102, 103, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 104, 105]
    o, h, l, c = ohlc(closes)
    trend, ev = structure(c, h, l, 2)
    up = [e for e in ev if e["dir"] == 1][-1]
    lo, hi, j = order_block(up, o, h, l, c)
    assert c[j] < o[j] and j < up["i"] and j >= up["origin"]


def test_fvg_and_tap_rule():
    h = np.array([10, 11, 12, 15, 15.5, 16.0])
    l = np.array([9, 10, 9.5, 13, 13.5, 14.0])
    gaps = fvgs(h, l, 1, 0)
    assert gaps[0] == (11.0, 13.0, 3)                                    # bar3 low 13 above bar1 high 11
    c = (h + l) / 2
    assert untapped((11.0, 13.0), 1, 3, h, l, c)                          # never revisited
    l2 = l.copy(); l2[4] = 12.5                                          # bar 4 traded back into the gap
    assert not untapped((11.0, 13.0), 1, 3, h, l2, c)


def _series():
    # 1H: low -> up-break (BOS) -> long run -> retrace to just above the order block
    closes = [100, 101, 102, 103, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 108.5, 107, 105.5, 104.5]
    return ohlc(closes)


def _agent():
    a = SMCAgent(CFG)
    a.c = CFG.agents.smc.override(min_bars=10, swing_len=2, max_dist_atr=50, min_score=0.3, event_max_age=40)
    return a


def kl(o, h, l, c):
    return {"time": np.arange(len(c)) * 3600, "open": o, "high": h, "low": l, "close": c}


def test_analyze_long_setup_levels():
    o, h, l, c = _series()
    up4 = ohlc([100, 101, 102, 103, 102, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
    r = _agent().analyze(Candidate("X_USDT", float(c[-1]), 1e9, 1, 1), kl(o, h, l, c), kl(*up4), 0)
    assert r is not None and r.side == "long" and r.retrace
    lo, hi = r.zone
    assert r.stop < lo and all(tp > hi for tp in r.tps)
    risk = hi - r.stop
    assert abs((r.tps[1] - hi) - 2 * risk) < 1e-9                          # tp_r = [1,2,3] R from the zone's first-touch edge


def test_analyze_rejects_when_htf_disagrees_and_when_zone_used():
    o, h, l, c = _series()
    down4 = ohlc([120 - i for i in range(24)])
    a = _agent()
    assert a.analyze(Candidate("X_USDT", float(c[-1]), 1e9, 1, 1), kl(o, h, l, c), kl(*down4), 0) is None
    # price already traded down through the order block earlier -> POI consumed -> no setup
    closes = list(c) + [97, 96, 95, 96, 97, 98, 99, 100, 101, 102, 103]
    o2, h2, l2, c2 = ohlc(closes)
    up4 = ohlc([100 + i for i in range(30)])
    r = a.analyze(Candidate("X_USDT", float(c2[-1]), 1e9, 1, 1), kl(o2, h2, l2, c2), kl(*up4), 0)
    assert r is None or r.side != "long" or r.zone[0] > 98.5               # the consumed block must not be reused
