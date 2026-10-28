import numpy as np
import pytest

from tradebot.config import load_config
from tradebot.strategies.multi_agent.rules import RulesAgent
from tradebot.strategies.multi_agent.types import Candidate

CFG = load_config("config.toml")


def ohlc(closes, pad=0.1):
    c = np.asarray(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    return {"time": np.arange(len(c)) * 14400, "open": o, "high": np.maximum(o, c) + pad, "low": np.minimum(o, c) - pad, "close": c}


# low -> up-break (BOS) -> run -> retrace; the bullish OB is the leg's lowest candle (close 98 -> low 97.9, high 99.1)
CLOSES = [100, 101, 102, 103, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 108.5, 107, 105.5, 104.5]


def agent(**kw):
    a = RulesAgent(CFG)
    base = dict(min_bars=10, swing_len=2, conditions=["price_in_ob_4h"])
    base.update(kw)
    a.c = CFG.agents.rules.override(**base)
    return a


def run(a, price, closes=CLOSES):
    k = ohlc(closes)
    return a.analyze(Candidate("X_USDT", price, 1e9, 0, 1), k, k, 0)


def test_inside_bullish_ob_gives_long_with_levels():
    r = run(agent(), 98.5)
    assert r and r.side == "long" and r.retrace
    lo, hi = r.zone
    assert lo <= 98.5 <= hi and r.stop < lo and all(t > hi for t in r.tps)
    assert r.tps == []                                                       # open_target = true: no fixed targets
    assert any("OB" in n for n in r.reasons)


def test_price_outside_the_ob_fails_the_condition():
    assert run(agent(), 104.0) is None         # above the block
    assert run(agent(), 96.0) is None          # below the block


def test_wick_beyond_far_edge_mitigates_but_none_mode_keeps_it():
    swept = CLOSES + [100, 101]                # later bars...
    k = ohlc(swept)
    k["low"][-2] = 97.0                         # ...one wick below the OB's far edge (97.9)
    a = agent()
    assert a.analyze(Candidate("X_USDT", 98.5, 1e9, 0, 1), k, k, 0) is None
    a2 = agent(mitigation="none")
    assert a2.analyze(Candidate("X_USDT", 98.5, 1e9, 0, 1), k, k, 0) is not None


def test_all_conditions_must_pass_and_unknown_condition_is_an_error():
    a = agent(conditions=["price_in_ob_4h", "price_in_ob_4h"])
    assert run(a, 98.5) is not None and run(a, 104.0) is None
    with pytest.raises(KeyError):
        run(agent(conditions=["no_such_condition"]), 98.5)


def test_open_target_false_uses_r_multiples():
    r = run(agent(open_target=False), 98.5)
    lo, hi = r.zone
    assert (r.tps[1] - hi) == pytest.approx(2 * (hi - r.stop))


# ───────────── condition 2: 15m PMO crosses its signal, then FIRST touch of the zero line ─────────────
from tradebot.strategies.multi_agent import rules as rules_mod            # noqa: E402
from tradebot.strategies.multi_agent.rules import signal_cross_then_zero_touch as sczt   # noqa: E402

NAN = np.nan


def test_long_trigger_on_first_zero_touch_after_signal_cross():
    p = np.array([NAN, -3, -2.5, -2, -1.5, -0.8, 0.3], float)
    g = np.array([NAN, -2, -2.2, -2.1, -1.7, -1.2, -0.5], float)      # PMO crosses above signal at idx3 (still < 0), touches 0 at idx6
    assert sczt(p, g, 1, 50, 1) == (3, 6)
    assert sczt(p[:-1], g[:-1], 1, 50, 1) is None                       # no touch yet -> no recommendation
    p2 = np.append(p, 0.9); g2 = np.append(g, 0.0)
    assert sczt(p2, g2, 1, 50, 1) is None                               # touch is one bar old and window=1 -> too late
    assert sczt(p2, g2, 1, 50, 2) == (3, 6)


def test_not_a_first_touch_or_no_prior_signal_cross():
    already_up = np.array([NAN, -3, -1, 0.5, -0.4, 0.2], float)         # touched zero at idx3 already; idx5 is a re-touch
    sg = np.array([NAN, -2, -1.5, -0.5, -0.3, -0.1], float)
    assert sczt(already_up, sg, 1, 50, 1) is None or sczt(already_up, sg, 1, 50, 1)[1] != 3
    no_cross = np.array([NAN, -3, -2.5, -1.5, -0.8, 0.3], float)
    below_sig = no_cross - 1.0                                         # PMO always BELOW its signal: never crossed up
    assert sczt(no_cross, below_sig + 2.5 + 0.0 * no_cross, 1, 50, 1) is None
    stale = np.array([NAN, -3, -2.5, -2, -1.5, -0.8, 0.3], float)
    assert sczt(stale, np.array([NAN, -2, -2.2, -2.1, -1.7, -1.2, -0.5], float), 1, 2, 1) is None   # cross older than max age


def test_short_is_the_mirror():
    p = -np.array([NAN, -3, -2.5, -2, -1.5, -0.8, 0.3], float)
    g = -np.array([NAN, -2, -2.2, -2.1, -1.7, -1.2, -0.5], float)
    assert sczt(p, g, -1, 50, 1) == (3, 6) and sczt(p, g, 1, 50, 1) is None


def test_engine_requires_both_conditions_and_skips_1m_confirmation(monkeypatch):
    a = agent(conditions=["price_in_ob_4h", "pmo15_zero_touch"], min_bars_15m=5)
    k = ohlc(CLOSES)
    pm = np.array([NAN] * 300 + [-3, -2.5, -2, -1.5, -0.8, 0.3], float)
    sg = np.array([NAN] * 300 + [-2, -2.2, -2.1, -1.7, -1.2, -0.5], float)
    k15 = ohlc(np.linspace(100, 101, len(pm)))
    loaded = []
    monkeypatch.setattr(rules_mod, "pmo", lambda *a_, **k_: (pm, sg, pm))
    r = a.analyze(Candidate("X_USDT", 98.5, 1e9, 0, 1), k, k, 0, lambda: (loaded.append(1), k15)[1])
    assert r and r.side == "long" and r.confirm is False and r.tps == [] and len(r.reasons) == 2 and loaded
    # price outside the OB: condition 1 fails and the 15m data is never even loaded
    loaded.clear()
    assert a.analyze(Candidate("X_USDT", 104.0, 1e9, 0, 1), k, k, 0, lambda: (loaded.append(1), k15)[1]) is None and not loaded
    flat = np.array([NAN] * 300 + [-2.0] * 6)
    monkeypatch.setattr(rules_mod, "pmo", lambda *a_, **k_: (flat, flat, flat))
    assert a.analyze(Candidate("X_USDT", 98.5, 1e9, 0, 1), k, k, 0, k15) is None            # PMO condition fails
    assert a.analyze(Candidate("X_USDT", 98.5, 1e9, 0, 1), k, k, 0, None) is None           # no 15m data -> fail closed


# ───────────── open-ended trade: no targets, trailing stop ─────────────
from tradebot.strategies.multi_agent.exit import ExitAgent                # noqa: E402
from tradebot.strategies.multi_agent.entry import EntryAgent              # noqa: E402
from tradebot.strategies.multi_agent.types import Recommendation          # noqa: E402
from tradebot.strategies.telegram_signals import paper                    # noqa: E402
from tradebot.strategies.telegram_signals.parser import Signal            # noqa: E402


def open_sig():
    return Signal(-1, 0, "X_USDT", "long", 100.0, 98.0, [], None)


def test_open_trade_has_no_targets_and_exits_only_by_stop():
    p = paper.open_position(open_sig(), 100.0, 25.0, 10, 5, 0, "exchange", 0.04, 0.005)
    cfg_ts = {"fee_pct": 0.04, "slip_pct": 0.0, "breakeven_after_tp1": True}
    for i, (h, l) in enumerate([(104, 99.5), (110, 108), (131, 129)]):          # runs far up: nothing closes it
        paper.step(p, (60 * (i + 1), 100, h, l, h), cfg_ts)
    assert not p.closed and p.remaining == 1.0 and p.tp_hit == 0
    p.stop = 120.0                                                              # the trailing stop has been tightened
    paper.step(p, (400, 130, 130, 119, 120), cfg_ts)
    assert p.closed and p.exit_reason == "stop" and p.realized > 0


def test_exit_agent_trails_open_trade_after_activation():
    ex = ExitAgent(CFG)
    p = paper.open_position(open_sig(), 100.0, 25.0, 10, 5, 0, "exchange", 0.04, 0.005)
    assert ex.manage(p, 101.0, 1.0, 0, 100).new_stop is None                    # +0.5R: not activated yet (activation 1R)
    a = ex.manage(p, 103.0, 1.0, 0, 100)                                        # +1.5R -> trail 2 ATR below
    assert a.new_stop == pytest.approx(103.0 - 2.0)
    p.stop = a.new_stop
    assert ex.manage(p, 102.0, 1.0, 0, 100).new_stop is None                    # never loosens


def test_entry_is_immediate_when_trigger_fired_and_risk_vote_ok_without_targets():
    r = run(agent(), 98.5)
    assert EntryAgent(CFG).plan(r, 98.5, {"open": np.array([1.0]), "close": np.array([0.5])}).action == "wait"   # OB alone: still needs 1m confirmation
    r.confirm = False                                                            # what the 15m PMO condition sets
    plan = EntryAgent(CFG).plan(r, 98.5, {"open": np.array([1.0]), "close": np.array([0.5])})   # last 1m candle even red
    assert plan.action == "market"
    assert ExitAgent(CFG).assess(r).veto is False and ExitAgent(CFG).assess(r).score == 1.0


# ───────────── alternative condition 2: cross above zero on 15m + higher timeframe above zero ─────────────
from tradebot.strategies.multi_agent.rules import cross_above_zero       # noqa: E402


def test_cross_above_zero_requires_both_lines_above_zero_and_a_fresh_cross():
    p = np.array([NAN, 0.5, 0.8, 1.2], float)
    g = np.array([NAN, 0.6, 0.9, 1.0], float)                  # PMO crosses above signal on the last bar, both > 0
    assert cross_above_zero(p, g, 1, 1) == 3
    assert cross_above_zero(p[:-1], g[:-1], 1, 1) is None      # no cross yet
    below = np.array([NAN, -0.5, -0.3, 0.05], float)
    gb = np.array([NAN, -0.4, -0.2, -0.1], float)              # crosses up but the signal is still below zero -> not "above zero"
    assert cross_above_zero(below, gb, 1, 1) is None
    neg = np.array([NAN, -0.5, -0.8, -1.2], float)
    gn = np.array([NAN, -0.6, -0.9, -1.0], float)              # mirror: crosses DOWN below zero -> short
    assert cross_above_zero(neg, gn, -1, 1) == 3 and cross_above_zero(neg, gn, 1, 1) is None
    assert cross_above_zero(p, g, 1, 0) is None                # window 0 -> nothing is recent enough


def _aligned_agent(**kw):
    a = agent(conditions=["price_in_ob_4h", "pmo_cross_zero_aligned"], min_bars_15m=5, htf_min_bars=5, htf="Min60", **kw)
    a.ind = CFG.pmo_zero_cross.indicator.override(warmup_bars=0)          # the synthetic series are short: no warm-up masking
    return a


def test_aligned_condition_needs_htf_and_15m_to_agree(monkeypatch):
    k = ohlc(CLOSES)
    n15, nh = 50, 40
    p15 = np.array([NAN] * 40 + [0.5, 0.8, 1.2, 1.1, 0.9, 0.7, 0.6, 0.55, 0.5, 0.45], float)
    s15 = np.array([NAN] * 40 + [0.6, 0.9, 1.0, 1.15, 0.95, 0.75, 0.62, 0.56, 0.51, 0.44], float)        # cross up on the last bar
    htf_pos = np.array([NAN] * 30 + [1.0] * 10, float)
    k15, kh = ohlc(np.linspace(100, 101, n15)), ohlc(np.linspace(100, 101, nh))

    def fake(htf_vals):
        return lambda close, *a_, **k_: (p15, s15, p15) if len(close) == n15 else (htf_vals, htf_vals, htf_vals)

    a = _aligned_agent()
    monkeypatch.setattr(rules_mod, "pmo", fake(htf_pos))
    r = a.analyze(Candidate("X_USDT", 98.5, 1e9, 0, 1), kh, k, 0, k15)
    assert r and r.side == "long" and r.confirm is False and "فوق الصفر" in r.reasons[-1]
    monkeypatch.setattr(rules_mod, "pmo", fake(-htf_pos))
    assert a.analyze(Candidate("X_USDT", 98.5, 1e9, 0, 1), kh, k, 0, k15) is None            # higher timeframe disagrees
    monkeypatch.setattr(rules_mod, "pmo", fake(htf_pos))
    assert a.analyze(Candidate("X_USDT", 98.5, 1e9, 0, 1), None, k, 0, k15) is None          # no HTF data -> fail closed
    assert a.analyze(Candidate("X_USDT", 104.0, 1e9, 0, 1), kh, k, 0, k15) is None           # price not inside the 4H OB
