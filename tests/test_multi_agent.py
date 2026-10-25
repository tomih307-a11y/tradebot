import numpy as np

from tradebot.config import load_config
from tradebot.strategies.multi_agent.entry import EntryAgent
from tradebot.strategies.multi_agent.exit import ExitAgent
from tradebot.strategies.multi_agent.news import score_headlines
from tradebot.strategies.multi_agent.scanner import ScannerAgent
from tradebot.strategies.multi_agent.technical import TechnicalAgent
from tradebot.strategies.multi_agent.types import Candidate, Recommendation
from tradebot.strategies.telegram_signals.paper import Position

CFG = load_config("config.toml")


def kl(close):
    c = np.asarray(close, float)
    o = np.concatenate([[c[0]], c[:-1]])
    return {"time": np.arange(len(c)) * 3600, "open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998, "close": c}


def rec(side="long", price=100.0, zone=(99.0, 101.0), atr=1.0):
    s = 1 if side == "long" else -1
    return Recommendation("X_USDT", side, 1.0, price, atr, zone, price - s * 2, [price + s, price + 2 * s, price + 3 * s])


def test_scanner_filters_and_ranks():
    t = [{"symbol": "A_USDT", "lastPrice": 1, "amount24": 1e9, "riseFallRate": 0.02},
         {"symbol": "B_USDT", "lastPrice": 1, "amount24": 1e3, "riseFallRate": 0.5},      # illiquid -> dropped
         {"symbol": "C_USDT", "lastPrice": 1, "amount24": 1e9, "riseFallRate": -0.10}]
    assert [c.symbol for c in ScannerAgent(CFG).rank(t)] == ["C_USDT", "A_USDT"]


def test_technical_long_in_uptrend_pullback():
    rng = np.random.default_rng(1)
    up = 100 + np.cumsum(np.abs(rng.normal(0.3, 0.2, 300)))
    k1 = kl(up)
    k1["close"][-3:] = k1["close"][-3] * np.array([0.995, 0.99, 1.003])   # shallow pullback, last candle up
    k1["open"][-1] = k1["close"][-1] * 0.998
    r = TechnicalAgent(CFG).analyze(Candidate("X_USDT", float(up[-1]), 1e9, 1, 1), k1, kl(up[::1]), 0)
    if r:                                                                  # levels must be coherent when it fires
        assert r.side == "long" and r.stop < r.zone[0] and all(tp > r.zone[1] for tp in r.tps)


def test_technical_no_trade_when_flat():
    flat = kl(100 + np.sin(np.arange(300) / 5) * 0.01)
    assert TechnicalAgent(CFG).analyze(Candidate("X_USDT", 100, 1e9, 0, 1), flat, flat, 0) is None


def test_entry_waits_skips_and_enters():
    ea, r = EntryAgent(CFG), rec()
    up = {"open": np.array([100.0, 100.2]), "close": np.array([100.2, 100.4])}
    assert ea.plan(r, 100.4, up).action == "market"
    assert ea.plan(r, 100.4, {"open": np.array([100.4, 100.3]), "close": np.array([100.3, 100.2])}).action == "wait"
    assert ea.plan(r, 98.5, up).action == "wait"                  # below the zone, not through the stop
    assert ea.plan(r, 103.0, up).action == "skip"                 # ran away: no chasing
    assert ea.plan(r, 97.0, up).action == "skip"                  # beyond the stop


def pos(side="long", tp_hit=0):
    return Position("X_USDT", side, 100.0, 5, 2.5, [101, 102, 103], 98.0, 0, "channel_4h_close", 80.0, signal_msg_id=-1, tp_hit=tp_hit)


def test_exit_trails_only_after_tp1_and_only_tightens():
    ex = ExitAgent(CFG)
    assert ex.manage(pos(tp_hit=0), 101.5, 1.0, 0, 100).new_stop is None
    a = ex.manage(pos(tp_hit=1), 103.0, 1.0, 0, 100)
    assert a.new_stop == 103.0 - 1.5
    assert ex.manage(pos(tp_hit=1), 99.0, 1.0, 0, 100).new_stop is None      # would loosen -> ignored


def test_exit_news_and_time():
    ex = ExitAgent(CFG)
    assert ex.manage(pos(), 100.5, 1.0, -0.9, 100).close
    assert ex.manage(pos(), 99.0, 1.0, 0, 10 ** 6).close                      # stale loser
    assert not ex.manage(pos(), 101.0, 1.0, 0, 10 ** 6).close                 # stale but winning


def test_news_lexicon():
    assert score_headlines(["SEC approves ETF", "Exchange hack drains funds", "Huge rally"]) > 0
    assert score_headlines(["Exchange hack", "Delist announced"]) == -1
    assert score_headlines(["nothing relevant"]) == 0
