import numpy as np

from tradebot.config import load_config
from tradebot.strategies.multi_agent.orchestrator import Orchestrator
from tradebot.strategies.telegram_signals.trader import Notifier, PaperExchange, SignalTrader

NOW = 1_800_000_000
TEXT = """🚨 فرصة جديدة | CRYPTO

🪙 العملة: X / USDT
📍 الاتجاه: LONG 🟢
⚙️ الرافعة: 5X

🎯 منطقة الدخول: 100
🛑 وقف الخسارة: 95 (-5%)

الأهداف:
✅ TP1: 101 (+1%) | ROI +5%
✅ TP2: 102 (+2%) | ROI +10%
✅ TP3: 103 (+3%) | ROI +15%
"""


class FC:
    page_limit = 1000

    def __init__(self): self.price = 100.0
    def last_price(self, s): return self.price
    def contract_detail(self, s): return {"maxLeverage": 50}
    def tickers(self): return []
    def klines(self, sym, interval, start, end):
        step = {"Min1": 60, "Min60": 3600}.get(interval, 3600)
        t = np.arange(start // step * step, end // step * step, step)
        p = np.full(len(t), self.price)
        return {"time": t.tolist(), "open": (p * 0.9995).tolist(), "close": p.tolist(), "high": (p * 1.001).tolist(),
                "low": (p * 0.999).tolist(), "vol": [1.0] * len(t), "amount": [1.0] * len(t)}


class Reader:
    def __init__(self, msgs): self.msgs = msgs
    def fetch(self): return self.msgs


class Silent(Notifier):
    def send(self, text): self.sent.append(text)
    def __init__(self, cfg): super().__init__(cfg); self.sent = []


def build(msgs):
    cfg = load_config("config.toml")
    fc, state = FC(), {}
    tr = SignalTrader(cfg, PaperExchange(cfg, fc, state), Silent(cfg), state, "/tmp/_st.json", "/tmp/_log.csv")
    o = Orchestrator(cfg, fc, tr, Reader(msgs))
    o.news._headlines = lambda: []                                 # no network in tests
    return cfg, fc, tr, o


def test_backlog_ignored_then_new_signal_enters_via_agent():
    old = {"id": 1, "ts": NOW - 10, "text": TEXT}
    cfg, fc, tr, orch = build([old])
    orch.cycle(NOW)
    assert not tr.open and not orch.pending                       # existing backlog never traded
    fc.price = 99.0                                               # below the entry zone (but above the stop)
    orch.reader.msgs = [old, {"id": 2, "ts": NOW, "text": TEXT}]
    orch.cycle(NOW + 5)
    assert "X_USDT" in orch.pending and not tr.open               # entry agent waits for the zone, no blind entry
    fc.price = 100.0
    orch.cycle(NOW + 10)                                          # price in zone + confirming 1m candles
    pos = tr.open["X_USDT"]
    assert pos.tps == [101.0, 102.0, 103.0] and pos.stop == 95.0 and pos.leverage == 5


def test_chased_signal_is_skipped_and_expired_cleaned():
    cfg, fc, tr, orch = build([])
    orch._first = False
    fc.price = 100.9                                              # already ran above the zone (+0.3%) and past chase limit
    orch.reader.msgs = [{"id": 5, "ts": NOW, "text": TEXT}]
    orch.cycle(NOW)
    assert "X_USDT" not in tr.open
