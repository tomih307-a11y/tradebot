import pytest

from tradebot.config import load_config
from tradebot.strategies.telegram_signals.live import LiveExchange
from tradebot.strategies.telegram_signals.parser import Signal
from tradebot.strategies.telegram_signals.trader import SkipSignal


class FakeClient:
    def __init__(self, fail_limit=False):
        self.calls, self.hold, self.fail_limit = [], 0, fail_limit

    def contract_detail(self, s): return {"contractSize": 0.01, "minVol": 1, "priceUnit": 0.1, "maxLeverage": 50}
    def last_price(self, s): return 100.0
    def account_equity(self, c="USDT"): return 25.0
    def submit_market_order(self, sym, side, vol, lev, mt, px, ext):
        self.calls.append(("open", side, vol)); self.hold = vol; return {"orderId": 1}
    def open_positions(self, sym=None):
        return [{"positionId": 7, "positionType": 1, "holdVol": self.hold, "holdAvgPrice": 100.0, "im": 2.5, "liquidatePrice": 80.0}] if self.hold else []
    def submit_limit_close(self, sym, side, vol, price, lev, pid, ext):
        if self.fail_limit: raise RuntimeError("boom")
        self.calls.append(("tp", side, vol, price)); return {"orderId": 10 + len(self.calls)}
    def submit_market_close(self, sym, side, vol, px, lev, pid, ext):
        self.calls.append(("mclose", side, vol)); self.hold = 0; return {"orderId": 99}
    def cancel_orders(self, ids): self.calls.append(("cancel", tuple(ids)))
    def place_stop(self, *a): return {"orderId": 5}
    def cancel_stop(self, i): pass
    def klines(self, *a): return {"time": []}


def sig():
    return Signal(symbol="X_USDT", side="long", entry=100.0, tps=[101.0, 102.0, 103.0], stop=95.0, leverage=10, msg_id=1, ts=0)


def make(fc):
    cfg = load_config("config.toml")
    return LiveExchange(cfg, fc, {})


def test_open_places_split_tps():
    fc = FakeClient()
    pos = make(fc).open(sig(), 10, 2.5, 1000)   # notional 25$ / (100*0.01) = 25 contracts
    assert fc.calls[0] == ("open", 1, 25)
    tps = [c for c in fc.calls if c[0] == "tp"]
    assert [c[2] for c in tps] == [8, 8, 9] and all(c[1] == 4 for c in tps)
    assert pos.ext["position_id"] == 7


def test_protection_failure_closes_at_market():
    fc = FakeClient(fail_limit=True)
    with pytest.raises(RuntimeError):
        make(fc).open(sig(), 10, 2.5, 1000)
    assert fc.calls[-1][0] == "mclose" and fc.hold == 0


def test_too_small_margin_skipped():
    with pytest.raises(SkipSignal):
        make(FakeClient()).open(sig(), 1, 0.5, 1000)   # 0.5$ -> 0 contracts


def test_tp_fill_detected_and_breakeven():
    fc = FakeClient()
    ex = make(fc)
    pos = ex.open(sig(), 10, 2.5, 1000)
    fc.hold = 17                                         # first TP (8 contracts) filled
    ev = ex.advance(pos, 1100)
    assert [e[1] for e in ev] == ["tp1"] and pos.tp_hit == 1 and pos.stop == pos.entry
    fc.hold = 0
    ex.advance(pos, 1200)
    assert pos.closed
