"""REAL execution on MEXC futures for the channel's signals (isolated margin, market entry, reduce-only limit TPs).

Same interface as PaperExchange so SignalTrader is unchanged. Safety rules:
 - if anything fails after the entry filled (TP/stop placement) the position is closed at market immediately;
 - the 4H-close stop rule (channel's rule) is evaluated here on closed 1m candles and closes at market;
 - stop_mode = "exchange" places a real stop-loss order instead.
"""
from __future__ import annotations

import logging
import math
import time

from tradebot.mexc.client import MexcClient

from . import paper
from .paper import FOUR_H, Position
from .trader import SkipSignal

log = logging.getLogger("telegram_signals")

OPEN_LONG, CLOSE_SHORT, OPEN_SHORT, CLOSE_LONG = 1, 2, 3, 4


def _oid(r) -> int:
    return int(r["orderId"] if isinstance(r, dict) else r)


def _round(price: float, unit: float, up: bool = False) -> float:
    if not unit:
        return price
    n = price / unit
    return round((math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)) * unit, 12)


class LiveExchange:
    def __init__(self, cfg, client: MexcClient, state: dict):
        t = cfg.telegram_signals
        self.cfg, self.t, self.client, self.state = cfg, t, client, state
        self._detail: dict = {}

    # ── queries ──
    def balance(self) -> float:
        return self.client.account_equity("USDT")

    def contract(self, symbol: str) -> dict | None:
        if symbol not in self._detail:
            try:
                self._detail[symbol] = self.client.contract_detail(symbol)
            except Exception:
                self._detail[symbol] = None
        return self._detail[symbol]

    def last_price(self, symbol: str) -> float:
        return self.client.last_price(symbol)

    def _position(self, pos: Position) -> dict | None:
        want = 1 if pos.side == "long" else 2
        for p in self.client.open_positions(pos.symbol):
            if int(p.get("positionId", 0)) == pos.ext.get("position_id") or (
                    not pos.ext.get("position_id") and int(p.get("positionType", 0)) == want):
                return p
        return None

    def _close_side(self, pos: Position) -> int:
        return CLOSE_LONG if pos.side == "long" else CLOSE_SHORT

    # ── actions ──
    def open(self, sig, leverage: int, margin: float, now: int) -> Position:
        d = self.contract(sig.symbol)
        cs, min_vol, unit = float(d["contractSize"]), int(d.get("minVol", 1)), float(d.get("priceUnit") or 0)
        px = self.last_price(sig.symbol)
        n = len(sig.tps)
        vol = int(margin * leverage / (px * cs))
        if vol < max(min_vol, n):
            raise SkipSignal(f"الهامش {margin:.2f}$ صغير لهذا العقد (الحجم {vol} عقد، الحد الأدنى {max(min_vol, n)})")
        s = 1 if sig.side == "long" else -1
        r = self.client.submit_market_order(sig.symbol, OPEN_LONG if s == 1 else OPEN_SHORT, vol, leverage, "isolated", px,
                                            f"tg{sig.msg_id}x{now % 100000}")
        oid = _oid(r)
        mp = None
        for _ in range(20):                                   # wait for the fill
            time.sleep(0.5)
            for p in self.client.open_positions(sig.symbol):
                if int(p.get("positionType", 0)) == (1 if s == 1 else 2) and float(p.get("holdVol", 0)) > 0:
                    mp = p
            if mp:
                break
        if not mp:
            raise RuntimeError(f"لم يظهر مركز بعد الأمر {oid} — تحقق يدوياً من MEXC")
        pos = Position(symbol=sig.symbol, side=sig.side, entry=float(mp["holdAvgPrice"]), leverage=leverage,
                       margin=float(mp.get("im") or margin), tps=list(sig.tps), stop=sig.stop, opened_ts=now,
                       stop_mode=self.t.stop_mode, liq_price=float(mp.get("liquidatePrice") or paper.liq_price(
                           float(mp["holdAvgPrice"]), sig.side, leverage, self.t.maintenance_margin)),
                       signal_msg_id=sig.msg_id, last_ts=now)
        total = int(float(mp["holdVol"]))
        pos.ext.update(position_id=int(mp["positionId"]), vol=total, unit=unit, tp_orders=[], stop_id=None, cs=cs)
        try:
            share = max(1, total // n)
            for i, tp in enumerate(sig.tps):
                v = total - share * (n - 1) if i == n - 1 else share
                price = _round(tp, unit, up=(s == -1))
                r = self.client.submit_limit_close(sig.symbol, self._close_side(pos), v, price, leverage,
                                                   pos.ext["position_id"], f"tp{i + 1}-{sig.msg_id}x{now % 100000}")
                pos.ext["tp_orders"].append({"id": _oid(r), "vol": v})
            if self.t.stop_mode == "exchange":
                r = self.client.place_stop(pos.ext["position_id"], total, _round(sig.stop, unit, up=(s == 1)))
                pos.ext["stop_id"] = _oid(r) if r is not None else None
        except Exception:
            log.exception("protection failed — closing at market")
            self._market_close(pos)
            raise
        return pos

    def _market_close(self, pos: Position) -> None:
        ids = [o["id"] for o in pos.ext.get("tp_orders", [])]
        if ids:
            try:
                self.client.cancel_orders(ids)
            except Exception:
                log.warning("cancel TP orders failed")
        if pos.ext.get("stop_id"):
            try:
                self.client.cancel_stop(pos.ext["stop_id"])
            except Exception:
                log.warning("cancel stop failed")
        mp = self._position(pos)
        if mp and float(mp.get("holdVol", 0)) > 0:
            self.client.submit_market_close(pos.symbol, self._close_side(pos), int(float(mp["holdVol"])),
                                            self.last_price(pos.symbol), pos.leverage, pos.ext["position_id"],
                                            f"cl-{pos.signal_msg_id}x{int(time.time()) % 100000}")

    def advance(self, pos: Position, now: int) -> list:
        n0, ext, s = len(pos.events), pos.ext, pos.s
        mp = self._position(pos)
        hold = int(float(mp["holdVol"])) if mp else 0
        sold = ext["vol"] - hold
        # filled take-profits (cumulative volume of the TP orders)
        cum = 0
        hit = 0
        for o in ext["tp_orders"]:
            cum += o["vol"]
            if sold >= cum:
                hit += 1
        while pos.tp_hit < hit:
            i = pos.tp_hit
            frac = ext["tp_orders"][i]["vol"] / ext["vol"]
            pnl = frac * pos.notional * s * (pos.tps[i] - pos.entry) / pos.entry
            pos.realized += pnl
            pos.events.append((now, f"tp{i + 1}", pos.tps[i], frac, pnl))
            pos.tp_hit += 1
            pos.remaining = round(hold / ext["vol"], 12)
            if pos.tp_hit == 1 and self.t.breakeven_after_tp1 and s * (pos.entry - pos.stop) > 0:
                self.move_stop(pos, pos.entry)
        if hold <= 0:
            pos.closed, pos.exit_ts = True, now
            pos.exit_reason = "all_tp" if pos.tp_hit >= len(pos.tps) else (pos.exit_reason or "closed_on_exchange")
            return pos.events[n0:]
        pos.remaining = round(hold / ext["vol"], 12)
        # channel rule: a CLOSED 4H candle beyond the stop -> market exit
        if pos.stop_mode == "channel_4h_close":
            try:
                d = self.client.klines(pos.symbol, "Min1", max(pos.last_ts, pos.opened_ts), now)
            except Exception as e:
                log.warning("klines failed %s: %s", pos.symbol, e)
                return pos.events[n0:]
            for i, t in enumerate(d.get("time", [])):
                if t + 60 > now:
                    break
                pos.last_ts = max(pos.last_ts, int(t))
                if (t + 60) % FOUR_H == 0 and s * (float(d["close"][i]) - pos.stop) < 0:
                    px = self.last_price(pos.symbol)
                    self._market_close(pos)
                    pnl = pos.remaining * pos.notional * s * (px - pos.entry) / pos.entry
                    pos.realized += pnl
                    pos.events.append((now, "stop_4h_close", px, pos.remaining, pnl))
                    pos.remaining, pos.closed, pos.exit_ts, pos.exit_reason = 0.0, True, now, "stop_4h_close"
                    break
        return pos.events[n0:]

    def move_stop(self, pos: Position, price: float) -> None:
        pos.stop = price
        if pos.stop_mode == "exchange" and pos.ext.get("stop_id"):
            self.client.change_stop(pos.ext["stop_id"], _round(price, pos.ext.get("unit", 0), up=(pos.s == 1)))
