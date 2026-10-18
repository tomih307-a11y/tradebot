"""Position model + candle-driven simulation of a signal trade (used by paper mode and by the history backtest).

Rules (all numbers from config):
 - entry: market fill (+slippage). Margin = margin_pct of balance, isolated => max loss = margin.
 - targets: equal share of the position closed at each TP (limit-like fill when the 1m HIGH/LOW touches it).
 - after TP1 the stop moves to the entry price (the channel does the same) if breakeven_after_tp1.
 - stop_mode "channel_4h_close": exit at the CLOSE of a 4H candle that closes beyond the stop (the channel's rule).
   stop_mode "exchange": exit at the stop price (gap -> open) as soon as the 1m candle touches it.
 - liquidation: isolated margin is lost if price touches the liquidation price (checked first, conservatively).
 - within one 1m candle the adverse event is assumed to happen BEFORE the favourable one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FOUR_H = 14400


@dataclass
class Position:
    symbol: str
    side: str
    entry: float                    # actual fill
    leverage: int
    margin: float
    tps: list[float]
    stop: float
    opened_ts: int
    stop_mode: str
    liq_price: float
    signal_msg_id: int = 0
    remaining: float = 1.0          # fraction of the position still open
    tp_hit: int = 0
    last_ts: int = 0                # last processed 1m candle open time
    realized: float = 0.0           # USDT, after fees (entry fee deducted at open)
    fees: float = 0.0
    closed: bool = False
    exit_reason: str | None = None
    exit_ts: int | None = None
    events: list = field(default_factory=list)   # (ts, kind, price, fraction, pnl)
    ext: dict = field(default_factory=dict)      # live-exchange ids (position id, order ids, vol)

    @property
    def s(self) -> int:
        return 1 if self.side == "long" else -1

    @property
    def notional(self) -> float:
        return self.margin * self.leverage

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)


def liq_price(entry: float, side: str, leverage: int, maint: float) -> float:
    s = 1 if side == "long" else -1
    return entry * (1 - s * (1.0 / leverage - maint))


def open_position(sig, fill: float, balance: float, margin_pct: float, leverage: int, ts: int, stop_mode: str,
                  fee_pct: float, maint: float) -> Position:
    margin = balance * margin_pct / 100.0
    p = Position(symbol=sig.symbol, side=sig.side, entry=fill, leverage=leverage, margin=margin, tps=list(sig.tps),
                 stop=sig.stop, opened_ts=ts, stop_mode=stop_mode, liq_price=liq_price(fill, sig.side, leverage, maint),
                 signal_msg_id=sig.msg_id, last_ts=ts)
    fee = p.notional * fee_pct / 100.0
    p.fees += fee
    p.realized -= fee
    p.events.append((ts, "open", fill, 1.0, -fee))
    return p


def _close(p: Position, ts: int, price: float, frac: float, kind: str, fee_pct: float) -> None:
    gross = frac * p.notional * p.s * (price - p.entry) / p.entry
    fee = frac * p.notional * fee_pct / 100.0
    pnl = gross - fee
    p.realized += pnl
    p.fees += fee
    p.remaining = round(p.remaining - frac, 12)
    p.events.append((ts, kind, price, frac, pnl))


def step(p: Position, candle, cfg_ts) -> None:
    """Advance one 1m candle (ts_open, o, h, l, c). cfg_ts: dict(fee_pct, slip_pct, breakeven_after_tp1)."""
    if p.closed:
        return
    ts, o, h, l, c = candle
    if ts <= p.last_ts:
        return
    p.last_ts = ts
    fee, slip, s = cfg_ts["fee_pct"], cfg_ts["slip_pct"] / 100.0, p.s
    adverse = l if s == 1 else h
    favourable = h if s == 1 else l

    def finish(reason: str):
        p.closed, p.exit_reason, p.exit_ts = True, reason, ts + 60

    # 1) liquidation (isolated margin lost)
    if s * (adverse - p.liq_price) <= 0:
        loss = p.remaining * p.margin
        p.realized -= loss
        p.events.append((ts, "liquidated", p.liq_price, p.remaining, -loss))
        p.remaining = 0.0
        return finish("liquidated")
    # 2) exchange-style stop
    if p.stop_mode == "exchange" and s * (adverse - p.stop) <= 0:
        px = o if s * (o - p.stop) <= 0 else p.stop
        _close(p, ts, px * (1 - s * slip), p.remaining, "stop", fee)
        return finish("stop")
    # 3) targets, in order
    share = 1.0 / len(p.tps)
    while p.tp_hit < len(p.tps) and s * (favourable - p.tps[p.tp_hit]) >= 0:
        last = p.tp_hit == len(p.tps) - 1
        _close(p, ts, p.tps[p.tp_hit], p.remaining if last else min(share, p.remaining), f"tp{p.tp_hit + 1}", fee)
        p.tp_hit += 1
        if p.tp_hit == 1 and cfg_ts["breakeven_after_tp1"]:
            p.stop = p.entry if s * (p.entry - p.stop) > 0 else p.stop
        if p.remaining <= 1e-9:
            return finish("all_tp")
    # 4) channel rule: 4H candle CLOSE beyond the stop
    if p.stop_mode == "channel_4h_close" and (ts + 60) % FOUR_H == 0 and s * (c - p.stop) < 0:
        _close(p, ts, c * (1 - s * slip), p.remaining, "stop_4h_close", fee)
        return finish("stop_4h_close")
