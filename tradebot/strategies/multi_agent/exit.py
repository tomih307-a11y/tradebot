"""Agent 5 — exit manager: trails the stop with ATR after TP1, cuts a stale losing trade, and tightens/closes on adverse news.
The stop is only ever tightened. Targets and the channel/ATR initial stop stay as set at entry."""
from __future__ import annotations

from tradebot.strategies.telegram_signals.paper import Position

from .types import ExitAction, Vote


class ExitAgent:
    def __init__(self, cfg):
        self.c = cfg.agents.exit

    def manage(self, pos: Position, price: float, atr: float, news_bias: float, now: int) -> ExitAction:
        s, c = pos.s, self.c
        upnl = s * (price - pos.entry)
        if news_bias * s <= -float(c.news_close_bias):
            return ExitAction(close=True, reason="خبر عكسي قوي")
        if now - pos.opened_ts > int(c.max_hold_sec) and upnl <= 0:
            return ExitAction(close=True, reason="انتهت مدة الاحتفاظ والصفقة غير رابحة")
        new = None
        if pos.tp_hit >= 1 and atr > 0:
            trail = price - s * float(c.trail_atr) * atr
            if s * (trail - pos.stop) > 0:
                new = trail
        if news_bias * s <= -float(c.news_tighten_bias) and upnl > 0 and s * (pos.entry - pos.stop) > 0:
            new = pos.entry if new is None or s * (pos.entry - new) > 0 else new
        return ExitAction(new_stop=new, reason="تحريك الوقف") if new is not None else ExitAction()

    def assess(self, rec) -> Vote:
        """Risk opinion before publishing: average reward/risk over the targets vs the stop distance."""
        s = 1 if rec.side == "long" else -1
        risk = s * (rec.zone[1] - rec.stop) if s == 1 else s * (rec.zone[0] - rec.stop)
        if risk <= 0:
            return Vote("exit", 0.0, True, "الوقف داخل منطقة الدخول")
        rr = sum(s * (tp - (rec.zone[1] if s == 1 else rec.zone[0])) for tp in rec.tps) / len(rec.tps) / risk
        return Vote("exit", min(1.0, rr / float(self.c.target_rr)), rr < float(self.c.min_rr), f"R:R≈{rr:.2f}")
