"""Agent 4 — entry timing: enter only inside the preferred zone, after the last closed 1m candle turns with the trade, and never
chase a move that already ran away. Goal: the smallest adverse excursion after entry."""
from __future__ import annotations

from .types import EntryPlan, Recommendation


class EntryAgent:
    def __init__(self, cfg):
        self.c = cfg.agents.entry

    def plan(self, rec: Recommendation, price: float, k1m: dict) -> EntryPlan:
        s = 1 if rec.side == "long" else -1
        lo, hi = rec.zone
        chase = float(self.c.max_chase_atr) * rec.atr
        if s * (price - rec.stop) <= 0:
            return EntryPlan("skip", "السعر تجاوز الوقف")
        if rec.retrace:                                  # POI setups: price is meant to come back to the zone
            if s * (price - (hi if s == 1 else lo)) > 0:
                return EntryPlan("wait", "انتظار عودة السعر إلى منطقة SMC")
        elif s * (price - (hi if s == 1 else lo)) > chase:
            return EntryPlan("skip", "السعر ركض بعيداً عن منطقة الدخول (لا مطاردة)")
        if not (lo <= price <= hi):
            return EntryPlan("wait", "انتظار عودة السعر إلى منطقة الدخول")
        n = int(self.c.confirm_candles)
        if len(k1m["close"]) < n:
            return EntryPlan("wait", "لا بيانات دقيقة كافية")
        opens, closes = k1m["open"][-n:], k1m["close"][-n:]
        if all(s * (c - o) > 0 for o, c in zip(opens, closes)):
            return EntryPlan("market", f"{n} شموع دقيقة مغلقة مع الاتجاه داخل المنطقة")
        return EntryPlan("wait", "انتظار تأكيد شمعة الدقيقة")
