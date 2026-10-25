"""Agent 2 — technical analyst: 4H sets the trend, 1H must show a pullback with momentum turning back with it.
Levels come from ATR. Every threshold is in config.toml [agents.technical]. (An unproven rule set: judge it by paper results.)"""
from __future__ import annotations

import numpy as np

from tradebot.indicators import atr_rma, ema, pmo, rsi

from .types import Candidate, Recommendation


class TechnicalAgent:
    def __init__(self, cfg):
        self.c = cfg.agents.technical
        self.ind = cfg.pmo_zero_cross.indicator

    def analyze(self, cand: Candidate, k1h: dict, k4h: dict, now: int) -> Recommendation | None:
        c = self.c
        if len(k1h["close"]) < int(c.min_bars) or len(k4h["close"]) < int(c.min_bars):
            return None
        c1, h1, l1 = k1h["close"], k1h["high"], k1h["low"]
        c4 = k4h["close"]
        e_fast4, e_slow4 = ema(c4, int(c.ema_fast)), ema(c4, int(c.ema_slow))
        if c4[-1] > e_slow4[-1] and e_fast4[-1] > e_slow4[-1]:
            side = "long"
        elif c4[-1] < e_slow4[-1] and e_fast4[-1] < e_slow4[-1]:
            side = "short"
        else:
            return None
        s = 1 if side == "long" else -1
        atr = float(atr_rma(h1, l1, c1, int(c.atr_len))[-1])
        e20 = float(ema(c1, int(c.ema_fast))[-1])
        r = float(rsi(c1, int(c.rsi_len))[-1])
        p, sg, _ = pmo(c1, self.ind.roc_smooth_len, self.ind.roc_scale, self.ind.pmo_smooth_len, self.ind.signal_len)
        price = float(c1[-1])
        checks = {
            "اتجاه 4س": True,
            "ارتداد إلى EMA السريع": abs(price - e20) <= float(c.pullback_atr) * atr,
            "RSI غير مبالغ": (float(c.rsi_lo) <= r <= float(c.rsi_hi)) if s == 1 else (100 - float(c.rsi_hi) <= r <= 100 - float(c.rsi_lo)),
            "PMO مع الاتجاه": bool(s * (p[-1] - sg[-1]) > 0),
            "شمعة الساعة الأخيرة مع الاتجاه": bool(s * (c1[-1] - k1h["open"][-1]) > 0),
        }
        score = sum(checks.values()) / len(checks)
        if score < float(c.min_score) or not np.isfinite(atr) or atr <= 0:
            return None
        lo, hi = sorted((e20 - float(c.zone_atr) * atr, e20 + float(c.zone_atr) * atr))
        entry = min(max(price, lo), hi)
        stop = entry - s * float(c.stop_atr) * atr
        tps = [entry + s * float(m) * atr for m in c.tp_atr]
        return Recommendation(cand.symbol, side, score, price, atr, (lo, hi), stop, tps,
                              [k for k, v in checks.items() if v], now)
