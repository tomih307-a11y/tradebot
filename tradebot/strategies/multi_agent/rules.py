"""User-defined rules analyst. A recommendation is produced only when EVERY condition in [agents.rules] conditions passes.
Each condition is a method `cond_<name>(ctx)` returning None (failed) or a dict that may set: side, zone, note, atr.
To add a condition: write `cond_<name>` below and append "<name>" to the config list.

cond_price_in_ob_4h : the current price lies inside a still-valid 4H order block.
   bullish break (BOS/CHoCH up) -> bullish OB (demand) -> long ; bearish break -> bearish OB (supply) -> short.
   OB candle  : ob_method "extreme" (the candle with the highest high / lowest low in the leg, as LuxAlgo) or "last_opposite".
   validity   : mitigation "wick" (a wick beyond the OB's far edge kills it), "close" (a close beyond it) or "none".
cond_pmo15_zero_touch : entry timeframe 15m. PMO first crossed ABOVE its signal line (long; below for short) while still on the
   wrong side of zero, and the FIRST bar where PMO then touches the zero line (PMO >= 0 for long, <= 0 for short) is the trigger.
   The recommendation fires on that bar (it must be among the last `pmo_touch_window_bars` closed 15m bars: 1 = the latest bar).
   Needs `side` from an earlier condition. The 15m trigger replaces the 1m confirmation of the entry agent (entry_tf_trigger).
Levels: zone = the OB, stop beyond its far edge by stop_buffer_atr x ATR(4H). open_target = true gives NO fixed targets (the exit
agent trails the stop); otherwise targets = tp_r multiples of the risk."""
from __future__ import annotations

import numpy as np

from tradebot.indicators import atr_rma, pmo

from .smc import order_block, structure
from .types import Candidate, Recommendation


def signal_cross_then_zero_touch(p: np.ndarray, sig: np.ndarray, s: int, cross_max_age: int, window: int):
    """Pure. p = PMO, sig = its signal line (NaN = masked warm-up), s = +1 long / -1 short. -> (cross_idx, touch_idx) or None.
    x = s*p, y = s*sig. Touch bar j: x[j-1] < 0 <= x[j] (first time PMO reaches the zero line) and n-1-j < window.
    Cross bar c < j (within cross_max_age bars): x[c]-y[c] >= 0 > x[c-1]-y[c-1] (PMO crossed above its signal) with x[c] < 0
    (it happened on the wrong side of zero) and x stayed < 0 on every bar from c to j-1 (no earlier touch)."""
    x, y = s * p, s * sig
    n = len(x)
    for j in range(n - 1, max(n - window - 1, 1), -1):
        if np.isnan(x[j]) or np.isnan(x[j - 1]) or not (x[j - 1] < 0 <= x[j]):
            continue
        for c in range(j - 1, max(j - cross_max_age, 1), -1):
            if x[c] >= 0 or np.isnan(x[c]) or np.isnan(y[c]) or np.isnan(x[c - 1]) or np.isnan(y[c - 1]):
                break                                    # PMO was already at/above zero: not a first touch after a cross
            if x[c] - y[c] >= 0 > x[c - 1] - y[c - 1]:
                return c, j
        return None
    return None


class RulesAgent:
    def __init__(self, cfg):
        self.c = cfg.agents.rules
        self.ind = cfg.pmo_zero_cross.indicator

    # ───────────── conditions ─────────────
    def cond_price_in_ob_4h(self, ctx: dict):
        c, k4 = self.c, ctx["k4h"]
        o, h, l, cl = k4["open"], k4["high"], k4["low"], k4["close"]
        if len(cl) < int(c.min_bars):
            return None
        _, events = structure(cl, h, l, int(c.swing_len))
        n, price = len(cl), ctx["price"]
        for ev in reversed(events):
            if n - 1 - ev["i"] > int(c.max_age_bars):
                break
            d = ev["dir"]
            ob = self._ob(ev, o, h, l, cl)
            if ob is None:
                continue
            lo, hi = ob
            after_l, after_h, after_c = l[ev["i"] + 1:], h[ev["i"] + 1:], cl[ev["i"] + 1:]
            if c.mitigation == "wick":
                dead = (after_l < lo).any() if d == 1 else (after_h > hi).any()
            elif c.mitigation == "close":
                dead = (after_c < lo).any() if d == 1 else (after_c > hi).any()
            else:
                dead = False
            if dead or not (lo <= price <= hi):
                continue
            atr = float(atr_rma(h, l, cl, int(c.atr_len))[-1])
            if not np.isfinite(atr) or atr <= 0:
                return None
            return {"side": "long" if d == 1 else "short", "zone": (lo, hi), "atr": atr,
                    "note": f"السعر داخل OB {'صاعد' if d == 1 else 'هابط'} 4س ({lo:.6g}–{hi:.6g}) بعد {ev['kind']}"}
        return None

    def cond_pmo15_zero_touch(self, ctx: dict):
        c, ind = self.c, self.ind
        side, k15 = ctx.get("side"), ctx.get("k15")
        k15 = k15() if callable(k15) else k15                   # loaded lazily: only symbols already inside a 4H OB
        if side is None or k15 is None or len(k15["close"]) < int(c.min_bars_15m):
            return None
        s = 1 if side == "long" else -1
        p, sig, _ = pmo(k15["close"], ind.roc_smooth_len, ind.roc_scale, ind.pmo_smooth_len, ind.signal_len)
        p, sig = p.copy(), sig.copy()
        p[: min(int(ind.warmup_bars), len(p))] = np.nan
        sig[: min(int(ind.warmup_bars), len(sig))] = np.nan
        hit = signal_cross_then_zero_touch(p, sig, s, int(c.pmo_cross_max_age_bars), int(c.pmo_touch_window_bars))
        if not hit:
            return None
        cb, j = hit
        n = len(p)
        return {"note": f"15د: PMO قطع الإشارة {'للأعلى' if s == 1 else 'للأسفل'} (قبل {n - 1 - cb} شمعة) ولمس خط الصفر الآن (قبل {n - 1 - j})",
                "confirm": not bool(c.entry_tf_trigger)}

    def _ob(self, ev, o, h, l, c):
        if self.c.ob_method == "last_opposite":
            r = order_block(ev, o, h, l, c)
            return (r[0], r[1]) if r else None
        a, b, d = ev["origin"], ev["i"], ev["dir"]
        if b <= a:
            return None
        j = a + int(np.argmin(l[a:b])) if d == 1 else a + int(np.argmax(h[a:b]))      # extreme candle of the leg
        return float(l[j]), float(h[j])

    # ───────────── engine ─────────────
    def analyze(self, cand: Candidate, k1h: dict, k4h: dict, now: int, k15=None) -> Recommendation | None:
        ctx = {"k1h": k1h, "k4h": k4h, "k15": k15, "price": cand.price if cand.price > 0 else float(k1h["close"][-1])}
        notes = []
        for name in self.c.conditions:
            fn = getattr(self, f"cond_{name}", None)
            if fn is None:
                raise KeyError(f"unknown condition {name!r} in [agents.rules] conditions")
            res = fn(ctx)
            if not res:
                return None
            ctx.update({k: v for k, v in res.items() if k != "note"})
            notes.append(res.get("note", name))
        side, zone, atr = ctx.get("side"), ctx.get("zone"), ctx.get("atr")
        if not (side and zone and atr):
            return None
        s = 1 if side == "long" else -1
        lo, hi = zone
        stop = (lo - float(self.c.stop_buffer_atr) * atr) if s == 1 else (hi + float(self.c.stop_buffer_atr) * atr)
        entry = hi if s == 1 else lo                           # first edge reached on a retrace (worst-case fill)
        risk = s * (entry - stop)
        if risk <= 0:
            return None
        tps = [] if bool(self.c.open_target) else [entry + s * float(r) * risk for r in self.c.tp_r]
        rec = Recommendation(cand.symbol, side, 1.0, ctx["price"], atr, (lo, hi), stop, tps, notes, now)
        rec.retrace = True
        rec.confirm = bool(ctx.get("confirm", True))
        return rec
