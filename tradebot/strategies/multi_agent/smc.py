"""Smart-Money-Concepts analyst. Closed candles only; a swing is known only n bars after it forms (no look-ahead).

 structure : swing highs/lows (fractal, n bars each side). A CLOSE beyond the last swing = BOS (with the trend) or CHoCH (against it).
 order block: the last opposite-colour candle in the leg that led to the break (bullish OB = last down candle before an up-break).
 FVG       : 3-candle imbalance (bullish: low[i] > high[i-2]).
 sweep     : a wick through an earlier swing that closes back inside (liquidity grab).
 dealing range: last swing high/low; long favoured in discount (below 50%), short in premium.
Setup = HTF (4H) trend and LTF (1H) structure agree, and an untapped POI (OB, else FVG) lies within reach of price.
Entry zone = the POI; stop beyond the POI's far edge; targets are R-multiples of the stop distance. All numbers: [agents.smc]."""
from __future__ import annotations

import numpy as np

from tradebot.indicators import atr_rma

from .types import Candidate, Recommendation


def pivots(high: np.ndarray, low: np.ndarray, n: int):
    """Confirmed fractal swings -> (highs, lows) as lists of (index, price). Index i needs bars up to i+n."""
    hs, ls = [], []
    for i in range(n, len(high) - n):
        if high[i] > high[i - n:i].max() and high[i] >= high[i + 1:i + n + 1].max():
            hs.append((i, float(high[i])))
        if low[i] < low[i - n:i].min() and low[i] <= low[i + 1:i + n + 1].min():
            ls.append((i, float(low[i])))
    return hs, ls


def structure(close: np.ndarray, high: np.ndarray, low: np.ndarray, n: int):
    """Walk the bars once. -> (trend, events). trend: +1/-1/0. event: dict(i, kind, dir, level, origin)."""
    trend, events = 0, []
    sh = sl = None                                   # [price, idx, broken]
    for i in range(len(close)):
        k = i - n                                    # swing at k becomes known now
        if k >= n:
            if high[k] > high[k - n:k].max() and high[k] >= high[k + 1:i + 1].max():
                sh = [float(high[k]), k, False]
            if low[k] < low[k - n:k].min() and low[k] <= low[k + 1:i + 1].min():
                sl = [float(low[k]), k, False]
        if sh and not sh[2] and close[i] > sh[0]:
            events.append({"i": i, "kind": "CHoCH" if trend == -1 else "BOS", "dir": 1, "level": sh[0],
                           "origin": sl[1] if sl else sh[1]})
            sh[2], trend = True, 1
        if sl and not sl[2] and close[i] < sl[0]:
            events.append({"i": i, "kind": "CHoCH" if trend == 1 else "BOS", "dir": -1, "level": sl[0],
                           "origin": sh[1] if sh else sl[1]})
            sl[2], trend = True, -1
    return trend, events


def order_block(ev: dict, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray):
    """Last opposite-colour candle between the leg's origin and the break. -> (lo, hi, idx) or None."""
    d, a, b = ev["dir"], ev["origin"], ev["i"]
    for j in range(b - 1, a - 1, -1):
        if (c[j] < o[j]) if d == 1 else (c[j] > o[j]):
            return float(l[j]), float(h[j]), j
    if b > a:                                        # no opposite candle: the extreme candle of the leg
        j = a + int(np.argmin(l[a:b])) if d == 1 else a + int(np.argmax(h[a:b]))
        return float(l[j]), float(h[j]), j
    return None


def fvgs(h: np.ndarray, l: np.ndarray, d: int, start: int):
    """Imbalances in direction d formed at/after `start`. -> list of (lo, hi, idx)."""
    out = []
    for i in range(max(start, 2), len(h)):
        if d == 1 and l[i] > h[i - 2]:
            out.append((float(h[i - 2]), float(l[i]), i))
        if d == -1 and h[i] < l[i - 2]:
            out.append((float(h[i]), float(l[i - 2]), i))
    return out


def untapped(zone, d: int, since: int, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
    """Valid POI: after `since` (excluding the current bar) price neither closed through it nor already traded into it."""
    lo, hi = zone[0], zone[1]
    for k in range(since + 1, len(c) - 1):
        if d == 1 and (c[k] < lo or l[k] <= hi):
            return False
        if d == -1 and (c[k] > hi or h[k] >= lo):
            return False
    return True


class SMCAgent:
    def __init__(self, cfg):
        self.c = cfg.agents.smc

    def analyze(self, cand: Candidate, k1h: dict, k4h: dict, now: int) -> Recommendation | None:
        c, n = self.c, int(self.c.swing_len)
        if len(k1h["close"]) < int(c.min_bars) or len(k4h["close"]) < int(c.min_bars):
            return None
        o, h, l, cl = k1h["open"], k1h["high"], k1h["low"], k1h["close"]
        trend, events = structure(cl, h, l, n)
        htf_trend, _ = structure(k4h["close"], k4h["high"], k4h["low"], n)
        if trend == 0 or not events:
            return None
        d = trend
        last = [e for e in events if e["dir"] == d][-1]
        if len(cl) - 1 - last["i"] > int(c.event_max_age):
            return None
        atr = float(atr_rma(h, l, cl, int(c.atr_len))[-1])
        if not np.isfinite(atr) or atr <= 0:
            return None
        price = float(cl[-1])
        ob = order_block(last, o, h, l, cl)
        gaps = [g for g in fvgs(h, l, d, last["origin"]) if untapped(g, d, g[2], h, l, cl)]
        poi, kind = None, ""
        if ob and untapped(ob, d, last["i"], h, l, cl):
            poi, kind = (ob[0], ob[1]), "OB"
        elif gaps:
            g = gaps[-1]
            poi, kind = (g[0], g[1]), "FVG"
        if poi is None:
            return None
        lo, hi = poi
        near = hi if d == 1 else lo                              # first edge price reaches when retracing
        if abs(price - near) > float(c.max_dist_atr) * atr:
            return None
        confluence = bool(ob and any(max(g[0], lo) < min(g[1], hi) for g in gaps)) if kind == "OB" else False
        hs, ls = pivots(h, l, n)
        rng_hi = max((p for _, p in hs[-3:]), default=float(h.max()))
        rng_lo = min((p for _, p in ls[-3:]), default=float(l.min()))
        mid = (rng_hi + rng_lo) / 2
        discount = price < mid if d == 1 else price > mid
        sweep = False
        for k in range(max(1, len(cl) - int(c.sweep_lookback)), len(cl)):
            for idx, p in (ls if d == 1 else hs):
                if idx < k - n and ((l[k] < p < cl[k]) if d == 1 else (h[k] > p > cl[k])):
                    sweep = True
        checks = {"اتجاه 4س مع الهيكل": htf_trend == d, f"كسر هيكل {last['kind']} حديث": True, f"منطقة {kind} غير ملموسة": True,
                  "تقاطع OB مع FVG": confluence, "سعر في خصم/علاوة مناسبة": bool(discount), "سحب سيولة": sweep}
        if bool(c.require_htf) and htf_trend != d:
            return None
        score = sum(checks.values()) / len(checks)
        if score < float(c.min_score):
            return None
        stop = (lo - float(c.stop_buffer_atr) * atr) if d == 1 else (hi + float(c.stop_buffer_atr) * atr)
        entry = near
        risk = d * (entry - stop)
        if risk <= 0:
            return None
        tps = [entry + d * float(r) * risk for r in c.tp_r]
        rec = Recommendation(cand.symbol, "long" if d == 1 else "short", score, price, atr, (lo, hi), stop, tps,
                             [k for k, v in checks.items() if v], now)
        rec.retrace = True
        return rec
