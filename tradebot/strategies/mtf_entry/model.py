"""Multi-timeframe PMO+MACD direction and entry zone (mirror of pine/mtf_pmo_macd_entry.pine). Closed candles only, pure functions."""
from __future__ import annotations

import numpy as np

from tradebot.indicators import atr_rma, ema, pmo

# (timeframe, weight) — daily down to 15m
FRAMES = (("Day1", 5), ("Hour4", 4), ("Min60", 3), ("Min30", 2), ("Min15", 1))
PMO_L1, PMO_L2, PMO_SIG, PMO_SCALE = 35, 20, 10, 10
MACD_F, MACD_S, MACD_G = 12, 26, 9
THRESHOLD = 0.5
ZONE_TF, EMA_FAST, EMA_SLOW, ATR_LEN, BUF_ATR = "Min60", 21, 50, 14, 0.25
TRAIL_ATR = 3.0   # trailing stop distance in ATR(14) of the zone timeframe; target is open
MIN_BARS = 120


def tf_state(close: np.ndarray) -> int:
    """+1 both PMO>signal and MACD>signal on the last closed bar, -1 both below, 0 mixed/insufficient."""
    if len(close) < MIN_BARS:
        return 0
    p, sg, _ = pmo(close, PMO_L1, PMO_SCALE, PMO_L2, PMO_SIG)
    m = ema(close, MACD_F) - ema(close, MACD_S)
    ms = ema(m, MACD_G)
    vals = (p[-1], sg[-1], m[-1], ms[-1])
    if any(np.isnan(v) for v in vals):
        return 0
    a = 1 if vals[0] > vals[1] else -1
    b = 1 if vals[2] > vals[3] else -1
    return (a + b) // 2


def direction(states: dict, frames=FRAMES, thr: float = THRESHOLD) -> tuple[int, float]:
    wsum = sum(w for _, w in frames)
    score = sum(states.get(tf, 0) * w for tf, w in frames) / wsum if wsum else 0.0
    return (1 if score >= thr else -1 if score <= -thr else 0), score


def entry_zone(h, l, c, d: int):
    """(lo, hi) between EMA21 and EMA50 of the zone timeframe, plus ATR buffer on the far side; None if neutral/insufficient."""
    if d == 0 or len(c) < MIN_BARS:
        return None
    ef, es = ema(c, EMA_FAST)[-1], ema(c, EMA_SLOW)[-1]
    atr = atr_rma(h, l, c, ATR_LEN)[-1]
    if np.isnan(ef) or np.isnan(es) or np.isnan(atr):
        return None
    hi, lo = max(ef, es), min(ef, es)
    return (lo - BUF_ATR * atr, hi) if d == 1 else (lo, hi + BUF_ATR * atr)


def touches(zone, bar_low: float, bar_high: float) -> bool:
    return zone is not None and bar_low <= zone[1] and bar_high >= zone[0]


def last_atr(h, l, c) -> float:
    a = atr_rma(h, l, c, ATR_LEN)
    return float(a[-1]) if len(a) else float("nan")


def trail_stop(side: int, extreme: float, atr: float, mult: float = TRAIL_ATR) -> float:
    """Chandelier trail: highest price since entry - mult*ATR (long), lowest + mult*ATR (short). Caller ratchets it (never loosens)."""
    return extreme - mult * atr if side == 1 else extreme + mult * atr


def ratchet(side: int, old: float | None, new: float) -> float:
    if old is None:
        return new
    return max(old, new) if side == 1 else min(old, new)


# ───────── early mode: catch the turn BEFORE the move (the 5-frame score above lags by design) ─────────
EARLY_NEAR_ATR, EARLY_FAR_ATR, EARLY_MAX_EXT_ATR, EARLY_ETA_BARS = 0.5, 1.0, 2.0, 3.0


def pmo_gap(close: np.ndarray) -> np.ndarray:
    p, sg, _ = pmo(close, PMO_L1, PMO_SCALE, PMO_L2, PMO_SIG)
    return p - sg


def turn_state(gap: np.ndarray, d: int):
    """How the 15m PMO relates to its signal in direction d: ('crossed', bars_ago) | ('imminent', eta_bars) | None."""
    if len(gap) < 5 or np.isnan(gap[-3:]).any():
        return None
    x = gap * d
    if x[-1] > 0:
        return ("crossed", 0 if x[-2] <= 0 else 1) if (x[-2] <= 0 or x[-3] <= 0) else None
    speed = x[-1] - x[-2]
    if speed > 0 and -x[-1] / speed <= EARLY_ETA_BARS:
        return ("imminent", float(-x[-1] / speed))
    return None


def early_setup(states: dict, h1, l1, c1, gap15, price: float):
    """Bias from Day1+4H (must agree or one neutral), price pulled back to the 1H zone, not already extended, and the 15m PMO turning
    (about to cross / just crossed) in the bias direction. Returns dict or None."""
    bias = states.get("Day1", 0) + states.get("Hour4", 0)
    d = 1 if bias > 0 else -1 if bias < 0 else 0
    if d == 0 or len(c1) < MIN_BARS:
        return None
    zone = entry_zone(h1, l1, c1, d)
    atr = last_atr(h1, l1, c1)
    ef = ema(c1, EMA_FAST)[-1]
    if zone is None or not atr == atr:
        return None
    ext = (price - ef) / atr * d          # >0 = price already beyond the EMA in the trend direction (move under way)
    if ext > EARLY_MAX_EXT_ATR:
        return None
    # near the zone: inside, or within a band on the pullback side
    gap_to = (zone[0] - price) if d == 1 else (price - zone[1])   # >0 = still short of the zone
    if gap_to > EARLY_FAR_ATR * atr or -gap_to > EARLY_NEAR_ATR * atr * 6:
        return None
    turn = turn_state(gap15, d)
    if turn is None:
        return None
    return {"dir": d, "zone": zone, "turn": turn, "ext_atr": ext, "away_atr": gap_to / atr}


# ───────── breakout alert: a quiet range broken on a volume spike, in EITHER direction (parameters are untested guesses) ─────────
BO_LOOKBACK, BO_VOL_MULT, BO_MAX_WIDTH_PCT, BO_BODY_FRAC, BO_MAX_MOVED_PCT = 96, 4.0, 10.0, 0.6, 15.0


def breakout(o, h, l, c, v, lookback: int = BO_LOOKBACK):
    """Look at the last two CLOSED bars. A break = close beyond the high/low of the previous `lookback` bars (a quiet range no wider than
    BO_MAX_WIDTH_PCT), on volume >= BO_VOL_MULT x the median volume of that range, with a strong candle body. Direction = the side that broke.
    Returns dict(dir, age (0 = last closed bar), vol_x, beyond_pct, moved_pct, range_lo, range_hi) or None."""
    n = len(c)
    if n < lookback + 3:
        return None
    for age in (0, 1):
        i = n - 1 - age
        lo, hi = float(np.min(l[i - lookback:i])), float(np.max(h[i - lookback:i]))
        med = float(np.median(v[i - lookback:i]))
        if lo <= 0 or med <= 0 or (hi - lo) / lo * 100 > BO_MAX_WIDTH_PCT:
            continue
        rng = h[i] - l[i]
        if rng <= 0 or v[i] < BO_VOL_MULT * med:
            continue
        up, dn = c[i] > hi, c[i] < lo
        if not (up or dn):
            continue
        d = 1 if up else -1
        body = (c[i] - o[i]) * d / rng
        if body < BO_BODY_FRAC:
            continue
        moved = (c[-1] / o[i] - 1) * 100 * d
        if moved > BO_MAX_MOVED_PCT:
            continue
        return {"dir": d, "age": age, "vol_x": float(v[i] / med), "beyond_pct": float((c[i] / (hi if up else lo) - 1) * 100 * d),
                "moved_pct": float(moved), "range_lo": lo, "range_hi": hi}
    return None


# ───────── squeeze breakout (Bollinger) — adapts to each coin's own volatility; MA50/200 only classify (parameters are untested guesses) ─────────
SQ_BB_LEN, SQ_BB_K, SQ_HIST, SQ_PCT, SQ_VOL_MULT = 20, 2.0, 100, 0.20, 3.0
SQ_BODY_FRAC, SQ_MAX_MOVED_PCT, SQ_VOL_BARS = 0.6, 15.0, 96


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _bb(c: np.ndarray):
    mid = _sma(c, SQ_BB_LEN)
    sd = np.full(len(c), np.nan)
    for i in range(SQ_BB_LEN - 1, len(c)):
        sd[i] = np.std(c[i - SQ_BB_LEN + 1:i + 1])
    return mid, mid + SQ_BB_K * sd, mid - SQ_BB_K * sd


def squeeze_breakout(o, h, l, c, v):
    """Last two CLOSED bars. Squeeze = Bollinger bandwidth of the bar BEFORE the break is in the lowest SQ_PCT of the last SQ_HIST bars (relative to the coin itself).
    Break = close outside the pre-break band, volume >= SQ_VOL_MULT x median of the previous SQ_VOL_BARS bars, strong body, not already run > SQ_MAX_MOVED_PCT.
    Direction = the side that broke. Trend label from SMA50/SMA200: 'with' if beyond both on the break side, 'against' if beyond both on the other side, else 'mixed'.
    Returns dict(dir, age, vol_x, squeeze_pct, beyond_pct, moved_pct, trend) or None."""
    n = len(c)
    if n < 215:
        return None
    mid, up, lo = _bb(c)
    bw = (up - lo) / mid
    m50, m200 = _sma(c, 50), _sma(c, 200)
    for age in (0, 1):
        i = n - 1 - age
        hist = bw[i - SQ_HIST:i]                        # bandwidth of bars before the break bar
        if np.isnan(hist).any() or np.isnan(m200[i]):
            continue
        rank = float((hist < bw[i - 1]).mean())          # 0 = narrowest in history
        if rank > SQ_PCT:
            continue
        med = float(np.median(v[i - SQ_VOL_BARS:i]))
        rng = h[i] - l[i]
        if med <= 0 or rng <= 0 or v[i] < SQ_VOL_MULT * med:
            continue
        d = 1 if c[i] > up[i - 1] else -1 if c[i] < lo[i - 1] else 0
        if d == 0 or (c[i] - o[i]) * d / rng < SQ_BODY_FRAC:
            continue
        moved = (c[-1] / o[i] - 1) * 100 * d
        if moved > SQ_MAX_MOVED_PCT:
            continue
        above = int(c[i] > m50[i]) + int(c[i] > m200[i])
        trend = "mixed" if above == 1 else ("with" if (above == 2) == (d == 1) else "against")
        return {"dir": d, "age": age, "vol_x": float(v[i] / med), "squeeze_pct": rank * 100,
                "beyond_pct": float((c[i] / (up[i - 1] if d == 1 else lo[i - 1]) - 1) * 100 * d), "moved_pct": float(moved), "trend": trend}
    return None


def squeeze_precompute(o, h, l, c, v):
    """Vectorised indicators for scanning every bar of a history (same definitions as squeeze_breakout)."""
    from numpy.lib.stride_tricks import sliding_window_view as swv
    n = len(c)
    mid = _sma(c, SQ_BB_LEN)
    sd = np.full(n, np.nan)
    if n >= SQ_BB_LEN:
        sd[SQ_BB_LEN - 1:] = swv(c, SQ_BB_LEN).std(axis=1)
    up, lo = mid + SQ_BB_K * sd, mid - SQ_BB_K * sd
    return {"up": up, "lo": lo, "bw": (up - lo) / mid, "m50": _sma(c, 50), "m200": _sma(c, 200)}


def squeeze_at(pre, o, h, l, c, v, i):
    """Signal on closed bar i (age 0), or None. Mirrors squeeze_breakout without the 'already moved' filter (moved = 0 at the signal bar)."""
    if i < 215:
        return None
    hist = pre["bw"][i - SQ_HIST:i]
    if np.isnan(hist).any() or np.isnan(pre["m200"][i]) or np.isnan(pre["bw"][i - 1]):
        return None
    rank = float((hist < pre["bw"][i - 1]).mean())
    if rank > SQ_PCT:
        return None
    med = float(np.median(v[i - SQ_VOL_BARS:i]))
    rng = h[i] - l[i]
    if med <= 0 or rng <= 0 or v[i] < SQ_VOL_MULT * med:
        return None
    d = 1 if c[i] > pre["up"][i - 1] else -1 if c[i] < pre["lo"][i - 1] else 0
    if d == 0 or (c[i] - o[i]) * d / rng < SQ_BODY_FRAC:
        return None
    above = int(c[i] > pre["m50"][i]) + int(c[i] > pre["m200"][i])
    trend = "mixed" if above == 1 else ("with" if (above == 2) == (d == 1) else "against")
    return {"dir": d, "vol_x": float(v[i] / med), "squeeze_pct": rank * 100, "trend": trend}
