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
