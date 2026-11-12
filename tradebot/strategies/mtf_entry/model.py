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
