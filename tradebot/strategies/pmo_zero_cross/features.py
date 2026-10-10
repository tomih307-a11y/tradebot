"""Per-symbol feature arrays on the 15m timeline.

Higher-timeframe values at 15m bar t are those of the LAST CLOSED candle of that
frame whose close time <= close time of the 15m bar (no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tradebot.data.klines import drop_unclosed
from tradebot.indicators import atr_rma, pmo
from tradebot.mexc.client import TF_SECONDS


@dataclass
class Frame:
    """Indicators of one timeframe on its own candles (warm-up already masked)."""
    close_time: np.ndarray
    pmo: np.ndarray
    signal: np.ndarray
    slope: np.ndarray
    atr: np.ndarray


def compute_frame(k: dict, tf: str, ind) -> Frame:
    c, h, l = k["close"], k["high"], k["low"]
    p, s, sl = pmo(c, ind.roc_smooth_len, ind.roc_scale, ind.pmo_smooth_len, ind.signal_len)
    a = atr_rma(h, l, c, ind.atr_len)
    w = min(ind.warmup_bars, len(c))
    for arr in (p, s, sl, a):
        arr[:w] = np.nan                       # slope was computed before masking
    return Frame(k["time"] + TF_SECONDS[tf], p, s, sl, a)


def align(close15: np.ndarray, frame_close: np.ndarray):
    """For each 15m close time -> (index of last closed HTF candle or -1, closes_now flag)."""
    j = np.searchsorted(frame_close, close15, side="right") - 1
    closed_now = np.zeros(len(close15), dtype=bool)
    ok = j >= 0
    closed_now[ok] = frame_close[j[ok]] == close15[ok]
    return j, closed_now


def _take(arr: np.ndarray, j: np.ndarray) -> np.ndarray:
    out = np.full(len(j), np.nan)
    ok = j >= 0
    out[ok] = arr[j[ok]]
    return out


@dataclass
class SymbolFeatures:
    symbol: str
    ts: np.ndarray                      # 15m open times (epoch s)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    liq24: np.ndarray
    atr15: np.ndarray
    pmo15: np.ndarray
    signal15: np.ndarray
    slope15: np.ndarray
    pmo1h: np.ndarray
    signal1h: np.ndarray
    slope1h: np.ndarray
    atr1h: np.ndarray
    closed1h: np.ndarray
    j1h: np.ndarray
    pmo4h: np.ndarray
    signal4h: np.ndarray
    slope4h: np.ndarray
    closed4h: np.ndarray
    j4h: np.ndarray
    _idx: dict = field(default_factory=dict, repr=False)

    def index_of(self, ts: int):
        if not self._idx:
            self._idx = {int(t): i for i, t in enumerate(self.ts)}
        return self._idx.get(int(ts))


def build_features(symbol: str, k15: dict, k1h: dict, k4h: dict, cfg_strat, now: float | None = None,
                   liquidity_window: int = 96) -> SymbolFeatures:
    """now: epoch seconds; candles not yet closed at `now` are discarded (live). None = trust input."""
    tf15, tf1h, tf4h = cfg_strat.tf_entry, cfg_strat.tf_1h, cfg_strat.tf_4h
    k15 = drop_unclosed(k15, tf15, now)
    k1h = drop_unclosed(k1h, tf1h, now)
    k4h = drop_unclosed(k4h, tf4h, now)
    ind = cfg_strat.indicator
    f15, f1, f4 = compute_frame(k15, tf15, ind), compute_frame(k1h, tf1h, ind), compute_frame(k4h, tf4h, ind)
    close15 = f15.close_time
    j1, c1 = align(close15, f1.close_time)
    j4, c4 = align(close15, f4.close_time)

    amt = np.nan_to_num(k15["amount"])
    cs = np.cumsum(amt)
    liq = np.full(len(amt), np.nan)
    if len(amt) >= liquidity_window:
        liq[liquidity_window - 1:] = cs[liquidity_window - 1:] - np.concatenate(([0.0], cs[:-liquidity_window]))

    return SymbolFeatures(
        symbol=symbol, ts=k15["time"], open=k15["open"], high=k15["high"], low=k15["low"], close=k15["close"],
        liq24=liq, atr15=f15.atr, pmo15=f15.pmo, signal15=f15.signal, slope15=f15.slope,
        pmo1h=_take(f1.pmo, j1), signal1h=_take(f1.signal, j1), slope1h=_take(f1.slope, j1),
        atr1h=_take(f1.atr, j1), closed1h=c1, j1h=j1,
        pmo4h=_take(f4.pmo, j4), signal4h=_take(f4.signal, j4), slope4h=_take(f4.slope, j4),
        closed4h=c4, j4h=j4,
    )
