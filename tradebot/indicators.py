"""TradingView-compatible indicators. All inputs are numpy arrays of CLOSED candles."""
from __future__ import annotations

import numpy as np


def roc(close: np.ndarray) -> np.ndarray:
    """roc[t] = 100 * (close[t] - close[t-1]) / close[t-1]; roc[0] = NaN."""
    out = np.full(len(close), np.nan)
    if len(close) > 1:
        out[1:] = 100.0 * (close[1:] - close[:-1]) / close[:-1]
    return out


def csf(x: np.ndarray, n: float) -> np.ndarray:
    """Custom smoothing: sm[0]=x[0]; sm[t]=sm[t-1]+(x[t]-sm[t-1])*(2/n).
    Seeded at the first non-NaN value (roc[0] is undefined); NaNs before it stay NaN."""
    out = np.full(len(x), np.nan)
    alpha = 2.0 / n
    prev = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            if not np.isnan(prev):
                out[i] = prev  # carry through an interior gap
            continue
        prev = v if np.isnan(prev) else prev + (v - prev) * alpha
        out[i] = prev
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """TradingView ta.ema: alpha = 2/(n+1), seeded with the first value."""
    out = np.full(len(x), np.nan)
    alpha = 2.0 / (n + 1)
    prev = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        prev = v if np.isnan(prev) else prev + (v - prev) * alpha
        out[i] = prev
    return out


def pmo(close: np.ndarray, roc_len: float, scale: float, pmo_len: float, signal_len: int):
    """Returns (PMO, SIGNAL, SLOPE) un-masked (caller applies warm-up masking)."""
    r = roc(close)
    p = csf(scale * csf(r, roc_len), pmo_len)
    sig = ema(p, signal_len)
    slope = np.full(len(p), np.nan)
    slope[1:] = p[1:] - p[:-1]
    return p, sig, slope


def atr_rma(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """TradingView ta.atr: RMA(TR, n), alpha = 1/n, seeded with SMA of first n TRs.
    TR[0] = high-low (no previous close)."""
    L = len(close)
    tr = np.empty(L)
    if L == 0:
        return tr
    tr[0] = high[0] - low[0]
    if L > 1:
        pc = close[:-1]
        tr[1:] = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - pc), np.abs(low[1:] - pc)])
    out = np.full(L, np.nan)
    if L < n:
        return out
    out[n - 1] = tr[:n].mean()
    for i in range(n, L):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out
