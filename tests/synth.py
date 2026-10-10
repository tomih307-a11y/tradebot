"""Deterministic synthetic candles (seeded) used by the tests — no network needed."""
from __future__ import annotations

import numpy as np

from tradebot.data.klines import FIELDS


def _agg(k15: dict, step: int) -> dict:
    t = k15["time"]
    g = (t // step)
    out = {f: [] for f in FIELDS}
    for gv in np.unique(g):
        m = g == gv
        if m.sum() != step // 900:
            continue                                   # incomplete group -> not a closed candle
        out["time"].append(int(gv * step))
        out["open"].append(k15["open"][m][0])
        out["high"].append(k15["high"][m].max())
        out["low"].append(k15["low"][m].min())
        out["close"].append(k15["close"][m][-1])
        out["vol"].append(k15["vol"][m].sum())
        out["amount"].append(k15["amount"][m].sum())
    return {f: np.asarray(out[f], dtype=np.int64 if f == "time" else float) for f in FIELDS}


def synth_klines(seed: int, days: int, start: int = 0, p0: float = 100.0):
    rng = np.random.default_rng(seed)
    n = days * 96
    regime = np.zeros(n)
    i = 0
    while i < n:
        L = int(rng.integers(40, 400))
        regime[i:i + L] = rng.normal(0, 0.0012)
        i += L
    ret = regime + rng.normal(0, 0.003, n)
    close = p0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate(([p0], close[:-1]))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0015, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0015, n)))
    vol = rng.uniform(1e3, 5e3, n)
    k15 = {"time": start + np.arange(n, dtype=np.int64) * 900, "open": open_, "high": high, "low": low,
           "close": close, "vol": vol, "amount": vol * close}
    return k15, _agg(k15, 3600), _agg(k15, 14400)
