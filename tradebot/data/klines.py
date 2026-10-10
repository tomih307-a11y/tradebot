"""Kline fetching, on-disk caching, closed-candle filtering, funding history."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from tradebot.mexc.client import TF_SECONDS, MexcClient

FIELDS = ("time", "open", "high", "low", "close", "vol", "amount")


def empty_klines() -> dict:
    return {f: np.array([]) for f in FIELDS}


def _from_raw(d: dict) -> dict:
    t = np.asarray(d.get("time", []), dtype=np.int64)
    out = {"time": t}
    for f in FIELDS[1:]:
        arr = d.get(f)
        out[f] = np.asarray(arr, dtype=float) if arr is not None and len(arr) == len(t) else np.full(len(t), np.nan)
    if np.isnan(out["amount"]).all():            # fall back to close*vol when turnover absent
        out["amount"] = out["close"] * np.nan_to_num(out["vol"])
    return _sorted_unique(out)


def _sorted_unique(k: dict) -> dict:
    t = k["time"]
    if len(t) == 0:
        return k
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    keep = np.concatenate(([True], np.diff(t_sorted) > 0))
    idx = order[keep]
    return {f: k[f][idx] for f in FIELDS}


def drop_unclosed(k: dict, tf: str, now: float | None) -> dict:
    """Keep only candles whose close time (open + tf) <= now. now=None keeps everything."""
    if now is None or len(k["time"]) == 0:
        return k
    mask = (k["time"] + TF_SECONDS[tf]) <= now
    return {f: k[f][mask] for f in FIELDS}


def _cache_path(data_dir, symbol, tf) -> Path:
    return Path(data_dir) / f"{symbol}_{tf}.csv"


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return empty_klines()
    rows = list(csv.reader(path.open()))[1:]
    if not rows:
        return empty_klines()
    cols = list(zip(*rows))
    return _from_raw({"time": [int(x) for x in cols[0]],
                      **{f: [float(x) for x in cols[i + 1]] for i, f in enumerate(FIELDS[1:])}})


def _save_cache(path: Path, k: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for i in range(len(k["time"])):
            w.writerow([int(k["time"][i])] + [repr(float(k[c][i])) for c in FIELDS[1:]])


def fetch_range(client: MexcClient, symbol: str, tf: str, start: int, end: int) -> dict:
    """Paginated download of [start, end] (open times)."""
    step = TF_SECONDS[tf]
    chunk = client.page_limit * step
    parts, cur = [], start
    while cur <= end:
        stop = min(cur + chunk - step, end)
        parts.append(_from_raw(client.klines(symbol, tf, cur, stop)))
        cur = stop + step
    if not parts:
        return empty_klines()
    merged = {f: np.concatenate([p[f] for p in parts]) for f in FIELDS}
    return _sorted_unique(merged)


def get_klines(client: MexcClient, data_dir: str, symbol: str, tf: str, start: int, end: int,
               now: float | None = None, refresh: bool = True) -> dict:
    """Cached history for backtests. Only closed candles are ever stored/returned."""
    now = time.time() if now is None else now
    path = _cache_path(data_dir, symbol, tf)
    cached = _load_cache(path)
    step = TF_SECONDS[tf]
    if refresh:
        need_from = start
        if len(cached["time"]):
            if cached["time"][0] <= start:
                need_from = int(cached["time"][-1]) + step
        if need_from <= end:
            new = drop_unclosed(fetch_range(client, symbol, tf, need_from, end), tf, now)
            cached = _sorted_unique({f: np.concatenate([cached[f], new[f]]) for f in FIELDS})
            _save_cache(path, cached)
    m = (cached["time"] >= start) & (cached["time"] <= end)
    return {f: cached[f][m] for f in FIELDS}


def get_funding(client: MexcClient, data_dir: str, symbol: str, refresh: bool = True) -> list[tuple[int, float]]:
    path = Path(data_dir) / f"{symbol}_funding.json"
    if path.exists():
        rows = [tuple(r) for r in json.loads(path.read_text())]
        if not refresh:
            return rows
    try:
        rows = client.funding_history(symbol)
    except Exception:                               # offline / unavailable -> caller falls back to fixed rate
        rows = [tuple(r) for r in json.loads(path.read_text())] if path.exists() else []
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
    return rows
