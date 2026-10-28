"""PMO zero-touch gallery + event study on MEXC 15m data.
 - Events: the FIRST bar where PMO reaches the zero line (from below = 'up', from above = 'down'), for PMO(35,20,10) and PMO(4,20,10).
 - Gallery: random real examples (liquid contracts) with candles, both PMOs, the touch bar and what happened afterwards.
 - Event study: signed forward return after the touch vs the same symbol's average bar (excess), at several horizons, with a day-block
   bootstrap CI, plus the move size after the touch vs normal (does volatility react?).
Output: <out>/gallery.json   Usage: python tools/gallery.py --out reports/gallery --days 150 --top 150
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.config import load_config                  # noqa: E402
from tradebot.data.klines import get_klines               # noqa: E402
from tradebot.indicators import pmo                       # noqa: E402
from tradebot.mexc.client import MexcClient              # noqa: E402

M15 = 900
HORIZONS = (1, 2, 4, 8, 16, 32)
WARM = 300
LENS = {"35": 35, "4": 4}


def touches(p: np.ndarray):
    """-> (up_idx, down_idx): first bar reaching zero from below / from above."""
    a, b = p[:-1], p[1:]
    ok = ~np.isnan(a) & ~np.isnan(b)
    up = np.where(ok & (a < 0) & (b >= 0))[0] + 1
    dn = np.where(ok & (a > 0) & (b <= 0))[0] + 1
    return up, dn


def fwd(close: np.ndarray, j: int, h: int):
    return close[j + h] / close[j] - 1.0 if j + h < len(close) else np.nan


def boot(vals_by_day, n=2000, seed=0):
    days = [np.asarray(v) for v in vals_by_day.values() if len(v)]
    if not days:
        return (float("nan"), float("nan"))
    sums, cnt = np.array([d.sum() for d in days]), np.array([len(d) for d in days], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), size=(n, len(days)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/gallery")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--examples", type=int, default=8, help="examples per direction and PMO length")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cfg = load_config(a.config)
    client = MexcClient(cfg.mexc)
    now = int(time.time())
    start = now - a.days * 86400
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    symbols = [t["symbol"] for t in tick[: a.top]]
    dd = cfg.pmo_zero_cross.backtest.data_dir
    ind = cfg.pmo_zero_cross.indicator
    rng = np.random.default_rng(a.seed)
    # per PMO length and direction: lists of events (excess signed forward returns) + example pool
    acc = {(L, d): {"ex": {h: [] for h in HORIZONS}, "day": {h: {} for h in HORIZONS}, "absr": [], "n": 0}
           for L in LENS for d in ("up", "down")}
    pool = {(L, d): [] for L in LENS for d in ("up", "down")}
    used = 0
    for n_, sym in enumerate(symbols, 1):
        try:
            k = get_klines(client, dd, sym, "Min15", start - 500 * M15, now)
        except Exception as e:
            print(f"  {sym}: data failed {e}", flush=True)
            continue
        c = k["close"]
        if len(c) < 1500:
            continue
        used += 1
        pm = {}
        for L, ln in LENS.items():
            p, sg, _ = pmo(c, ln, ind.roc_scale, ind.pmo_smooth_len, ind.signal_len)
            p, sg = p.copy(), sg.copy()
            p[:WARM] = np.nan
            sg[:WARM] = np.nan
            pm[L] = (p, sg)
        base = {h: np.nanmean(c[h:] / c[:-h] - 1.0) for h in HORIZONS}
        base_abs4 = np.nanmean(np.abs(c[4:] / c[:-4] - 1.0))
        for L in LENS:
            up, dn = touches(pm[L][0])
            for d, idxs, sgn in (("up", up, 1), ("down", dn, -1)):
                for j in idxs:
                    if k["time"][j] < start or j + max(HORIZONS) >= len(c):
                        continue
                    A = acc[(L, d)]
                    A["n"] += 1
                    day = int(k["time"][j]) // 86400
                    for h in HORIZONS:
                        ex = sgn * (fwd(c, j, h) - base[h])
                        A["ex"][h].append(ex)
                        A["day"][h].setdefault(day, []).append(ex)
                    f4 = abs(fwd(c, j, 4))
                    A["absr"].append(f4 / base_abs4 if base_abs4 > 0 else np.nan)
                    if n_ <= 40 and j >= 80:
                        pool[(L, d)].append((sym, int(j)))
        if n_ % 25 == 0:
            print(f"  [{n_}/{len(symbols)}] {sym}", flush=True)
        if n_ <= 40:
            pm_cache = pm
            pool.setdefault("_data", {})[sym] = {"k": {f: k[f] for f in ("time", "open", "high", "low", "close")}, "pm": pm_cache}

    stats = {}
    for (L, d), A in acc.items():
        row = {"events": A["n"], "mean_abs_move_ratio_4bars": float(np.nanmean(A["absr"])) if A["absr"] else None, "h": {}}
        for h in HORIZONS:
            x = np.asarray(A["ex"][h])
            if len(x) == 0:
                continue
            lo, hi = boot(A["day"][h])
            row["h"][str(h)] = {"mean_pct": float(x.mean() * 100), "ci_lo": lo * 100, "ci_hi": hi * 100,
                                "win_pct": float((x > 0).mean() * 100), "n": int(len(x))}
        stats[f"{L}_{d}"] = row

    examples = {}
    data = pool.pop("_data", {})
    for (L, d), lst in pool.items():
        if not lst:
            continue
        pick = rng.choice(len(lst), size=min(a.examples, len(lst)), replace=False)
        out = []
        for i in pick:
            sym, j = lst[int(i)]
            dat = data[sym]
            k, (p35, s35), (p4, s4) = dat["k"], dat["pm"]["35"], dat["pm"]["4"]
            lo, hi = j - 80, min(len(k["close"]), j + 41)
            rnd = lambda arr, nd: [None if np.isnan(v) else round(float(v), nd) for v in arr[lo:hi]]
            cl = k["close"]
            out.append({
                "symbol": sym, "touch": j - lo,
                "time": datetime.fromtimestamp(int(k["time"][j]) + M15, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "o": rnd(k["open"], 8), "h": rnd(k["high"], 8), "l": rnd(k["low"], 8), "c": rnd(k["close"], 8),
                "pmo35": rnd(p35, 4), "sig35": rnd(s35, 4), "pmo4": rnd(p4, 4), "sig4": rnd(s4, 4),
                "fwd": {str(h): (None if j + h >= len(cl) else round(float(cl[j + h] / cl[j] - 1) * 100, 3)) for h in (4, 8, 16, 32)},
            })
        examples[f"{L}_{d}"] = sorted(out, key=lambda e: e["time"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    meta = {"symbols_used": used, "days": a.days, "top": a.top, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "pmo": {"roc_scale": ind.roc_scale, "smooth": ind.pmo_smooth_len, "signal": ind.signal_len}}
    (Path(a.out) / "gallery.json").write_text(json.dumps({"meta": meta, "stats": stats, "examples": examples}, ensure_ascii=False, separators=(",", ":")))
    print(json.dumps({"meta": meta, "stats": stats}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
