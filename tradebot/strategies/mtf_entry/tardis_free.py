"""Study on Tardis FREE first-of-month days for MEXC futures (rules fixed in docs/TARDIS_FREE.md). Public data, no key.
Usage: python -m tradebot.strategies.mtf_entry.tardis_free OUT_FILE [workers]"""
from __future__ import annotations

import csv
import gzip
import io
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

DAYS = ["2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01", "2026-11-01"]
DESIGN = DAYS[:3]
TIERS = (10.0, 20.0, 30.0)
START_MIN, END_MIN, STEP, AHEAD = 360, 1440 - 60, 5, 60
SIGNALS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "K1", "K2", "K3", "K4"]
NAMES = {"S1": "ضغط شراء 15د", "S2": "قفزة نشاط", "S3": "مراكز جديدة +2%/س", "S4": "قرب القاع 6س", "S5": "تمويل سالب", "S6": "لم تبدأ (±2%/س)", "S7": "فرق سالب",
         "S9": "انفجار شراء 5د", "K1": "ضغط شراء+قفزة", "K2": "مراكز تتراكم والسعر ساكن", "K3": "انفجار شراء+لم تبدأ", "K4": "ضغط شراء+مراكز جديدة"}
BASE = "https://datasets.tardis.dev/v1/mexc-futures"


def ffill(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    first = np.where(~np.isnan(a))[0]
    if len(first):
        out[:first[0]] = np.nan
    return out


def day_arrays(trades_rows, ticker_rows, csize: float, day_start_us: int):
    """Per-minute (1440) arrays. trades_rows: iterable of (ts_us, side, price, amount); ticker_rows: (ts_us, oi, funding, index, mark) with NaN for missing."""
    n = 1440
    o, h, l, c = (np.full(n, np.nan) for _ in range(4))
    buy, sell, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    for ts, side, p, a in trades_rows:
        m = int((ts - day_start_us) // 60_000_000)
        if not 0 <= m < n:
            continue
        if np.isnan(o[m]):
            o[m] = p
            h[m] = l[m] = p
        h[m], l[m], c[m] = max(h[m], p), min(l[m], p), p
        usdt = p * a * csize
        if side == "buy":
            buy[m] += usdt
        else:
            sell[m] += usdt
        cnt[m] += 1
    oi, fund, idx, mark = (np.full(n, np.nan) for _ in range(4))
    for ts, o_i, f_, i_, m_ in ticker_rows:
        m = int((ts - day_start_us) // 60_000_000)
        if 0 <= m < n:
            if not np.isnan(o_i):
                oi[m] = o_i
            if not np.isnan(f_):
                fund[m] = f_
            if not np.isnan(i_):
                idx[m] = i_
            if not np.isnan(m_):
                mark[m] = m_
    cc = ffill(c)
    return {"o": o, "h": np.where(np.isnan(h), cc, h), "l": np.where(np.isnan(l), cc, l), "c": cc, "buy": buy, "sell": sell, "cnt": cnt,
            "oi": ffill(oi), "fund": ffill(fund), "idx": ffill(idx), "mark": ffill(mark)}


def signals_and_labels(d):
    """For every decision minute t in range: flags S*/K* and fmax (% max high of the next 60 minutes vs close_t)."""
    ts = np.arange(START_MIN, END_MIN, STEP)
    c, h = d["c"], d["h"]
    cum = lambda x: np.concatenate(([0.0], np.cumsum(x)))
    cb, cs_, cn = cum(d["buy"]), cum(d["sell"]), cum(d["cnt"])
    out = {k: np.zeros(len(ts), bool) for k in SIGNALS}
    fmax = np.full(len(ts), np.nan)
    ok = np.ones(len(ts), bool)
    tot = d["buy"] + d["sell"]
    c15 = np.array([tot[t - 14:t + 1].sum() for t in range(1440)])                       # turnover of the 15 minutes ending at t
    b5 = np.array([d["buy"][max(t - 4, 0):t + 1].sum() for t in range(1440)])
    for k, t in enumerate(ts):
        if np.isnan(c[t]) or np.isnan(c[t - 60]) or c[t] <= 0:
            ok[k] = False
            continue
        buy15, sell15 = cb[t + 1] - cb[t - 14], cs_[t + 1] - cs_[t - 14]
        n15 = cn[t + 1] - cn[t - 14]
        s1 = n15 >= 30 and (buy15 + sell15) > 0 and (buy15 - sell15) / (buy15 + sell15) >= 0.3
        base15 = np.median(c15[14:t])                                                    # 15-min turnovers earlier in the day (no look-ahead)
        s2 = base15 > 0 and c15[t] / base15 >= 4
        s3 = (not np.isnan(d["oi"][t])) and d["oi"][t - 60] > 0 and d["oi"][t] / d["oi"][t - 60] - 1 >= 0.02
        lo6, hi6 = np.nanmin(d["l"][t - 359:t + 1]), np.nanmax(d["h"][t - 359:t + 1])
        s4 = hi6 > lo6 and (c[t] - lo6) / (hi6 - lo6) <= 0.15
        s5 = (not np.isnan(d["fund"][t])) and d["fund"][t] <= -0.00005
        s6 = abs(c[t] / c[t - 60] - 1) <= 0.02
        s7 = (not np.isnan(d["mark"][t])) and (not np.isnan(d["idx"][t])) and d["idx"][t] > 0 and (d["mark"][t] - d["idx"][t]) / d["idx"][t] <= -0.001
        bb = np.array([b5[x] for x in range(14, t, 5)])
        bm = np.median(bb) if len(bb) else 0
        s9 = bm > 0 and b5[t] / bm >= 5
        out["S1"][k], out["S2"][k], out["S3"][k], out["S4"][k], out["S5"][k], out["S6"][k], out["S7"][k], out["S9"][k] = s1, s2, s3, s4, s5, s6, s7, s9
        out["K1"][k], out["K2"][k], out["K3"][k], out["K4"][k] = s1 and s2, s3 and s6, s9 and s6, s1 and s3
        fmax[k] = (np.nanmax(h[t + 1:t + 1 + AHEAD]) / c[t] - 1) * 100
    return {key: v[ok] for key, v in out.items()}, fmax[ok]


def http_get(url: str, tries: int = 4):
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "tradebot-study"}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** a)
        except Exception:
            time.sleep(2 ** a)
    return None


def fnum(x):
    try:
        return float(x) if x not in ("", None) else float("nan")
    except ValueError:
        return float("nan")


def load_one(args):
    sym, day, csize = args
    y, m, dd = day.split("-")
    try:
        t_raw = http_get(f"{BASE}/trades/{y}/{m}/{dd}/{sym}.csv.gz")
        k_raw = http_get(f"{BASE}/derivative_ticker/{y}/{m}/{dd}/{sym}.csv.gz")
        if not t_raw or not k_raw:
            return sym, day, None
        t_rows = [(int(r["timestamp"]), r["side"], float(r["price"]), float(r["amount"])) for r in csv.DictReader(io.StringIO(gzip.decompress(t_raw).decode()))]
        k_rows = [(int(r["timestamp"]), fnum(r["open_interest"]), fnum(r["funding_rate"]), fnum(r["index_price"]), fnum(r["mark_price"]))
                  for r in csv.DictReader(io.StringIO(gzip.decompress(k_raw).decode()))]
        if len(t_rows) < 200:
            return sym, day, None
        day_us = int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) * 1_000_000
        return sym, day, signals_and_labels(day_arrays(t_rows, k_rows, csize, day_us))
    except Exception as e:
        print("skip", sym, day, e, file=sys.stderr)
        return sym, day, None


def main(out, workers=8):
    from .paper import NON_CRYPTO

    detail = json.loads(http_get("https://contract.mexc.com/api/v1/contract/detail"))["data"]
    csize = {c["symbol"]: float(c.get("contractSize") or 1.0) for c in detail if c.get("quoteCoin") == "USDT"}
    cov = json.loads(http_get("https://api.tardis.dev/v1/exchanges/mexc-futures"))
    avail = {s["id"]: s["availableSince"][:10] for s in cov["availableSymbols"]}
    jobs = [(s, d, csize[s]) for s in sorted(csize) if s.endswith("_USDT") and not s.endswith("STOCK_USDT") and s not in NON_CRYPTO and s in avail
            for d in DAYS if avail[s] <= d]
    print(f"{len(jobs)} ملفاً (عقد×يوم)", flush=True)
    res = {d: {"flags": {k: [] for k in SIGNALS}, "fmax": [], "n": 0} for d in DAYS}
    with ThreadPoolExecutor(workers) as ex:
        for i, (sym, day, r) in enumerate(ex.map(load_one, jobs)):
            if r:
                flags, fmax = r
                res[day]["n"] += 1
                res[day]["fmax"].append(fmax)
                for k in SIGNALS:
                    res[day]["flags"][k].append(flags[k])
            if i % 200 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    lines = ["علامات لحظية قبل الانفجار الصاعد (Tardis، 5 أيام مجانية) · القواعد: docs/TARDIS_FREE.md · استكشافي", ""]
    A = {}
    for grp, days in (("design", DESIGN), ("hold", DAYS[3:])):
        fm = np.concatenate([x for d in days for x in res[d]["fmax"]]) if any(res[d]["fmax"] for d in days) else np.array([])
        A[grp] = (fm, {k: np.concatenate([x for d in days for x in res[d]["flags"][k]]) for k in SIGNALS} if len(fm) else {})
        lines.append(f"{grp}: أيام {days} · عقود×يوم {sum(res[d]['n'] for d in days)} · نقاط قرار {len(fm)} · انفجارات (نقاط) ≥10%: {int((fm >= 10).sum())} · ≥20%: {int((fm >= 20).sum())} · ≥30%: {int((fm >= 30).sum())}")
    lines += ["", f"{'الرمز':<4}{'العلامة':<26}{'معلَّمة(تصميم)':>14}{'معلَّمة(تحقق)':>14}{'≥10%':>8}{'رفع':>6}{'≥20%':>8}{'رفع':>6}{'≥30%':>8}{'رفع':>6}{'استدعاء10':>10}  (التحقق؛ الأساس بين قوسين)"]
    fh, flh = A["hold"]
    fd, fld = A["design"]
    if len(fh):
        base = [float((fh >= x).mean()) for x in TIERS]
        lines.append("الأساس (تحقق): " + " · ".join(f"≥{int(x)}%: {b * 100:.2f}%" for x, b in zip(TIERS, base)) + f"  |  (تصميم): " + " · ".join(f"≥{int(x)}%: {(fd >= x).mean() * 100:.2f}%" for x in TIERS))
        for k in SIGNALS:
            nd, nh = int(fld[k].sum()) if len(fd) else 0, int(flh[k].sum())
            if nh == 0:
                lines.append(f"{k:<4}{NAMES[k]:<26}{nd:>14}{nh:>14}   لا نقاط")
                continue
            hits = [float((fh[flh[k]] >= x).mean()) for x in TIERS]
            lifts = [hh / b if b else float("nan") for hh, b in zip(hits, base)]
            rec = float((flh[k] & (fh >= 10)).sum() / max((fh >= 10).sum(), 1))
            lines.append(f"{k:<4}{NAMES[k]:<26}{nd:>14}{nh:>14}{hits[0] * 100:>7.1f}%{lifts[0]:>6.1f}{hits[1] * 100:>7.1f}%{lifts[1]:>6.1f}{hits[2] * 100:>7.1f}%{lifts[2]:>6.1f}{rec * 100:>9.0f}%")
        lines += ["", "نفس العلامات على أيام التصميم (للمقارنة):"]
        bd = [float((fd >= x).mean()) for x in TIERS]
        for k in SIGNALS:
            if len(fd) and fld[k].sum():
                hits = [float((fd[fld[k]] >= x).mean()) for x in TIERS]
                lines.append(f"{k:<4}{NAMES[k]:<26}{int(fld[k].sum()):>14}   " + "  ".join(f"≥{int(x)}%: {hh * 100:.1f}% (×{hh / b if b else float('nan'):.1f})" for x, hh, b in zip(TIERS, hits, bd)))
    text = "\n".join(lines)
    print(text)
    open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
