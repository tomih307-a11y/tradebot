"""Funding + slippage test of the 'spike then hold' rule. Triggers as spike_exits.py (15m volume >=4x usual, 15m return >=+2%), enter next 5m open, hold 4h (48 candles) or 1 day (288), hard stop -5%.
Variants: slippage 0.1/0.3/0.5% each side, funding on/off (MEXC 8h funding at 00/08/16 UTC; long pays rate*position). 0.2% fees. Triggers vs random control, week-block CI, halves."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scanner as S

now = int(time.time())
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX"}
DAYS = 45
rng = np.random.default_rng(3)
SLIPS = [0.1, 0.3, 0.5]
HOLDS = [48, 288]


def fetch(sym):
    a, out = now - DAYS * 86400, {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
    while a < now:
        b = min(a + 300 * 1000, now)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min5&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            return None
        for k in out:
            out[k] += d[k]
        a = b
    seen, keep = set(), []
    for i, t in enumerate(out["time"]):
        if t not in seen:
            seen.add(t); keep.append(i)
    return {k: np.array([out[k][i] for i in keep], float) for k in out}


def funding(sym):
    rec = {}
    for pg in (1, 2, 3):
        try:
            d = S.get(f"https://contract.mexc.com/api/v1/contract/funding_rate/history?symbol={sym}&page_num={pg}&page_size=100")["data"]
        except Exception:
            break
        for x in d.get("resultList", []):
            rec[int(x["settleTime"]) // 1000] = float(x["fundingRate"])
        if not d.get("resultList"):
            break
    return rec


def fsum(rec, t1, t2):
    """funding paid by a long between t1 and t2 (percent); settlement times snapped to the 8h grid"""
    tot, s = 0.0, (t1 // 28800 + 1) * 28800
    while s <= t2:
        best = min(rec, key=lambda k: abs(k - s)) if rec else None
        if best is not None and abs(best - s) < 1800:
            tot += rec[best] * 100
        s += 28800
    return tot


def sim(d, i, n, slip, rec, use_f):
    H, L, C, T = d["high"], d["low"], d["close"], d["time"]
    e = d["open"][i + 1] * (1 + slip / 100)
    sl = e * 0.95
    for k in range(i + 1, i + 1 + n):
        if L[k] <= sl:
            px, tx = sl, T[k]; break
    else:
        px, tx = C[i + n], T[i + n]
    pct = (px * (1 - slip / 100) / e - 1) * 100 - 0.2
    if use_f:
        pct -= fsum(rec, int(T[i + 1]), int(tx))
    return pct


def work(sym):
    d = fetch(sym)
    if d is None or len(d["time"]) < 3000:
        return [], {}
    rec = funding(sym)
    c, v = d["close"], d["vol"]
    n = len(c)
    pts, last = [], -99
    for i in range(300, n - 600):
        base = np.median(v[i - 288:i - 2]) * 3 or 1e-9
        if (v[i] + v[i - 1] + v[i - 2]) / base >= 4 and (c[i] / c[i - 3] - 1) * 100 >= 2 and i - last >= 12:
            last = i; pts.append((i, "T"))
    pts += [(int(rng.integers(300, n - 600)), "C") for _ in range(3 * len(pts))]
    out = []
    for i, kind in pts:
        vals = [sim(d, i, h, s, rec, f) for h in HOLDS for s in SLIPS for f in (False, True)]
        out.append((sym, int(d["time"][i]), kind, vals, fsum(rec, int(d["time"][i + 1]), int(d["time"][i + 1]) + 4 * 3600)))
    return out, rec


detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
        and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
with ThreadPoolExecutor(6) as ex:
    parts = list(ex.map(work, tick))
res = [r for p, _ in parts for r in p]
nf = sum(1 for _, rec in parts if rec)
T = [r for r in res if r[2] == "T"]; Cn = [r for r in res if r[2] == "C"]
print(f"عملات {len(tick)} (تمويل متوفر لـ {nf}) · إشارات {len(T)} · عشوائي {len(Cn)}")
fav = [r[4] for r in T]
print(f"متوسط تمويل نافذة 4 ساعات على الإشارات: {np.mean(fav):+.3f}% (أعلى 5%: {np.percentile(fav, 95):+.3f}% · أدنى 5%: {np.percentile(fav, 5):+.3f}%)\n")
ts = sorted(r[1] for r in T); mid = ts[len(ts) // 2]
wk = np.array([r[1] // (7 * 86400) for r in T]); wu = np.unique(wk)
print(f"{'الإمساك':8s} {'انزلاق':7s} {'تمويل':6s} {'إشارات':>8s} {'عشوائي':>8s} {'CI إشارات':>15s} {'نصف1':>7s} {'نصف2':>7s}")
j = 0
for h in HOLDS:
    for s in SLIPS:
        for f in (False, True):
            a = np.array([r[3][j] for r in T]); b = np.array([r[3][j] for r in Cn])
            bs = [np.concatenate([a[wk == w] for w in rng.choice(wu, len(wu))]).mean() for _ in range(800)]
            h1 = np.mean([r[3][j] for r in T if r[1] < mid]); h2 = np.mean([r[3][j] for r in T if r[1] >= mid])
            print(f"{'4 ساعات' if h == 48 else 'يوم':8s} {s:5.1f}% {'نعم' if f else 'لا':6s} {a.mean():+7.2f}% {b.mean():+7.2f}% {np.percentile(bs, 2.5):+6.2f}..{np.percentile(bs, 97.5):+5.2f} {h1:+6.2f}% {h2:+6.2f}%")
            j += 1
