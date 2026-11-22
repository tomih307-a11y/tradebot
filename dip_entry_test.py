"""Entry AFTER a sharp fall with a reversal confirmation (1H, ~75 crypto futures coins, Aug 20 - now), wider structural stop, several exits, time halves, random-entry control.
Setup (any): fall >=15% within 24h | >=30% below the 7-day high | >=10% below TEMA50. De-duplicated every 24 candles per coin. Confirmation within 12 candles after the setup:
 C0 none (enter at the setup close) | C1 higher low: green candle closing above the previous high with a low above the recent low | C2 PMO(1H) crosses above its orange below zero
 C3 reversal candle: closes above the last two highs after a red candle | C4 MACD(1H) crosses up below zero.   Entry = confirmation candle close.
Stop = under the lowest low of the setup window (-0.5%), distance clamped 2%..15% (wider skipped). Exits (72h max): X10 target +10% | X25 target +25% | XT trailing stop 8% below the highest high (floor = stop) | X48 hold 48h.
Costs 0.2% round trip. Control: random 1H candle with the same stop distance and exit. Times UTC+4."""
import random
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import scanner as S

now = int(time.time())
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX", "USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}
CONF = ("C0", "C1", "C2", "C3", "C4")
EXITS = ("X10", "X25", "XT", "X48")


def fetch(sym, a, b_end):
    out = {k: [] for k in ("time", "open", "high", "low", "close")}
    while a < b_end:
        b = min(a + 3600 * 1000, b_end)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min60&start={a}&end={b}")["data"]; break
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
    return {k: [out[k][i] for i in keep] for k in out}


def sim(ent, stp, e, hi, lo, c, n, ex):
    """returns net % (cost included)"""
    hh = ent
    for j in range(e + 1, min(e + 73, n)):
        if ex == "XT":
            lvl = max(stp, hh * 0.92)
            if lo[j] <= lvl:
                return (lvl / ent - 1) * 100 - 0.2
            hh = max(hh, hi[j])
            continue
        if lo[j] <= stp:
            return (stp / ent - 1) * 100 - 0.2
        if ex == "X10" and hi[j] >= ent * 1.10:
            return 10.0 - 0.2
        if ex == "X25" and hi[j] >= ent * 1.25:
            return 25.0 - 0.2
        if ex == "X48" and j == e + 48:
            return (c[j] / ent - 1) * 100 - 0.2
    return (c[min(e + 72, n - 1)] / ent - 1) * 100 - 0.2


def work(sym):
    d = fetch(sym, int(datetime(2026, 8, 20, tzinfo=timezone.utc).timestamp()), now)
    if not d or len(d["time"]) < 800:
        return []
    t, o, hi, lo, c = (d[k] for k in ("time", "open", "high", "low", "close"))
    n = len(c)
    p, ps = S.pmo(c)
    te = S.tema(c, 50)
    m = [a - b for a, b in zip(S.ema(c, 12), S.ema(c, 26))]
    g = S.ema(m, 9)
    rnd = random.Random(hash(sym) % 100000)
    rows, last = [], -99
    for i in range(400, n - 100):
        setup = c[i] / max(hi[i - 24:i]) - 1 <= -0.15 or c[i] / max(hi[i - 168:i]) - 1 <= -0.30 or c[i] / te[i] - 1 <= -0.10
        if not setup or i - last < 24:
            continue
        last = i
        found = {}
        for j in range(i, i + 13):
            L = min(lo[max(i - 5, 0):j])if j > i else lo[i]
            if "C0" not in found and j == i:
                found["C0"] = j
            if j == i:
                continue
            green = c[j] > o[j]
            if "C1" not in found and green and c[j] > hi[j - 1] and lo[j] > L:
                found["C1"] = j
            if "C2" not in found and p[j - 1] <= ps[j - 1] and p[j] > ps[j] and p[j] < 0:
                found["C2"] = j
            if "C3" not in found and green and c[j] > max(hi[j - 1], hi[j - 2]) and c[j - 1] < o[j - 1]:
                found["C3"] = j
            if "C4" not in found and m[j - 1] <= g[j - 1] and m[j] > g[j] and m[j] < 0:
                found["C4"] = j
        for cf, j in found.items():
            ent = c[j]
            stp = min(lo[max(i - 5, 0):j + 1]) * 0.995
            dist = ent / stp - 1
            if dist > 0.15:
                continue
            if dist < 0.02:
                stp = ent * 0.98; dist = 0.02
            for ex in EXITS:
                res = sim(ent, stp, j, hi, lo, c, n, ex)
                rb = rnd.randrange(400, n - 100)                                # control: random candle, same stop distance & exit
                e2 = c[rb]
                res2 = sim(e2, e2 * (1 - dist), rb, hi, lo, c, n, ex)
                rows.append((cf, ex, t[j], res, res2, dist * 100, sym))
    return rows


detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
        and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
with ThreadPoolExecutor(4) as ex_:
    rows = [r for part in ex_.map(work, tick) for r in part]
rng = np.random.default_rng(0)
half = np.median([r[2] for r in rows])
print(f"{len(tick)} عملة · شموع الساعة · 20 أغسطس → الآن · الدخول بعد هبوط حاد + تأكيد · وقف تحت قاع الهبوط · تكلفة 0.2%")
print(f"{'التأكيد':26s} {'الخروج':>7s} {'صفقات':>6s} {'وسيط الوقف':>10s} | {'رابحة':>6s} {'ضُرب الوقف':>9s} {'متوسط %':>8s} {'وسيط %':>7s} {'ثقة 95%':>16s} | {'الضابط %':>8s} {'الفرق':>7s} | {'نصف1':>7s} {'نصف2':>7s}")
names = {"C0": "بلا تأكيد (إغلاق الهبوط)", "C1": "قاع أعلى + شمعة صاعدة", "C2": "PMO 1س يعبر البرتقالي", "C3": "شمعة انعكاس", "C4": "تقاطع MACD صاعد"}
exn = {"X10": "هدف 10%", "X25": "هدف 25%", "XT": "وقف متحرك 8%", "X48": "48 ساعة"}
for cf in CONF:
    for ex in EXITS:
        g = [r for r in rows if r[0] == cf and r[1] == ex]
        if len(g) < 40:
            continue
        a = np.array([r[3] for r in g]); b = np.array([r[4] for r in g]); ts = np.array([r[2] for r in g])
        wk = {}
        for tt, rr in zip(ts, a):
            wk.setdefault(tt // (7 * 86400), []).append(rr)
        ks = list(wk)
        sm = np.array([sum(wk[x]) for x in ks]); cn = np.array([len(wk[x]) for x in ks])
        bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(300))]
        h1 = a[ts < half].mean(); h2 = a[ts >= half].mean()
        stopr = np.mean([r[3] <= -(r[5] - 0.1) for r in g]) * 100
        print(f"{names[cf]:26s} {exn[ex]:>13s} {len(g):6d} {st.median(r[5] for r in g):9.1f}% | {np.mean(a>0)*100:5.0f}% {stopr:8.0f}% {a.mean():+7.2f}% {np.median(a):+6.2f}% [{np.percentile(bs,2.5):+5.2f},{np.percentile(bs,97.5):+5.2f}] | {b.mean():+7.2f}% {a.mean()-b.mean():+6.2f} | {h1:+6.2f}% {h2:+6.2f}%")
    print()
best = [r for r in rows if r[0] == "C1" and r[1] == "XT"]
bc = {}
for r in best:
    bc[r[6]] = bc.get(r[6], 0) + r[3]
tot = sum(bc.values())
print("تركّز عملات (C1 + وقف متحرك): ", ", ".join(f"{k[:-5]} {v/tot*100:.0f}%" for k, v in sorted(bc.items(), key=lambda kv: -kv[1])[:6]) if tot > 0 else "المجموع سالب", "| عدد العملات:", len({r[6] for r in best}))
