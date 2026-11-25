"""Back-test of the channel's inferred rule + template (1H signals, ~75 crypto futures, Aug 20 - now).
Rule variants (evaluated at each 1H close): R0 = 1H close above SMA5/10/30 (1H) AND the 15m close above SMA5/10/30/60 (15m), freshly crossed (1H close 3 candles earlier was below the 1H SMA10);
R1 = R0 + last 1H volume >= 3x its 7-day median; R2 = R1 + at least 10% below the 7-day high; B = baseline (every 1H close).
Template: entry = 1H close; stop -5%; TPs +3.0 +6.4 +9.8 +13.2 +16.6 +20.0 %; 48h horizon; 15m candles (same-candle stop/TP ambiguity counts as stop). Costs 0.2% price.
Per-position result = money split equally over the 6 TPs (a tranche exits at its TP if reached before the stop, else at the stop (-5%) if hit, else at the horizon close). x20 = ROI on margin. Times UTC+4."""
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import scanner as S

now = int(time.time())
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX", "USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}
TPS = [3.0, 6.4, 9.8, 13.2, 16.6, 20.0]
STOP = 5.0
COST = 0.2


def fetch(sym, itv, step, a, b_end):
    out = {k: [] for k in ("time", "high", "low", "close", "vol")}
    while a < b_end:
        b = min(a + step * 1000, b_end)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={b}")["data"]; break
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


def sma(x, n):
    c = np.cumsum(np.insert(np.array(x, float), 0, 0))
    r = np.full(len(x), np.nan)
    r[n - 1:] = (c[n:] - c[:-n]) / n
    return r


def work(sym):
    start = int(datetime(2026, 8, 20, tzinfo=timezone.utc).timestamp())
    h1, m15 = fetch(sym, "Min60", 3600, start, now), fetch(sym, "Min15", 900, start, now)
    if not h1 or not m15 or len(h1["time"]) < 800 or len(m15["time"]) < 3000:
        return []
    c1, v1 = np.array(h1["close"]), np.array(h1["vol"])
    s1 = {n: sma(c1, n) for n in (5, 10, 30)}
    c15, hi15, lo15 = np.array(m15["close"]), np.array(m15["high"]), np.array(m15["low"])
    s15 = {n: sma(c15, n) for n in (5, 10, 30, 60)}
    t15 = {t: i for i, t in enumerate(m15["time"])}
    n15 = len(c15)
    rows, last = [], {"R0": -99, "R1": -99, "R2": -99, "B": -99}
    for i in range(700, len(c1) - 1):
        tc = h1["time"][i] + 3600
        if tc + 48 * 3600 > now:
            break
        j = t15.get(tc - 900)                                  # the 15m candle that closes exactly at the 1H close
        if j is None or j + 192 >= n15:
            continue
        price = c1[i]
        r0 = (price > s1[5][i] and price > s1[10][i] and price > s1[30][i] and c15[j] > max(s15[n][j] for n in (5, 10, 30, 60)) and c1[i - 3] < s1[10][i - 3])
        vmed = np.median(v1[i - 168:i])
        r1 = r0 and v1[i] >= 3 * (vmed or 1e-12)
        r2 = r1 and price / max(h1["high"][i - 168:i + 1]) - 1 <= -0.10
        flags = {"R0": r0, "R1": r1, "R2": r2, "B": True}
        # sequential simulation on 15m candles after the entry; stop checked first inside a candle (conservative)
        def run(policy):
            stop_lvl = -STOP
            nxt = 0
            exits = [None] * 6
            hit1 = None
            for k in range(j + 1, j + 193):
                if lo15[k] <= price * (1 + stop_lvl / 100):
                    for n in range(nxt, 6):
                        exits[n] = stop_lvl
                    return exits, hit1, True
                while nxt < 6 and hi15[k] >= price * (1 + TPS[nxt] / 100):
                    exits[nxt] = TPS[nxt]
                    if nxt == 0:
                        hit1 = k
                    if policy == 2 and nxt == 0:
                        stop_lvl = 0.0                              # stop to entry after TP1
                    if policy == 3:
                        stop_lvl = 0.0 if nxt == 0 else TPS[nxt - 1]  # stop to the previous TP after each TP
                    nxt += 1
            for n in range(nxt, 6):
                exits[n] = (end_px / price - 1) * 100
            return exits, hit1, False
        end_px = c15[j + 192]
        r1, h1_, st1 = run(1)
        r2, _, _ = run(2)
        r3, _, _ = run(3)
        p1, p2, p3 = (np.mean(r) - COST for r in (r1, r2, r3))
        tp_hit = [exits_ is not None and exits_ == TPS[n] for n, exits_ in enumerate(r1)]
        stop_before_tp1 = st1 and r1[0] == -STOP
        for k, f in flags.items():
            if f and i - last[k] >= 12:
                last[k] = i
                rows.append((k, tc, tp_hit, stop_before_tp1, (p1, p2, p3), sym))
    return rows


detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
        and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
with ThreadPoolExecutor(4) as ex:
    rows = [r for part in ex.map(work, tick) for r in part]
rng = np.random.default_rng(0)
half = np.median([r[1] for r in rows])
print(f"{len(tick)} عملة · شموع الساعة · 20 أغسطس → الآن · الدخول عند إغلاق الساعة · وقف -5% · أهداف +3/+6.4/+9.8/+13.2/+16.6/+20% · أفق 48 ساعة · تكلفة 0.2%")
print(f"\n{'القاعدة':40s} {'إشارات':>7s} | " + " ".join(f"{'TP'+str(n):>5s}" for n in range(1, 7)) + f" | {'وقف قبل TP1':>11s} | متوسط المركز % (×20 على الهامش): {'1) وقف 5% ثابت':>18s} {'2) وقف للدخول بعد TP1':>22s} {'3) وقف يلحق الأهداف':>20s}")
names = {"B": "الأساس: كل إغلاق ساعة", "R0": "R0: اختراق المتوسطات (15د+ساعة) حديث", "R1": "R1: R0 + حجم آخر ساعة ≥3×", "R2": "R2: R1 + ≥10% تحت قمة 7 أيام"}
for k in ("B", "R0", "R1", "R2"):
    g = [r for r in rows if r[0] == k]
    if len(g) < 30:
        print(f"{names[k]:40s} {len(g):7d}"); continue
    tp = np.array([r[2] for r in g]).mean(axis=0) * 100
    stp = np.mean([r[3] for r in g]) * 100
    P = np.array([r[4] for r in g]); ts = np.array([r[1] for r in g])
    def ci(col):
        wk = {}
        for tt, rr in zip(ts, P[:, col]):
            wk.setdefault(tt // (7 * 86400), []).append(rr)
        ks = list(wk)
        sm = np.array([sum(wk[x]) for x in ks]); cn = np.array([len(wk[x]) for x in ks])
        bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(300))]
        return f"{P[:, col].mean():+.2f}% ({P[:, col].mean()*20:+.0f}%) [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]"
    print(f"{names[k]:40s} {len(g):7d} | " + " ".join(f"{x:4.0f}%" for x in tp) + f" | {stp:10.0f}% | {ci(0):>30s} {ci(1):>30s} {ci(2):>30s}")
    print(f"{'':40s} {'':7s}   نصفا الفترة (سياسة 2): {P[ts<half][:,1].mean():+.2f}% / {P[ts>=half][:,1].mean():+.2f}%")
g = [r for r in rows if r[0] == "R1"]
bc = {}
for r in g:
    bc[r[5]] = bc.get(r[5], 0) + r[4][1]
tot = sum(bc.values())
print("\nتركّز العملات (R1): " + (", ".join(f"{k[:-5]} {v/tot*100:.0f}%" for k, v in sorted(bc.items(), key=lambda kv: -kv[1])[:6]) if tot > 0 else "المجموع سالب") + f" | عدد العملات: {len(bc)}")
