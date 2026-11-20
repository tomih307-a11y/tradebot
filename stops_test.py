"""PMO-only test (35,20,10). No MACD / TEMA / order block. Entries use PMO states on 15m and larger frames only; exits use PMO only (+hard stop, max 24h).
Entry variants: touch (blue crosses zero up), alert (blue>orange, rising, below zero, <=3 candles of current speed from zero), each with larger-frame PMO filters.
Exits: E1 = 15m blue crosses below orange; E2 = 15m blue falls back below zero. Baseline = every 15m bar with the same exits. 70 days, net of 0.1% costs. R = pnl / stop distance (stop = 1.5 ATR clamp 0.5-6%)."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scanner as S

SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}
DAYS = {"Min15": 70, "Min60": 130, "Hour4": 300, "Day1": 300}
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX"}


def kl(sym, itv, now):
    out = {k: [] for k in ("time", "high", "low", "close")}
    a = now - DAYS[itv] * 86400
    while a < now:
        b = min(a + SEC[itv] * 1000, now)
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
    seen, idx = set(), []
    for i, tt in enumerate(out["time"]):
        if tt not in seen and tt + SEC[itv] <= now:
            seen.add(tt); idx.append(i)
    return {k: [out[k][i] for i in idx] for k in out}


def fstate(d, per, tc, price):
    times, closes = d["time"], d["close"]
    lo_, hi_ = 0, len(times)
    while lo_ < hi_:
        mid = (lo_ + hi_) // 2
        if times[mid] + per <= tc:
            lo_ = mid + 1
        else:
            hi_ = mid
    series = closes[max(0, lo_ - 700):lo_] + [price]
    if len(series) < 120:
        return None
    p, s = S.pmo(series)
    n = len(series) - 1
    return {"pos": p[n] > 0, "above": p[n] > s[n], "up": p[n] > p[n - 1]}


def work(sym):
    now = int(time.time())
    k15, kd = kl(sym, "Min15", now), kl(sym, "Day1", now)
    if not (k15 and kd):
        return []
    t, hi, lo, c = k15["time"], k15["high"], k15["low"], k15["close"]
    n = len(c)
    p, ps = S.pmo(c)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    rows = []
    last = -99
    for i in range(400, n - 97):
        if not (p[i - 1] <= 0 < p[i]) or i - last < 4:
            continue
        last = i
        ent = c[i]
        d1 = fstate(kd, 86400, t[i] + 900, ent)
        orig = max(min(ent - 1.5 * atr[i], ent * 0.995), ent * 0.94)
        risk = ent - orig
        row = {"t": t[i], "match": bool(d1 and d1["pos"]), "riskpct": risk / ent * 100}
        for name in ("orig", "none", "entry", "be1r", "wide3"):
            stp = {"orig": orig, "none": 0.0, "entry": ent, "be1r": orig, "wide3": max(ent - 3 * atr[i], ent * 0.88)}[name]
            moved = False
            out = None
            for j in range(i + 1, i + 97):
                if lo[j] <= stp:
                    out = stp; break
                if p[j] < ps[j] and p[j - 1] >= ps[j - 1]:
                    out = c[j]; break
                if name == "be1r" and not moved and hi[j] >= ent + risk:
                    stp = ent; moved = True
            if out is None:
                out = c[i + 96]
            pct = (out / ent - 1) * 100 - 0.1
            row[name] = pct
            row[name + "_R"] = pct / (risk / ent * 100)
        rows.append(row)
    return rows


def main():
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    with ThreadPoolExecutor(5) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    rng = np.random.default_rng(0)
    names = {"orig": "وقف 1.5×ATR (الحالي)", "none": "بلا وقف", "entry": "وقف عند سعر الدخول", "be1r": "وقف ينتقل للدخول بعد +1R", "wide3": "وقف واسع 3×ATR"}

    def line(sel, label):
        g = [r for r in rows if sel(r)]
        if len(g) < 30:
            print(f"{label}: {len(g)}"); return
        print(f"\n{label} — {len(g)} صفقة (وسيط مسافة الوقف الأصلي {np.median([r['riskpct'] for r in g]):.1f}%)")
        print(f"   {'':26s} {'متوسط %':>9s} {'متوسط R':>9s} {'حدّ الثقة R 95%':>18s} {'رابحة':>7s} {'أسوأ صفقة %':>12s} {'وسيط %':>8s}")
        wk = {}
        for k in names:
            xs = np.array([r[k] for r in g]); xr = np.array([r[k + "_R"] for r in g])
            slots = {}
            for r in g:
                slots.setdefault(r["t"], []).append(r[k + "_R"])
            ks = list(slots)
            sm = np.array([sum(slots[x]) for x in ks]); cn = np.array([len(slots[x]) for x in ks])
            bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(300))]
            print(f"   {names[k]:26s} {xs.mean():+9.3f} {xr.mean():+9.3f} [{np.percentile(bs,2.5):+6.2f},{np.percentile(bs,97.5):+6.2f}] {np.mean(xs>0)*100:6.0f}% {xs.min():+11.1f} {np.median(xs):+8.2f}")

    print(f"{len(tick)} عملة · 70 يوماً · لمس PMO للصفر 15د · خروج: الأزرق يقطع البرتقالي هبوطاً أو 24 ساعة · تكلفة 0.1% · R = العائد ÷ مسافة وقف 1.5×ATR")
    line(lambda r: True, "كل اللمسات")
    line(lambda r: r["match"], "تطابق القاعدة (اليومي فوق الصفر)")
    line(lambda r: not r["match"], "اليومي تحت الصفر")


main()
