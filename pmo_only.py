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
    k15, k1, k4, kd = kl(sym, "Min15", now), kl(sym, "Min60", now), kl(sym, "Hour4", now), kl(sym, "Day1", now)
    if not (k15 and k1 and k4 and kd):
        return []
    t, hi, lo, c = k15["time"], k15["high"], k15["low"], k15["close"]
    n = len(c)
    p, ps = S.pmo(c)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    rows = []
    last_touch = last_alert = -99
    for i in range(400, n - 97):
        touch = p[i - 1] <= 0 < p[i]
        sp = p[i] - p[i - 1]
        alert = p[i] < 0 and p[i] > ps[i] and sp > 0 and -p[i] <= 3 * sp
        sel_touch = touch and i - last_touch >= 4
        sel_alert = alert and i - last_alert >= 8 and not touch
        if touch: last_touch = i
        if alert: last_alert = i
        base_ok = (i % 4 == 0)                                                      # baseline: every hour on the hour
        if not (sel_touch or sel_alert or base_ok):
            continue
        ent = c[i]
        tc = t[i] + 900
        row = {"t": t[i], "touch": sel_touch, "alert": sel_alert, "base": base_ok}
        if sel_touch or sel_alert:
            row["h1"], row["h4"], row["d1"] = fstate(k1, 3600, tc, ent), fstate(k4, 14400, tc, ent), fstate(kd, 86400, tc, ent)
        stp = max(min(ent - 1.5 * atr[i], ent * 0.995), ent * 0.94)
        risk = ent - stp
        cost = 0.001 * ent / risk
        for ex in ("E1", "E2"):
            R = None
            for j in range(i + 1, i + 97):
                if lo[j] <= stp:
                    R = -1.0; break
                if ex == "E1" and p[j] < ps[j] and p[j - 1] >= ps[j - 1]:
                    R = (c[j] - ent) / risk; break
                if ex == "E2" and p[j] <= 0 < p[j - 1]:
                    R = (c[j] - ent) / risk; break
            if R is None:
                R = (c[i + 96] - ent) / risk
            row[ex] = R - cost
        rows.append(row)
    return rows


def main():
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    with ThreadPoolExecutor(5) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    rng = np.random.default_rng(0)

    def ci(sel, key):
        wk = {}
        for r in rows:
            if sel(r):
                wk.setdefault(r["t"], []).append(r[key])
        if len(wk) < 10:
            return "—"
        ks = list(wk)
        s = np.array([sum(wk[k]) for k in ks]); cnt = np.array([len(wk[k]) for k in ks])
        bs = [s[ix].sum() / cnt[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(300))]
        allv = [v for k in ks for v in wk[k]]
        return f"{np.mean(allv):+.3f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] رابحة {np.mean([v>0 for v in allv])*100:.0f}%"

    ok = lambda x, k: x is not None and x[k]
    groups = [("الأساس: دخول كل ساعة عشوائياً", lambda r: r["base"]),
              ("لمس صفر 15د (PMO فقط)", lambda r: r["touch"]),
              ("لمس + PMO الساعة فوق الصفر", lambda r: r["touch"] and ok(r["h1"], "pos")),
              ("لمس + الساعة والأربع ساعات فوق الصفر", lambda r: r["touch"] and ok(r["h1"], "pos") and ok(r["h4"], "pos")),
              ("لمس + الساعة فوق البرتقالي وصاعد", lambda r: r["touch"] and ok(r["h1"], "above") and ok(r["h1"], "up")),
              ("لمس + اليومي فوق الصفر", lambda r: r["touch"] and ok(r["d1"], "pos")),
              ("لمس + اليومي تحت الصفر وصاعد", lambda r: r["touch"] and r["d1"] is not None and not r["d1"]["pos"] and r["d1"]["up"]),
              ("إنذار قبل اللمس (PMO فقط)", lambda r: r["alert"]),
              ("إنذار + الساعة فوق الصفر", lambda r: r["alert"] and ok(r["h1"], "pos")),
              ("إنذار + الساعة والأربع ساعات فوق الصفر", lambda r: r["alert"] and ok(r["h1"], "pos") and ok(r["h4"], "pos"))]
    print(f"{len(tick)} عملة · 70 يوماً · خروج E1: PMO الأزرق يقطع البرتقالي هبوطاً | E2: PMO يعود تحت الصفر | وقف 1.5×ATR | أقصى 24 ساعة | تكلفة 0.1%")
    for ex in ("E1", "E2"):
        print(f"\n== خروج {ex}: متوسط R [حدّ الثقة 95%]")
        for nm, sel in groups:
            n = sum(1 for r in rows if sel(r))
            print(f"{nm:42s} {n:6d} | {ci(sel, ex)}")


main()
