"""Mirror test for SHORTS with the same PMO(35,20,10) logic. Entry: 15m blue crosses zero DOWN. Larger-frame filters mirrored (below zero). Exits: E1 = blue crosses above orange | E2 = blue back above zero.
Stop = 1.5xATR above entry (clamp 0.5-6%), max 24h, costs 0.1% (no funding). Baseline: short every hour. 180 days, crypto futures. R = pnl / stop distance. CI = week-block bootstrap."""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np
import scanner as S
import pmo_only as P

P.DAYS = {"Min15": 180, "Min60": 130, "Hour4": 300, "Day1": 300}


def fs(d, per, tc, price):
    return P.fstate(d, per, tc, price)


def work(sym):
    now = int(time.time())
    k15, k1, k4, kd = P.kl(sym, "Min15", now), P.kl(sym, "Min60", now), P.kl(sym, "Hour4", now), P.kl(sym, "Day1", now)
    if not all((k15, k1, k4, kd)):
        return []
    t, hi, lo, c = k15["time"], k15["high"], k15["low"], k15["close"]
    n = len(c)
    p, ps = S.pmo(c)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    rows = []
    last = last_a = -99
    for i in range(400, n - 97):
        touch = p[i - 1] >= 0 > p[i] and i - last >= 4
        sp = p[i - 1] - p[i]
        alert = p[i] > 0 and p[i] < ps[i] and sp > 0 and p[i] <= 3 * sp and i - last_a >= 8 and not touch      # about to touch zero from above
        if touch: last = i
        if alert: last_a = i
        base = i % 4 == 0
        if not (touch or alert or base):
            continue
        ent = c[i]
        row = {"t": t[i], "touch": touch, "alert": alert, "base": base}
        if touch or alert:
            tc = t[i] + 900
            row["h1"], row["h4"], row["d1"] = fs(k1, 3600, tc, ent), fs(k4, 14400, tc, ent), fs(kd, 86400, tc, ent)
        stp = min(max(ent + 1.5 * atr[i], ent * 1.005), ent * 1.06)
        risk = stp - ent
        for ex in ("E1", "E2"):
            out = None
            for j in range(i + 1, i + 97):
                if hi[j] >= stp:
                    out = stp; break
                if ex == "E1" and p[j] > ps[j] and p[j - 1] <= ps[j - 1]:
                    out = c[j]; break
                if ex == "E2" and p[j] >= 0 > p[j - 1]:
                    out = c[j]; break
            if out is None:
                out = c[i + 96]
            pct = (ent / out - 1) * 100 if False else ((ent - out) / ent) * 100 - 0.1
            row[ex] = pct
            row[ex + "R"] = pct / (risk / ent * 100)
        rows.append(row)
    return rows


def main():
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    ETF = P.ETF | {"USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD"}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    with ThreadPoolExecutor(4) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    rng = np.random.default_rng(0)

    def stats(g, xn):
        xr = np.array([r[xn + "R"] for r in g]); xp = np.array([r[xn] for r in g])
        wk = {}
        for r in g:
            wk.setdefault(r["t"] // (7 * 86400), []).append(r[xn + "R"])
        ks = list(wk)
        sm = np.array([sum(wk[x]) for x in ks]); cn = np.array([len(wk[x]) for x in ks])
        bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(400))]
        return f"n={len(g):6d} متوسط {xp.mean():+.3f}% / {xr.mean():+.3f}R | ثقة [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] | رابحة {np.mean(xp>0)*100:.0f}% | وسيط {np.median(xp):+.2f}%"

    neg = lambda x, k: x is not None and not x["pos"] if k == "pos" else x is not None and x[k]
    below = lambda x: x is not None and not x["pos"]
    groups = [("شورت عشوائي كل ساعة (مرجع)", lambda r: r["base"]),
              ("لمس صفر هبوطاً 15د (PMO فقط)", lambda r: r["touch"]),
              ("+ الساعة تحت الصفر", lambda r: r["touch"] and below(r["h1"])),
              ("+ الساعة والأربع ساعات تحت الصفر", lambda r: r["touch"] and below(r["h1"]) and below(r["h4"])),
              ("+ اليومي تحت الصفر", lambda r: r["touch"] and below(r["d1"])),
              ("+ اليومي والأربع ساعات والساعة تحت الصفر", lambda r: r["touch"] and below(r["d1"]) and below(r["h4"]) and below(r["h1"])),
              ("+ اليومي فوق الصفر (عكس الاتجاه)", lambda r: r["touch"] and r["d1"] is not None and r["d1"]["pos"]),
              ("إنذار قبل اللمس الهبوطي (PMO فقط)", lambda r: r["alert"]),
              ("إنذار + اليومي تحت الصفر", lambda r: r["alert"] and below(r["d1"]))]
    print(f"{len(tick)} عملة · 180 يوماً · شورت · وقف 1.5×ATR فوق الدخول · حد أقصى 24س · تكلفة 0.1% بلا تمويل · E1: الأزرق يقطع البرتقالي صعوداً | E2: PMO يعود فوق الصفر")
    for xn in ("E1", "E2"):
        print(f"\\n== خروج {xn}")
        for nm, sel in groups:
            g = [r for r in rows if sel(r)]
            if len(g) >= 30:
                print(f"{nm:44s} {stats(g, xn)}")
    best = [r for r in rows if r["touch"] and below(r["d1"])]
    print("\\n== تفصيل: لمس هبوطي + اليومي تحت الصفر (E1) حسب الشهر")
    bym = {}
    for r in best:
        bym.setdefault(datetime.fromtimestamp(r["t"], timezone.utc).strftime("%Y-%m"), []).append(r)
    for k in sorted(bym):
        if len(bym[k]) >= 30:
            print(f"{k}: {stats(bym[k], 'E1')}")
    xp = np.sort(np.array([r["E1"] for r in best]))
    print(f"بعد حذف أعلى 1%: {xp[: int(len(xp)*0.99)].mean():+.3f}% | أعلى 5%: {xp[: int(len(xp)*0.95)].mean():+.3f}% | أفضل 5 صفقات: {', '.join(f'{v:+.1f}%' for v in xp[-5:])}")
    half = np.median([r["t"] for r in best])
    print("النصف الأول :", stats([r for r in best if r["t"] < half], "E1"))
    print("النصف الثاني:", stats([r for r in best if r["t"] >= half], "E1"))
    b2 = [r for r in rows if r["base"]]
    print("\\n== المرجع: شورت عشوائي حسب الشهر (يبيّن اتجاه السوق نفسه)")
    bym = {}
    for r in b2:
        bym.setdefault(datetime.fromtimestamp(r["t"], timezone.utc).strftime("%Y-%m"), []).append(r)
    for k in sorted(bym):
        print(f"{k}: {stats(bym[k], 'E1')}")


if __name__ == "__main__":
    main()
