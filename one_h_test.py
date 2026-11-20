"""Re-ordered conditions: condition 1 = PMO zero-touch on the 4H frame (forming 4H candle updated at 15m resolution), entry from the 15m frame.
Signals: S0 all 15m touches | S1 15m touch with a 4H touch in the last 8h and 4H PMO still above zero | S2 same within 24h | S3 the 4H touch itself (entry at that 15m bar)
S4 15m touch while 4H PMO is about to touch zero (blue>orange, rising, <=3 candles) | S5 15m touch while 4H above zero for more than 24h | S6 15m touch while 4H below zero (not about to touch)
Exits: X1 = 15m blue crosses below orange | stop 1.5xATR(15m) | 24h.  X2 = 4H blue (forming) crosses below 4H orange | stop 1.5xATR(4H) clamp 0.5-10% | 72h.  Costs 0.1%.
180 days of 15m."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scanner as S
import pmo_only as P

P.DAYS = {"Min15": 180, "Min60": 400, "Hour4": 420, "Day1": 300}
A1, A2, AS = 2 / 35, 2 / 20, 2 / 11


def work(sym):
    now = int(time.time())
    k15, k4 = P.kl(sym, "Min15", now), P.kl(sym, "Min60", now)
    kd = P.kl(sym, "Day1", now)
    kh4 = P.kl(sym, "Hour4", now)
    if not (k15 and k4 and kd and kh4):
        return []
    t, hi, lo, c = k15["time"], k15["high"], k15["low"], k15["close"]
    t4, h4, l4, c4 = k4["time"], k4["high"], k4["low"], k4["close"]
    n, m4 = len(c), len(c4)
    roc4 = [0.0] + [(c4[i] / c4[i - 1] - 1) * 100 for i in range(1, m4)]
    s1 = S.smooth(roc4, 35)
    s2 = S.smooth([10 * v for v in s1], 20)
    sg = S.ema(s2, 10)
    tr4 = [h4[k] - l4[k] if k == 0 else max(h4[k] - l4[k], abs(h4[k] - c4[k - 1]), abs(l4[k] - c4[k - 1])) for k in range(m4)]
    atr4 = S.ema(tr4, 14)
    p, ps = S.pmo(c)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    pf, sf, kk = [None] * n, [None] * n, [None] * n
    k = 0
    for i in range(n):
        tc = t[i] + 900
        while k + 1 < m4 and t4[k + 1] + 3600 <= tc:
            k += 1
        if t4[k] + 3600 > tc or k < 150:
            continue
        roc = (c[i] / c4[k] - 1) * 100
        a = A1 * roc + (1 - A1) * s1[k]
        b = A2 * 10 * a + (1 - A2) * s2[k]
        pf[i], sf[i], kk[i] = b, AS * b + (1 - AS) * sg[k], k
    ev, last_ev, last_touch = [], -999, -99
    rows = []
    for i in range(1, n - 289):
        if pf[i] is None or pf[i - 1] is None:
            continue
        event = pf[i - 1] <= 0 < pf[i] and i - last_ev >= 4
        if event:
            last_ev = i
            ev.append(i)
        touch = p[i - 1] <= 0 < p[i] and i - last_touch >= 4
        if touch:
            last_touch = i
        if not (touch or event or i % 4 == 0):
            continue
        k = kk[i]
        ago = (i - last_ev) * 0.25 if last_ev > 0 else 999                  # hours since last 4H touch (at 15m resolution)
        slope = s2[k] - s2[k - 1]
        pos = pf[i] > 0
        alert = (not pos) and pf[i] > sf[i] and slope > 0 and -pf[i] <= 3 * slope
        sig = {"S7": False, "S8": False, "S9": False, "S0": touch, "S1": touch and pos and ago <= 4, "S2": touch and pos and ago <= 12, "S3": event, "S4": touch and alert,
               "S5": touch and pos and ago > 12, "S6": touch and (not pos) and not alert, "BASE": i % 4 == 0}
        if not any(sig.values()):
            continue
        ent = c[i]
        row = {"t": t[i], "sym": sym, "sig": sig}
        if event:
            d1 = P.fstate(kd, 86400, t[i] + 900, c[i]); h4s = P.fstate(kh4, 14400, t[i] + 900, c[i])
            sig["S7"] = bool(d1 and d1["pos"]); sig["S8"] = bool(h4s and h4s["pos"]); sig["S9"] = bool(d1 and d1["pos"] and h4s and h4s["pos"])
        s_a = max(min(ent - 1.5 * atr[i], ent * 0.995), ent * 0.94)
        s_b = max(min(ent - 1.5 * atr4[k], ent * 0.995), ent * 0.90)
        for xn, stp0, horizon in (("X1", s_a, 96), ("X2", s_b, 96)):
            risk = ent - stp0
            out = None
            for j in range(i + 1, i + horizon + 1):
                if lo[j] <= stp0:
                    out = stp0; break
                if xn == "X1" and p[j] < ps[j] and p[j - 1] >= ps[j - 1]:
                    out = c[j]; break
                if xn == "X2" and pf[j] is not None and pf[j - 1] is not None and pf[j] < sf[j] and pf[j - 1] >= sf[j - 1]:
                    out = c[j]; break
            if out is None:
                out = c[i + horizon]
            pct = (out / ent - 1) * 100 - 0.1
            row[xn] = pct
            row[xn + "R"] = pct / (risk / ent * 100)
        rows.append(row)
    return rows


def main():
    from datetime import datetime, timezone
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in P.ETF]
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
        bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(500))]
        return f"n={len(g):6d} متوسط {xp.mean():+.3f}% / {xr.mean():+.3f}R | ثقة (كتل أسبوعية) [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] | رابحة {np.mean(xp>0)*100:.0f}% | وسيط {np.median(xp):+.2f}%"

    names = {"BASE": "دخول عشوائي كل ساعة (مرجع)", "S0": "كل لمسات 15د", "S1": "لمس 15د بعد لمس ساعة خلال 4س (الساعة فوق الصفر)", "S2": "لمس 15د بعد لمس ساعة خلال 12س",
             "S3": "لمس الساعة نفسه (دخول لحظة اللمس)", "S4": "لمس 15د والساعة على وشك اللمس", "S5": "لمس 15د والساعة فوق الصفر منذ أكثر من 12س", "S6": "لمس 15د والساعة تحت الصفر (غير قريب)",
             "S7": "لمس الساعة + اليومي فوق الصفر", "S8": "لمس الساعة + 4س فوق الصفر", "S9": "لمس الساعة + اليومي و4س فوق الصفر"}
    print(f"{len(tick)} عملة · 180 يوماً · الشرط الأول = لمس PMO للصفر على فريم الساعة · X1: خروج سريع (PMO 15د) 24س · X2: خروج على الساعة (PMO الساعة، وقف 1.5×ATR ساعة) 24س · تكلفة 0.1%")
    for xn in ("X1", "X2"):
        print(f"\n== {xn}")
        for sk, nm in names.items():
            g = [r for r in rows if r["sig"].get(sk)]
            if len(g) >= 30:
                print(f"{nm:50s} {stats(g, xn)}")
    S3 = [r for r in rows if r["sig"]["S3"]]
    for sk in ("S3", "S7"):
        g = [r for r in rows if r["sig"].get(sk)]
        if len(g) < 50:
            continue
        print(f"\n== تفصيل {names[sk]} (X2)")
        bym = {}
        for r in g:
            bym.setdefault(datetime.fromtimestamp(r["t"], timezone.utc).strftime("%Y-%m"), []).append(r)
        for k in sorted(bym):
            if len(bym[k]) >= 20:
                print(f"{k}: {stats(bym[k], 'X2')}")
        xp = np.sort(np.array([r["X2"] for r in g]))
        print(f"بعد حذف أعلى 1%: {xp[: int(len(xp)*0.99)].mean():+.3f}% | أعلى 5%: {xp[: int(len(xp)*0.95)].mean():+.3f}% | أعلى 5 صفقات: {', '.join(f'{v:+.1f}%' for v in xp[-5:])}")
        bc = {}
        for r in g:
            bc[r["sym"]] = bc.get(r["sym"], 0) + r["X2"]
        tot = sum(bc.values())
        print("أعلى 5 عملات مساهمة:", ", ".join(f"{k[:-5]} {v/tot*100:.0f}%" for k, v in sorted(bc.items(), key=lambda kv: -kv[1])[:5]) if tot > 0 else "المجموع سالب")
        half = np.median([r["t"] for r in g])
        print("النصف الأول :", stats([r for r in g if r["t"] < half], "X2"))
        print("النصف الثاني:", stats([r for r in g if r["t"] >= half], "X2"))


if __name__ == "__main__":
    main()
