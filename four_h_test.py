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

P.DAYS = {"Min15": 180, "Min60": 130, "Hour4": 420, "Day1": 300}
A1, A2, AS = 2 / 35, 2 / 20, 2 / 11


def work(sym):
    now = int(time.time())
    k15, k4 = P.kl(sym, "Min15", now), P.kl(sym, "Hour4", now)
    if not (k15 and k4):
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
        while k + 1 < m4 and t4[k + 1] + 14400 <= tc:
            k += 1
        if t4[k] + 14400 > tc or k < 150:
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
        event = pf[i - 1] <= 0 < pf[i] and i - last_ev >= 8
        if event:
            last_ev = i
            ev.append(i)
        touch = p[i - 1] <= 0 < p[i] and i - last_touch >= 4
        if touch:
            last_touch = i
        if not (touch or event or i % 16 == 0):
            continue
        k = kk[i]
        ago = (i - last_ev) * 0.25 if last_ev > 0 else 999                  # hours since last 4H touch (at 15m resolution)
        slope = s2[k] - s2[k - 1]
        pos = pf[i] > 0
        alert = (not pos) and pf[i] > sf[i] and slope > 0 and -pf[i] <= 3 * slope
        sig = {"S0": touch, "S1": touch and pos and ago <= 8, "S2": touch and pos and ago <= 24, "S3": event, "S4": touch and alert,
               "S5": touch and pos and ago > 24, "S6": touch and (not pos) and not alert, "BASE": i % 16 == 0}
        if not any(sig.values()):
            continue
        ent = c[i]
        row = {"t": t[i], "sym": sym, "sig": sig}
        s_a = max(min(ent - 1.5 * atr[i], ent * 0.995), ent * 0.94)
        s_b = max(min(ent - 1.5 * atr4[k], ent * 0.995), ent * 0.90)
        for xn, stp0, horizon in (("X1", s_a, 96), ("X2", s_b, 288)):
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
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in P.ETF]
    with ThreadPoolExecutor(4) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    rng = np.random.default_rng(0)
    names = {"S0": "كل لمسات 15د (مرجع)", "S1": "لمس 15د بعد لمس 4س خلال 8 ساعات (4س فوق الصفر)", "S2": "لمس 15د بعد لمس 4س خلال 24 ساعة", "S3": "لمس 4س نفسه (دخول لحظة اللمس)",
             "S4": "لمس 15د و4س على وشك اللمس", "S5": "لمس 15د و4س فوق الصفر منذ أكثر من 24س", "S6": "لمس 15د و4س تحت الصفر (غير قريب)"}
    print(f"{len(tick)} عملة · 180 يوماً · خروج X1: قطع PMO 15د للبرتقالي | وقف 1.5×ATR 15د | 24س ·· X2: قطع PMO 4س للبرتقالي | وقف 1.5×ATR 4س | 72س · تكلفة 0.1%")
    for xn, label in (("X1", "X1 (خروج سريع)"), ("X2", "X2 (خروج على 4 ساعات)")):
        print(f"\\n== {label}")
        print(f"{'':52s} {'صفقات':>6s} {'متوسط %':>8s} {'متوسط R':>8s} {'حدّ الثقة R 95%':>17s} {'رابحة':>6s} {'وسيط %':>7s}")
        for sk, nm in names.items():
            g = [r for r in rows if r["sig"][sk]]
            if len(g) < 30:
                print(f"{nm:52s} {len(g):6d}"); continue
            xs = np.array([r[xn] for r in g]); xr = np.array([r[xn + "R"] for r in g])
            slots = {}
            for r in g:
                slots.setdefault(r["t"], []).append(r[xn + "R"])
            ks = list(slots)
            sm = np.array([sum(slots[x]) for x in ks]); cn = np.array([len(slots[x]) for x in ks])
            bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(300))]
            print(f"{nm:52s} {len(g):6d} {xs.mean():+8.3f} {xr.mean():+8.3f} [{np.percentile(bs,2.5):+6.2f},{np.percentile(bs,97.5):+6.2f}] {np.mean(xs>0)*100:5.0f}% {np.median(xs):+7.2f}")


if __name__ == "__main__":
    main()
