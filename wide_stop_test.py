"""Wider stops on the 'second wave' group (and context groups). Signal = 15m PMO zero-touch up with conditions as in second_wave_test. Stops: 1.5xATR (reference), 2.5xATR, 3.5xATR,
below the 24h low, fixed 4%, fixed 6%, fixed 10%. Exits: E1 = blue crosses below orange, T24 = hold 24h (stop only). Costs 0.1%. Output in % of price (the stop size changes R)."""
import time
from concurrent.futures import ThreadPoolExecutor
import statistics as st
import numpy as np
import scanner as S
import scanner_pmo as P

P.DAYS.update({"Min15": 180, "Min60": 130, "Hour4": 300, "Day1": 300})
ETF = P.ETF | {"USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}
STOPS = ("atr15", "atr25", "atr35", "low24", "pct4", "pct6", "pct10")


def work(sym):
    now = int(time.time())
    k15, k4, kd = P.kl(sym, "Min15", now), P.kl(sym, "Hour4", now), P.kl(sym, "Day1", now)
    if not all((k15, k4, kd)):
        return []
    t, hi, lo, c = k15["time"], k15["high"], k15["low"], k15["close"]
    n = len(c)
    p, ps = S.pmo(c)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    rows, last = [], -99
    for i in range(700, n - 100):
        if not (p[i - 1] <= 0 < p[i]) or i - last < 4:
            continue
        last = i
        ent, tc = c[i], t[i] + 900
        d1 = P.fstate(kd, 86400, tc, ent)
        f = {"d1": bool(d1 and d1["pos"]), "hot": atr[i] / (st.median(atr[max(0, i - 960):i]) or 1e-12) >= 1.2, "week": ent / c[i - 672] - 1 > 0,
             "pull": (ent / max(hi[i - 95:i + 1]) - 1) * 100 <= -6}
        if not (f["d1"] or f["pull"]):
            continue
        stops = {"atr15": ent - 1.5 * atr[i], "atr25": ent - 2.5 * atr[i], "atr35": ent - 3.5 * atr[i], "low24": min(lo[i - 95:i + 1]) * 0.998,
                 "pct4": ent * 0.96, "pct6": ent * 0.94, "pct10": ent * 0.90}
        row = {"t": t[i], "sym": sym, "f": f}
        for sn, stp in stops.items():
            stp = max(min(stp, ent * 0.995), ent * 0.85)
            for xn in ("E1", "T24"):
                out, hit = None, False
                for j in range(i + 1, i + 97):
                    if lo[j] <= stp:
                        out, hit = stp, True; break
                    if xn == "E1" and p[j] < ps[j] and p[j - 1] >= ps[j - 1]:
                        out = c[j]; break
                if out is None:
                    out = c[i + 96]
                row[f"{sn}_{xn}"] = ((out / ent - 1) * 100 - 0.1, hit, (1 - stp / ent) * 100)
        rows.append(row)
    return rows


def main():
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    with ThreadPoolExecutor(4) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    rng = np.random.default_rng(0)
    G = {"W1: اليومي فوق الصفر": lambda f: f["d1"], "W3: + متقلبة + صاعدة أسبوعاً": lambda f: f["d1"] and f["hot"] and f["week"],
         "W4: + تراجع ≥6% (الموجة الثانية)": lambda f: f["d1"] and f["hot"] and f["week"] and f["pull"]}
    half = np.median([r["t"] for r in rows])
    print(f"{len(tick)} عملة · 180 يوماً · تكلفة 0.1% · النتائج بالنسبة المئوية من السعر")
    for gname, fn in G.items():
        g = [r for r in rows if fn(r["f"])]
        print(f"\\n==== {gname}: {len(g)} صفقة")
        for xn, xl in (("E1", "خروج E1 (قطع PMO)"), ("T24", "خروج T24 (24 ساعة)")):
            print(f"-- {xl}")
            print(f"   {'الوقف':8s} {'مسافته':>7s} {'متوسط %':>9s} {'ثقة 95% (أسابيع)':>20s} {'وسيط %':>8s} {'رابحة':>6s} {'ضُرب الوقف':>10s} {'أسوأ %':>8s} | {'نصف1 %':>8s} {'نصف2 %':>8s}")
            for sn in STOPS:
                xs = np.array([r[f"{sn}_{xn}"][0] for r in g]); hits = np.mean([r[f"{sn}_{xn}"][1] for r in g]); dist = np.mean([r[f"{sn}_{xn}"][2] for r in g])
                wk = {}
                for r in g:
                    wk.setdefault(r["t"] // (7 * 86400), []).append(r[f"{sn}_{xn}"][0])
                ks = list(wk)
                sm = np.array([sum(wk[x]) for x in ks]); cn = np.array([len(wk[x]) for x in ks])
                bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(300))]
                h1 = np.mean([r[f"{sn}_{xn}"][0] for r in g if r["t"] < half]); h2 = np.mean([r[f"{sn}_{xn}"][0] for r in g if r["t"] >= half])
                print(f"   {sn:8s} {dist:6.1f}% {xs.mean():+9.3f} [{np.percentile(bs,2.5):+7.2f},{np.percentile(bs,97.5):+7.2f}] {np.median(xs):+8.2f} {np.mean(xs>0)*100:5.0f}% {hits*100:9.0f}% {xs.min():+8.1f} | {h1:+8.2f} {h2:+8.2f}")


if __name__ == "__main__":
    main()
