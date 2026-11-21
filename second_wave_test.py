"""'Second wave' rule test. Signal = 15m PMO zero-touch up. Pre-conditions measured at the signal bar (past only):
 hot = ATR(14) >= 1.2 x its 10-day median | week = coin up over 7 days | d1 = daily PMO above zero | h4 = 4H PMO above zero | pull = pullback >= 6% from the 24h high.
Variants stack the conditions. Trades: stop 1.5xATR; exit E1 = blue crosses below orange (24h) or TR = trailing stop (highest high - 2xATR, 24h); costs 0.1%. 180 days.
Reports mean R, week-block CI, win rate, mean %, median %, outlier trimming, time halves, and how often +10% was reached within 24h."""
import time
from concurrent.futures import ThreadPoolExecutor
import statistics as st
import numpy as np
import scanner as S
import scanner_pmo as P

P.DAYS.update({"Min15": 180, "Min60": 130, "Hour4": 300, "Day1": 300})
ETF = P.ETF | {"USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}


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
        d1 = P.fstate(kd, 86400, tc, ent); h4 = P.fstate(k4, 14400, tc, ent)
        feats = {"hot": atr[i] / (st.median(atr[max(0, i - 960):i]) or 1e-12) >= 1.2, "week": ent / c[i - 672] - 1 > 0, "d1": bool(d1 and d1["pos"]), "h4": bool(h4 and h4["pos"]),
                 "pull": (ent / max(hi[i - 95:i + 1]) - 1) * 100 <= -6}
        stp0 = min(max(ent - 1.5 * atr[i], ent * 0.94), ent * 0.995)
        risk = ent - stp0
        row = {"t": t[i], "sym": sym, "f": feats, "mfe": (max(hi[i + 1:i + 97]) / ent - 1) * 100}
        for xn in ("E1", "TR"):
            stp, hh, out = stp0, ent, None
            for j in range(i + 1, i + 97):
                if xn == "TR":
                    stp = max(stp0, hh - 2 * atr[i])
                if lo[j] <= stp:
                    out = stp; break
                hh = max(hh, hi[j])
                if xn == "E1" and p[j] < ps[j] and p[j - 1] >= ps[j - 1]:
                    out = c[j]; break
            if out is None:
                out = c[i + 96]
            pct = (out / ent - 1) * 100 - 0.1
            row[xn] = pct
            row[xn + "R"] = pct / (risk / ent * 100)
        rows.append(row)
    return rows


def main():
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    with ThreadPoolExecutor(4) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    rng = np.random.default_rng(0)
    V = {"W0": ("كل لمسات الصفر", lambda f: True), "W1": ("اليومي فوق الصفر (شرطنا الحالي)", lambda f: f["d1"]),
         "W2": ("+ عملة متقلبة (ATR ≥1.2×المعتاد)", lambda f: f["d1"] and f["hot"]), "W3": ("+ صاعدة خلال أسبوع", lambda f: f["d1"] and f["hot"] and f["week"]),
         "W4": ("+ تراجع ≥6% عن قمة 24س (الموجة الثانية)", lambda f: f["d1"] and f["hot"] and f["week"] and f["pull"]),
         "W5": ("+ الأربع ساعات فوق الصفر", lambda f: f["d1"] and f["hot"] and f["week"] and f["pull"] and f["h4"]),
         "W6": ("تراجع ≥6% + اليومي فوق الصفر (بلا غيره)", lambda f: f["d1"] and f["pull"]),
         "W7": ("صاعدة أسبوعاً + تراجع ≥6% (بلا اليومي)", lambda f: f["week"] and f["pull"])}

    def line(g, xn):
        R = np.array([r[xn + "R"] for r in g]); P_ = np.array([r[xn] for r in g])
        wk = {}
        for r in g:
            wk.setdefault(r["t"] // (7 * 86400), []).append(r[xn + "R"])
        ks = list(wk)
        sm = np.array([sum(wk[x]) for x in ks]); cn = np.array([len(wk[x]) for x in ks])
        bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(400))]
        s = np.sort(P_)
        return (f"n={len(g):5d} R {R.mean():+.3f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] | % {P_.mean():+.2f} وسيط {np.median(P_):+.2f} | رابحة {np.mean(P_>0)*100:2.0f}% | "
                f"بلا أعلى 1%: {s[:int(len(s)*0.99)].mean():+.2f}% | +10% خلال 24س: {np.mean([r['mfe']>=10 for r in g])*100:.0f}%")

    print(f"{len(tick)} عملة · 180 يوماً · {len(rows)} لمسة صفر 15د · وقف 1.5×ATR · تكلفة 0.1%")
    half = np.median([r["t"] for r in rows])
    for xn, nm in (("E1", "خروج E1: قطع PMO للبرتقالي"), ("TR", "خروج TR: وقف متحرك 2×ATR")):
        print(f"\\n== {nm}")
        for k, (lab, fn) in V.items():
            g = [r for r in rows if fn(r["f"])]
            if len(g) >= 30:
                print(f"{lab:44s} {line(g, xn)}")
    print("\\n== W4 حسب نصفي الفترة (E1 و TR)")
    g = [r for r in rows if V["W4"][1](r["f"])]
    for xn in ("E1", "TR"):
        for nm, h in (("النصف الأول", [r for r in g if r["t"] < half]), ("النصف الثاني", [r for r in g if r["t"] >= half])):
            if len(h) >= 20:
                print(f"{xn} {nm:14s} {line(h, xn)}")
    bc = {}
    for r in g:
        bc[r["sym"]] = bc.get(r["sym"], 0) + r["TR"]
    tot = sum(bc.values())
    print("W4 تركّز العملات (TR):", ", ".join(f"{k[:-5]} {v/tot*100:.0f}%" for k, v in sorted(bc.items(), key=lambda kv: -kv[1])[:6]) if tot > 0 else "المجموع سالب")


if __name__ == "__main__":
    main()
