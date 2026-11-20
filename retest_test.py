"""Entry-at-retest test. Signal = 15m PMO blue crosses zero UP (optionally daily PMO above zero). Instead of buying the signal bar close, wait for a retest:
 imm   = buy signal close (reference)
 lim05 = limit at close - 0.5xATR (fill if low reaches it within 3h)      lim10 = limit at close - 1.0xATR (3h)
 tema  = limit at TEMA50 if it is below the close (fill within 3h)        pmo2  = after the touch, blue crosses above orange AGAIN while still above zero (within 6h): buy that close
Stop = 1.5xATR below ENTRY (clamp 0.5-6%). Exit = blue crosses below orange, or stop, or 24h after entry. Costs 0.1%. 180 days. Limit orders fill at the limit price (optimistic).
Per-signal view: fill rate, mean R of filled trades, the reference 'imm' R on the SAME signals (to catch adverse selection), and expected % per signal (non-fills = 0)."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scanner as S
import pmo_only as P

P.DAYS = {"Min15": 180, "Min60": 130, "Hour4": 300, "Day1": 300}
ETF = P.ETF | {"USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}


def run_trade(ent, j0, p, ps, lo, c, atr_i, n):
    stp = min(max(ent - 1.5 * atr_i, ent * 0.94), ent * 0.995)
    risk = ent - stp
    out = None
    for j in range(j0, min(j0 + 96, n)):
        if lo[j] <= stp:
            out = stp; break
        if j > j0 and p[j] < ps[j] and p[j - 1] >= ps[j - 1]:
            out = c[j]; break
    if out is None:
        out = c[min(j0 + 96, n - 1)]
    pct = (out / ent - 1) * 100 - 0.1
    return pct, pct / (risk / ent * 100)


def work(sym):
    now = int(time.time())
    k15, kd = P.kl(sym, "Min15", now), P.kl(sym, "Day1", now)
    if not (k15 and kd):
        return []
    t, hi, lo, c = k15["time"], k15["high"], k15["low"], k15["close"]
    n = len(c)
    p, ps = S.pmo(c)
    te = S.tema(c, 50)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    rows, last = [], -99
    for i in range(400, n - 150):
        if not (p[i - 1] <= 0 < p[i]) or i - last < 4:
            continue
        last = i
        d1 = P.fstate(kd, 86400, t[i] + 900, c[i])
        row = {"t": t[i], "sym": sym, "d1pos": bool(d1 and d1["pos"])}
        res = {"imm": (c[i], i + 1)}
        for name, off in (("lim05", 0.5), ("lim10", 1.0)):
            L = c[i] - off * atr[i]
            for j in range(i + 1, i + 13):
                if lo[j] <= L:
                    res[name] = (L, j); break
        if te[i] < c[i]:
            for j in range(i + 1, i + 13):
                if lo[j] <= te[i]:
                    res["tema"] = (te[i], j); break
        for j in range(i + 1, i + 25):
            if p[j] <= 0:
                break
            if p[j] > ps[j] and p[j - 1] <= ps[j - 1] and any(p[k] < ps[k] for k in range(i + 1, j)):
                res["pmo2"] = (c[j], j + 1); break
        for name, (ent, j0) in res.items():
            pct, R = run_trade(ent, j0 if name in ("imm", "pmo2") else j0, p, ps, lo, c, atr[i], n)
            row[name] = (pct, R)
        rows.append(row)
    return rows


def main():
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    with ThreadPoolExecutor(4) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    rng = np.random.default_rng(0)
    names = {"imm": "دخول فوري (مرجع)", "lim05": "أمر معلّق -0.5×ATR", "lim10": "أمر معلّق -1.0×ATR", "tema": "اختبار TEMA50", "pmo2": "تقاطع PMO صاعد ثانٍ فوق الصفر"}

    def block(sel, title):
        g = [r for r in rows if sel(r)]
        print(f"\\n== {title}: {len(g)} إشارة")
        print(f"{'':28s} {'نُفّذ':>6s} {'نسبة':>5s} | {'متوسط R للمنفّذ':>16s} {'ثقة 95% (أسابيع)':>18s} {'رابحة':>6s} {'وسيط %':>7s} | {'R فوري لنفس الإشارات':>20s} | {'% لكل إشارة':>11s}")
        for k, nm in names.items():
            f = [r for r in g if k in r]
            if len(f) < 30:
                print(f"{nm:28s} {len(f):6d}"); continue
            R = np.array([r[k][1] for r in f]); P_ = np.array([r[k][0] for r in f])
            wk = {}
            for r in f:
                wk.setdefault(r["t"] // (7 * 86400), []).append(r[k][1])
            ks = list(wk)
            sm = np.array([sum(wk[x]) for x in ks]); cn = np.array([len(wk[x]) for x in ks])
            bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(300))]
            imm = np.mean([r["imm"][1] for r in f])
            per_sig = P_.sum() / len(g)
            print(f"{nm:28s} {len(f):6d} {len(f)/len(g)*100:4.0f}% | {R.mean():+16.3f} [{np.percentile(bs,2.5):+6.2f},{np.percentile(bs,97.5):+6.2f}] {np.mean(P_>0)*100:5.0f}% {np.median(P_):+7.2f} | {imm:+20.3f} | {per_sig:+11.3f}")
    block(lambda r: True, "كل لمسات الصفر على 15د")
    block(lambda r: r["d1pos"], "لمس الصفر + اليومي فوق الصفر (شرطنا)")
    block(lambda r: not r["d1pos"], "لمس الصفر + اليومي تحت الصفر")


if __name__ == "__main__":
    main()
