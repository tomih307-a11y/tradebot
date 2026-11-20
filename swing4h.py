"""4H swing test with an unconditional baseline: forward net returns after (a) PMO bullish cross far below zero ("bottom") with MACD hist turning up,
(b) PMO zero-touch up, (c) EVERY 4H bar (baseline). 730 days, crypto futures. Hold 4d/7d/14d, optional -8% stop. Bootstrap by week."""
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import scanner as S

SEC4 = 14400


def kl(sym, now, days=730):
    out = {k: [] for k in ("time", "high", "low", "close")}
    a = now - days * 86400
    while a < now:
        b = min(a + SEC4 * 1000, now)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Hour4&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            return None
        for k in out:
            out[k] += d[k]
        a = b
    seen, idx = set(), []
    for i, tt in enumerate(out["time"]):
        if tt not in seen and tt + SEC4 <= now:
            seen.add(tt); idx.append(i)
    return {k: [out[k][i] for i in idx] for k in out}


def work(sym):
    now = int(time.time())
    d = kl(sym, now)
    if not d or len(d["close"]) < 400:
        return []
    t, c, hi, lo = d["time"], d["close"], d["high"], d["low"]
    p, ps = S.pmo(c)
    m = [a - b for a, b in zip(S.ema(c, 12), S.ema(c, 26))]
    g = S.ema(m, 9)
    n = len(c)
    rows = []
    for i in range(250, n - 84):
        ent = c[i]
        row = {"sym": sym, "t": t[i], "bottom": (p[i - 1] <= ps[i - 1] and p[i] > ps[i] and p[i] < -1.0 and (m[i] - g[i]) > (m[i - 1] - g[i - 1]) and m[i] - g[i] > -abs(m[i]) * 0.5),
               "touch": p[i - 1] <= 0 < p[i], "pmo": p[i]}
        row["bottom"] = bool(row["bottom"])
        for nm, h in (("r4d", 24), ("r7d", 42), ("r14d", 84)):
            row[nm] = (c[i + h] / ent - 1) * 100 - 0.1
        for nm, h in (("s7d", 42), ("s14d", 84)):                                   # -8% stop, then exit at horizon close
            r = None
            for j in range(i + 1, i + h + 1):
                if lo[j] <= ent * 0.92:
                    r = -8.0 - 0.1; break
            row[nm] = r if r is not None else (c[i + h] / ent - 1) * 100 - 0.1
        rows.append(row)
    return rows


def main():
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX"}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    with ThreadPoolExecutor(5) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    rng = np.random.default_rng(0)
    print(f"{len(tick)} عملة · {len(rows)} شمعة 4س (أساس) · خصم 0.1% تكلفة")

    def ci(sel, key):
        xs = [(r["t"] // (7 * 86400), r[key]) for r in rows if sel(r)]
        wk = {}
        for w, v in xs:
            wk.setdefault(w, []).append(v)
        ks = list(wk)
        s = np.array([sum(wk[k]) for k in ks]); cnt = np.array([len(wk[k]) for k in ks])
        bs = []
        for _ in range(300):
            ix = rng.integers(0, len(ks), len(ks)); bs.append(s[ix].sum() / cnt[ix].sum())
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        return f"{np.mean([v for _, v in xs]):+.2f}% [{lo_:+.2f},{hi_:+.2f}]"

    half = np.median([r["t"] for r in rows])
    groups = (("كل شموع 4س (الأساس)", lambda r: True), ("قاع: تقاطع PMO تحت -1 + مكدي يرتفع", lambda r: r["bottom"]),
              ("لمس PMO للصفر", lambda r: r["touch"]), ("قاع — النصف الأول", lambda r: r["bottom"] and r["t"] < half), ("قاع — النصف الثاني", lambda r: r["bottom"] and r["t"] >= half))
    print(f"{'':40s} {'n':>7s} | " + " | ".join(f"{k:>22s}" for k in ("r4d", "r7d", "r14d", "s7d", "s14d")))
    for nm, sel in groups:
        n = sum(1 for r in rows if sel(r))
        print(f"{nm:40s} {n:7d} | " + " | ".join(f"{ci(sel, k):>22s}" for k in ("r4d", "r7d", "r14d", "s7d", "s14d")))


main()
