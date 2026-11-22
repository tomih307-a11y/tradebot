"""Where does a bounce start? Test candidate 'places' on EVERY 1H candle of ~75 crypto futures coins (Aug 20 - now), using only information known at the candle close.
Outcome from that close: max rise within 48h, and a trade 'target +10% before stop -4%' (48h limit, costs 0.2%, same-candle ambiguity counts as stop).
Places: deep fall, 48h-low sweep + reclaim, long lower wick, capitulation volume, PMO(1H) oversold and turning up, far below TEMA50, OB touch (1H), and combinations.
Signals are de-duplicated (12 candles per coin per place). Times UTC+4."""
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import luxalgo_ob as LX
import scanner as S

now = int(time.time())
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX", "USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}


def fetch(sym, a, b_end):
    out = {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
    while a < b_end:
        b = min(a + 3600 * 1000, b_end)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min60&start={a}&end={b}")["data"]; break
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


def work(sym):
    d = fetch(sym, int(datetime(2026, 8, 20, tzinfo=timezone.utc).timestamp()), now)
    if not d or len(d["time"]) < 800:
        return []
    t, o, hi, lo, c, vo = (d[k] for k in ("time", "open", "high", "low", "close", "vol"))
    n = len(c)
    p, ps = S.pmo(c)
    te = S.tema(c, 50)
    _, snaps = LX.run(o, hi, lo, c)
    rows = []
    last = {}
    for i in range(400, n - 49):
        rng = (hi[i] - lo[i]) or 1e-12
        lowprev = min(lo[i - 48:i - 1])
        flags = {
            "deep": c[i] / max(hi[i - 168:i]) - 1 <= -0.30,
            "fall24": c[i] / max(hi[i - 24:i]) - 1 <= -0.15,
            "sweep": lo[i] < lowprev and c[i] > lowprev,
            "wick": (min(o[i], c[i]) - lo[i]) / rng >= 0.5 and c[i] > (hi[i] + lo[i]) / 2,
            "capit": vo[i] >= 3 * (st.median(vo[i - 168:i]) or 1e-12) and c[i] < o[i],
            "pmo_os": p[i] <= -2 and p[i] > p[i - 1],
            "below_tema": c[i] / te[i] - 1 <= -0.10,
            "ob": any(ob["bias"] == 1 and lo[i] <= ob["hi"] and lo[i] >= ob["lo"] * 0.99 for ob in snaps[i - 1]),
        }
        flags["sweep+wick"] = flags["sweep"] and flags["wick"]
        flags["deep+sweep"] = flags["deep"] and flags["sweep"]
        flags["fall24+wick"] = flags["fall24"] and flags["wick"]
        flags["capit+wick"] = flags["capit"] and flags["wick"]
        flags["deep+capit"] = flags["deep"] and flags["capit"]
        flags["fall24+sweep+wick"] = flags["fall24"] and flags["sweep"] and flags["wick"]
        flags["pmo_os+sweep"] = flags["pmo_os"] and flags["sweep"]
        flags["pmo_os+wick"] = flags["pmo_os"] and flags["wick"]
        flags["fall24+pmo_os"] = flags["fall24"] and flags["pmo_os"]
        flags["all"] = True
        ent = c[i]
        r48 = (max(hi[i + 1:i + 49]) / ent - 1) * 100
        res = None
        for j in range(i + 1, i + 49):
            if lo[j] <= ent * 0.96:
                res = -4.0 - 0.2; break
            if hi[j] >= ent * 1.10:
                res = 10.0 - 0.2; break
        if res is None:
            res = (c[i + 48] / ent - 1) * 100 - 0.2
        for k, v in flags.items():
            if v and (k == "all" or i - last.get(k, -99) >= 12):
                if k != "all":
                    last[k] = i
                rows.append((k, t[i], r48, res))
    return rows


detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
        and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
with ThreadPoolExecutor(4) as ex:
    rows = [r for part in ex.map(work, tick) for r in part]
by = {}
for k, ts, r48, res in rows:
    by.setdefault(k, []).append((ts, r48, res))
base = by["all"]
b20 = np.mean([x[1] >= 20 for x in base]) * 100
b10 = np.mean([x[1] >= 10 for x in base]) * 100
bw = np.mean([x[2] > 9 for x in base]) * 100
print(f"{len(tick)} عملة · شموع الساعة · المعدل العام (كل الشموع): {len(base)} شمعة · وصول +10% خلال 48س {b10:.0f}% · +20%: {b20:.1f}% · هدف +10% قبل وقف -4%: {bw:.1f}% · متوسط الصفقة {np.mean([x[2] for x in base]):+.2f}%")
rng = np.random.default_rng(0)
print(f"\n{'المكان (الشرط عند إغلاق الشمعة)':26s} {'إشارات':>7s} | {'+10% خلال 48س':>13s} {'+20%':>6s} {'+50%':>6s} {'وسيط الصعود':>11s} | {'هدف+10% قبل وقف-4%':>20s} {'ضُرب الوقف':>10s} {'متوسط الصفقة %':>15s} {'ثقة 95% (أسابيع)':>18s} | {'ارتفاع النسبة ×':>14s}")
for k in ("deep", "fall24", "sweep", "wick", "capit", "pmo_os", "below_tema", "ob", "sweep+wick", "deep+sweep", "fall24+wick", "capit+wick", "deep+capit", "fall24+sweep+wick", "pmo_os+sweep", "pmo_os+wick", "fall24+pmo_os"):
    g = by.get(k, [])
    if len(g) < 40:
        print(f"{k:26s} {len(g):7d}"); continue
    r48 = np.array([x[1] for x in g]); res = np.array([x[2] for x in g]); ts = np.array([x[0] for x in g])
    wk = {}
    for tt, rr in zip(ts, res):
        wk.setdefault(tt // (7 * 86400), []).append(rr)
    ks = list(wk)
    sm = np.array([sum(wk[x]) for x in ks]); cn = np.array([len(wk[x]) for x in ks])
    bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(300))]
    print(f"{k:26s} {len(g):7d} | {np.mean(r48>=10)*100:12.0f}% {np.mean(r48>=20)*100:5.1f}% {np.mean(r48>=50)*100:5.1f}% {np.median(r48):+10.1f}% | {np.mean(res>9)*100:19.1f}% {np.mean(res<-3.9)*100:9.0f}% {res.mean():+14.2f}% [{np.percentile(bs,2.5):+5.2f},{np.percentile(bs,97.5):+5.2f}] | {np.mean(r48>=20)*100/b20:13.1f}×")
