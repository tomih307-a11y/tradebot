"""Candidate scanner (long only, MEXC USDT futures, crypto only). It does NOT recommend: it lists recent signals with all the context the user judges by.
Signal types on 15m:  touch = PMO(35,20,10) blue crosses zero UP;  early = PMO blue crosses above orange + MACD(12,26,9) liquidity entry (cross up or hist turns green) within 3 candles.
Usage: python3 scanner2.py [hours_back=6]   — times are UTC+4."""
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import luxalgo_ob as LX
import scanner as S

TZ = 4
SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}
DAYS = {"Min15": 20, "Min60": 60, "Hour4": 200, "Day1": 250}
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "TQQQ", "SQQQ", "TSLL", "NVDL", "MSTU", "CONL", "SPCX", "NVIDIA", "US30", "TESLA", "BITO"}
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)


def kl(sym, itv, now):
    out = {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
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


def frame(d, per, tc, price):
    times, closes = d["time"], d["close"]
    lo_, hi_ = 0, len(times)
    while lo_ < hi_:
        mid = (lo_ + hi_) // 2
        if times[mid] + per <= tc:
            lo_ = mid + 1
        else:
            hi_ = mid
    series = closes[max(0, lo_ - 700):lo_] + [price]                      # closed candles + the forming candle at the price now
    if len(series) < 120:
        return None
    p, s = S.pmo(series)
    m = [a - b for a, b in zip(S.ema(series, 12), S.ema(series, 26))]
    g = S.ema(m, 9)
    n = len(series) - 1
    h = [a - b for a, b in zip(m, g)]
    cross_dn = any(p[k - 1] > s[k - 1] and p[k] <= s[k] for k in range(n - 2, n + 1))
    mdown = any(m[k - 1] > g[k - 1] and m[k] <= g[k] for k in range(n - 2, n + 1))
    return {"p": p[n], "above": p[n] > s[n], "up": p[n] > p[n - 1], "green": h[n] > 0, "hrise": h[n] > h[n - 1], "pdown": cross_dn, "mdown": mdown}


def fmt(f):
    if not f:
        return "؟"
    return (f"PMO {f['p']:+.1f} {'فوق' if f['p'] > 0 else 'تحت'} الصفر {'↑' if f['up'] else '↓'} {'فوق' if f['above'] else 'تحت'} البرتقالي | مكدي {'أخضر' if f['green'] else 'أحمر'}{'↑' if f['hrise'] else '↓'}"
            + (" | ⚠ تقاطع PMO هبوطي حديث" if f["pdown"] else "") + (" | ⚠ تقاطع مكدي هبوطي حديث" if f["mdown"] else ""))


def work(args):
    sym, hours, btc = args
    now = int(time.time())
    try:
        k15, k1, k4, kd = kl(sym, "Min15", now), kl(sym, "Min60", now), kl(sym, "Hour4", now), kl(sym, "Day1", now)
    except Exception:
        return None
    if not (k15 and k1 and k4 and kd) or len(k15["close"]) < 500:
        return None
    t, o, hi, lo, c, vo = (k15[k] for k in ("time", "open", "high", "low", "close", "vol"))
    n = len(c)
    p, ps = S.pmo(c)
    te = S.tema(c, 50)
    m = [a - b for a, b in zip(S.ema(c, 12), S.ema(c, 26))]
    g = S.ema(m, 9)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    _, snaps = LX.run(o, hi, lo, c)
    found = None
    for i in range(n - 1, max(n - 1 - hours * 4, 300), -1):
        touch = p[i - 1] <= 0 < p[i]
        pc = p[i - 1] <= ps[i - 1] and p[i] > ps[i]
        mc = any(m[k - 1] <= g[k - 1] and m[k] > g[k] for k in range(i - 2, i + 1)) or any(m[k - 1] - g[k - 1] <= 0 < m[k] - g[k] for k in range(i - 2, i + 1))
        early = pc and mc
        if touch or early:
            found = (i, touch, early)
            break
    if not found:
        return None
    i, touch, early = found
    ent, tc = c[i], t[i] + 900
    bulls = [ob for ob in snaps[i] if ob["bias"] == 1]
    inside = [ob for ob in bulls if ob["lo"] <= ent <= ob["hi"]]
    below = [ob for ob in bulls if ob["lo"] < ent and ent / ob["lo"] - 1 <= 0.08]
    if inside:
        ob = inside[0]
        obtxt = f"داخل أوردر بلوك [{ob['lo']:.6g}–{ob['hi']:.6g}]"
        stop = ob["lo"]
    elif below:
        ob = max(below, key=lambda x: x["hi"])
        obtxt = f"أقرب أوردر بلوك صاعد تحت السعر [{ob['lo']:.6g}–{ob['hi']:.6g}] بعد {(ent / ob['hi'] - 1) * 100:.1f}%"
        stop = ob["lo"]
    else:
        obtxt = "لا أوردر بلوك صاعد قريب (≤8%)"
        stop = min(lo[i - 15:i + 1])
    base = st.median(vo[max(0, i - 960):i - 8]) or 1e-12
    bi = btc["idx"].get(t[i])
    return {"sym": sym, "t": t[i], "type": ("لمس الصفر" if touch else "") + (" + " if touch and early else "") + ("مبكر (تقاطع PMO+مكدي)" if early else ""),
            "price": ent, "now": c[-1], "p15": p[i], "macd15": m[i] > g[i], "tema": ent > te[i], "atr": atr[i] / ent * 100, "vol": max(vo[i - 7:i + 1]) / base,
            "ob": obtxt, "stop": stop, "stopd": (ent / stop - 1) * 100,
            "h1": frame(k1, 3600, tc, ent), "h4": frame(k4, 14400, tc, ent), "d1": frame(kd, 86400, tc, ent), "btc": btc["state"]}


def main():
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    now = int(time.time())
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    b15, b1, b4 = kl("BTC_USDT", "Min15", now), kl("BTC_USDT", "Min60", now), kl("BTC_USDT", "Hour4", now)
    btc = {"idx": {tt: k for k, tt in enumerate(b15["time"])}, "state": f"1س {fmt(frame(b1, 3600, now, b15['close'][-1]))} | 4س {fmt(frame(b4, 14400, now, b15['close'][-1]))}"}
    with ThreadPoolExecutor(5) as ex:
        res = [r for r in ex.map(work, [(x["symbol"], hours, btc) for x in tick]) if r]
    res.sort(key=lambda r: -r["t"])
    chg = {x["symbol"]: float(x.get("riseFallRate") or 0) * 100 for x in tick}
    print(f"ماسح المرشحين · آخر {hours} ساعات · {len(tick)} عقد كريبتو · التوقيت UTC+{TZ} · {dt(now):%d %H:%M}")
    print("هذه ليست توصيات. لا أضمن ربحاً؛ فحوصاتنا لم تجد قاعدة رابحة مثبتة. القرار لك.")
    print(f"بيتكوين الآن: {btc['state']}\n")
    for r in res:
        print(f"== {r['sym'][:-5]}USDT.P · {dt(r['t']):%d %H:%M} · {r['type']} · سعر الإشارة {r['price']:.6g} (الآن {r['now']:.6g}، 24س {chg.get(r['sym'], 0):+.1f}%)")
        print(f"   15د: PMO {r['p15']:+.2f} · مكدي {'فوق' if r['macd15'] else 'تحت'} الإشارة · {'فوق' if r['tema'] else 'تحت'} TEMA50 · تذبذب ATR {r['atr']:.2f}% · حجم ×{r['vol']:.1f}")
        print(f"   {r['ob']} · وقف مقترح للعرض {r['stop']:.6g} (بعد {r['stopd']:.1f}%)")
        print(f"   1س: {fmt(r['h1'])}")
        print(f"   4س: {fmt(r['h4'])}")
        print(f"   يومي: {fmt(r['d1'])}")
    print(f"\nعدد المرشحين: {len(res)}")


main()
