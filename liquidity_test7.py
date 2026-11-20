"""Liquidity-entry test (long only). For every 15m PMO(35,20,10) zero-touch (up) in the last 14 days, mark whether the 1H and 4H frames show
"start or approach of liquidity entry" (definition agreed with the user), then print raw forward price numbers (no verdicts).
Frame OK = MACD ok OR PMO ok.  MACD ok: hist>0, or hist<0 and shrinking (rising).  PMO ok: blue>orange, or blue<orange, rising, gap shrinking.
Extra: 4H PMO blue higher than 2 candles ago.  Pass = 1H OK and 4H OK and 4H PMO rising.  Frames include the forming candle (price at the touch)."""
import json
import os
import statistics as st
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import luxalgo_ob as LX
import scanner as S

TZ = 4
SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400}
DAYS = {"Min15": 70, "Min60": 130, "Hour4": 300}
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)


def kline(sym, itv, now):
    step = SEC[itv] * 1000
    start = now - DAYS[itv] * 86400
    out = {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
    a = start
    while a < now:
        b = min(a + step, now)
        d = None
        for r in range(3):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={b}")["data"]
                break
            except Exception:
                time.sleep(1 + r)
        if d is None:
            return None
        for k in out:
            out[k] += d[k]
        a = b
    seen, idx = set(), []
    for i, tt in enumerate(out["time"]):
        if tt not in seen:
            seen.add(tt); idx.append(i)
    o = {k: [out[k][i] for i in idx] for k in out}
    return o["time"], o["high"], o["low"], o["close"], o["open"], o["vol"]


def frame(times, closes, period, tc, price):
    series = [closes[k] for k in range(len(times)) if times[k] + period <= tc][-700:] + [price]
    if len(series) < 150:
        return None
    p, s = S.pmo(series)
    ml = [a - b for a, b in zip(S.ema(series, 12), S.ema(series, 26))]
    sg = S.ema(ml, 9)
    h = [a - b for a, b in zip(ml, sg)]
    n = len(series) - 1
    macd_ok = h[n] > 0 or h[n] > h[n - 1]
    green = h[n] > 0 or any(ml[k - 1] <= sg[k - 1] and ml[k] > sg[k] for k in range(n - 2, n + 1))    # strict: green hist, or bullish cross in last 3 candles
    gap, gp = p[n] - s[n], p[n - 1] - s[n - 1]
    pmo_ok = gap > 0 or (p[n] > p[n - 1] and gap > gp)
    return {"green": green, "macd": macd_ok, "pmo": pmo_ok, "ok": macd_ok or pmo_ok, "rise2": p[n] > p[n - 2]}


BTC = {}


def work(sym):
    now = int(time.time())
    k15, k60, k4 = kline(sym, "Min15", now), kline(sym, "Min60", now), kline(sym, "Hour4", now)
    if not (k15 and k60 and k4):
        return []
    t, hi, lo, c, op, vo = k15
    keep = [i for i in range(len(t)) if t[i] + 900 <= now]
    t, hi, lo, c, op, vo = [[x[i] for i in keep] for x in (t, hi, lo, c, op, vo)]
    p, ps = S.pmo(c)
    te = S.tema(c, 50)
    tr_ = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(len(c))]
    atr = S.ema(tr_, 14)
    m15 = [a - b for a, b in zip(S.ema(c, 12), S.ema(c, 26))]
    g15 = S.ema(m15, 9)
    _, snaps = LX.run(op, hi, lo, c)
    out = []
    last = -99
    for i in range(400, len(c)):
        if os.environ.get("MODE") == "touch":
            if not (p[i - 1] <= 0 < p[i]) or i - last < 8:
                continue
        else:
            pc = p[i - 1] <= ps[i - 1] and p[i] > ps[i]
            mc = any(m15[k - 1] <= g15[k - 1] and m15[k] > g15[k] for k in range(i - 2, i + 1)) or \
                 any(m15[k - 1] - g15[k - 1] <= 0 < m15[k] - g15[k] for k in range(i - 2, i + 1))
            if not (pc and mc) or i - last < 8:
                continue
        last = i
        tc = t[i] + 900
        f1, f4 = frame(k60[0], k60[3], 3600, tc, c[i]), frame(k4[0], k4[3], 14400, tc, c[i])
        if not f1 or not f4:
            continue
        ent = c[i]
        fwd = i + 48 < len(c)
        up = dn = first = None
        if fwd:
            up = (max(hi[i + 1:i + 49]) / ent - 1) * 100
            dn = (min(lo[i + 1:i + 49]) / ent - 1) * 100
            first = 0
            for j in range(i + 1, i + 49):                       # which came first: +3% or -3%
                if hi[j] >= ent * 1.03 and lo[j] <= ent * 0.97:
                    first = 0; break
                if hi[j] >= ent * 1.03:
                    first = 1; break
                if lo[j] <= ent * 0.97:
                    first = -1; break
        b = BTC.get(t[i])
        bulls = [ob for ob in snaps[i] if ob["bias"] == 1]
        inside = any(ob["lo"] <= c[i] <= ob["hi"] for ob in bulls)
        near = any(ob["hi"] < c[i] <= ob["hi"] * 1.03 for ob in bulls)
        base = st.median(vo[max(0, i - 960):i - 8]) or 1e-12                                # median volume of the previous ~10 days
        vm = max(vo[i - 7:i + 1]) / base                                            # biggest 15m volume in the last 2 hours (x median)
        vt = vo[i] / base
        extra = {"vm": vm, "vt": vt, "tema": c[i] > te[i], "m15": m15[i] > g15[i], "ob_in": inside, "ob_near": inside or near}
        trade = None
        if i + 96 < len(c):
            stops = {"fixed2": ent * 0.98, "swing": min(lo[i - 15:i + 1]) * 0.999, "atr": ent - 1.5 * atr[i]}
            near_obs = [ob for ob in bulls if ob["lo"] < ent and ent / ob["lo"] - 1 <= 0.08]
            if near_obs:
                stops["ob"] = max(near_obs, key=lambda ob: ob["hi"])["lo"] * 0.9995
            else:
                stops["ob"] = ent * 0.98
            trade = {}
            for sn, stp in stops.items():
                stp = max(min(stp, ent * 0.995), ent * 0.94)                      # distance between 0.5% and 6%
                risk = ent - stp
                cost = 0.001 * ent / risk
                for name in ("A", "B"):
                    R = None
                    for j in range(i + 1, i + 97):
                        if lo[j] <= stp:
                            R = -1.0; break
                        if name == "A" and c[j] < te[j] and (m15[j] - g15[j]) < 0:
                            R = (c[j] - ent) / risk; break
                        if name == "B" and m15[j] < g15[j] and m15[j - 1] >= g15[j - 1]:
                            R = (c[j] - ent) / risk; break
                    if R is None:
                        R = (c[i + 96] - ent) / risk
                    trade[f"{name}_{sn}"] = R - cost
        later = None
        for j in range(i + 1, min(i + 49, len(c))):
            if p[j - 1] <= 0 < p[j]:
                later = (j - i, (c[j] / ent - 1) * 100); break
        if p[i] > 0:
            later = None
        out.append({"trade": trade, "later": later, "below0": p[i] < 0, "sym": sym, "t": t[i], "f1": f1, "f4": f4, "loose": f1["ok"] and f4["ok"] and f4["rise2"], "pass": f1["green"] and f4["green"], "pass2": f1["green"] and f4["green"] and f4["rise2"], "btc": b, "x": extra, "up": up, "dn": dn, "first": first, "fwd": fwd})
    return out


def report(name, rows):
    raw = [r for r in rows if r["fwd"]]
    if not raw:
        print(f"{name:34s} 0")
        return
    ups, dns = [r["up"] for r in raw], [r["dn"] for r in raw]
    a = sum(1 for r in raw if r["first"] == 1) / len(raw) * 100
    b = sum(1 for r in raw if r["first"] == -1) / len(raw) * 100
    slots = {}                                                       # one vote per 15m time slot (touches in the same slot share the market move)
    for r in raw:
        slots.setdefault(r["t"], []).append(r)
    sa = st.mean(sum(1 for x in v if x["first"] == 1) / len(v) for v in slots.values()) * 100
    sb = st.mean(sum(1 for x in v if x["first"] == -1) / len(v) for v in slots.values()) * 100
    su = st.mean(st.mean(x["up"] for x in v) for v in slots.values())
    sd = st.mean(st.mean(x["dn"] for x in v) for v in slots.values())
    print(f"{name:34s} {len(raw):5d} لمسة | وسيط صعود {st.median(ups):+5.2f}% هبوط {st.median(dns):+5.2f}% | +3% أولاً {a:4.1f}% -3% أولاً {b:4.1f}% || {len(slots):4d} وقت: متوسط صعود {su:+5.2f}% هبوط {sd:+5.2f}% | +3% {sa:4.1f}% -3% {sb:4.1f}%")


ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "TQQQ", "SQQQ", "TSLL", "NVDL", "MSTU", "CONL", "SPCX"}


def ok_name(sym):
    base = sym[:-5]
    return not (base.endswith("STOCK") or base in ETF)


def btc_load():
    now = int(time.time())
    k15, k60, k4 = kline("BTC_USDT", "Min15", now), kline("BTC_USDT", "Min60", now), kline("BTC_USDT", "Hour4", now)
    for i in range(len(k15[0])):
        tc = k15[0][i] + 900
        if tc > now:
            continue
        f1, f4 = frame(k60[0], k60[3], 3600, tc, k15[3][i]), frame(k4[0], k4[3], 14400, tc, k15[3][i])
        if f1 and f4:
            BTC[k15[0][i]] = f1["green"] and f4["green"]


def main():
    detail = {d["symbol"]: d for d in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000]
    removed = [x for x in tick if not ok_name(x)]
    tick = [x for x in tick if ok_name(x)]
    btc_load()
    print(f"عملات مفحوصة: {len(tick)} (استُبعد {len(removed)} سهماً/صندوقاً) · آخر 14 يوماً · دخول = إغلاق شمعة لمس الصفر · التوقيت UTC+{TZ}")
    print("المفحوصة:", " ".join(x[:-5] for x in tick), "\n")
    with ThreadPoolExecutor(6) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    print("الشرط الصارم = المكدي أخضر أو تقاطع صاعد حديث (آخر 3 شموع) على الساعة والأربع ساعات معاً")
    print("بيتكوين OK = نفس الشرط على بيتكوين لحظة اللمس\n")
    import random
    mode = "لمس الصفر" if os.environ.get("MODE") == "touch" else "الدخول المبكر (تقاطع PMO + مكدي)"
    print(f"نوع الدخول: {mode} · بلا أي شرط أوردر بلوك على الدخول · دخول عند إغلاق الشمعة · 15د · 70 يوماً")
    print("الوقف: ثابت 2% | آخر قاع 16 شمعة | 1.5×ATR | تحت أقرب أوردر بلوك (للمقارنة فقط). مسافة الوقف بين 0.5% و6%. تكلفة 0.1%. خروج A: TEMA50+أحمر | B: تقاطع المكدي هبوطاً. أقصى 24 ساعة.\n")
    rnd = random.Random(1)
    X = lambda r, k: r["x"][k]

    def ci(sel, key):
        tr = [r for r in rows if r["trade"] and sel(r)]
        slots = {}
        for r in tr:
            slots.setdefault(r["t"], []).append(r["trade"][key])
        keys = list(slots)
        bs = []
        for _ in range(250):
            ks = [rnd.choice(keys) for _ in keys]
            bs.append(sum(sum(slots[k]) for k in ks) / sum(len(slots[k]) for k in ks))
        bs.sort()
        m = st.mean(r["trade"][key] for r in tr)
        return f"{m:+.3f} [{bs[6]:+.2f},{bs[243]:+.2f}]"

    groups = [("كل الإشارات", lambda r: True), ("تحت الصفر (PMO)", lambda r: r["below0"]), ("الفريمات الكبرى (صارم)", lambda r: r["pass"]),
              ("صارم+4س+TEMA+مكدي (بلا OB)", lambda r: r["pass2"] and X(r, "tema") and X(r, "m15")), ("حجم ≥×3", lambda r: X(r, "vm") >= 3),
              ("بيتكوين OK", lambda r: r["btc"]), ("بيتكوين غير OK", lambda r: not r["btc"])]
    for ex in ("A", "B"):
        print(f"== خروج {ex}: متوسط R [حدّ الثقة 95%]")
        print(f"{'':30s} {'صفقات':>6s}  {'وقف 2%':>20s} {'قاع 16 شمعة':>20s} {'1.5×ATR':>20s} {'أقرب OB':>20s}")
        for nm, sel in groups:
            n = sum(1 for r in rows if r["trade"] and sel(r))
            if n < 20:
                continue
            print(f"{nm:30s} {n:6d}  " + " ".join(f"{ci(sel, f'{ex}_{sn}'):>20s}" for sn in ("fixed2", "swing", "atr", "ob")))
        print()
    print("== الحركة (+3% قبل -3% خلال 12س)")
    report("كل الإشارات", rows)
    report("الفريمات الكبرى (صارم)", [r for r in rows if r["pass"]])
    report("حجم ≥×3", [r for r in rows if X(r, "vm") >= 3])


if __name__ == "__main__":
    main()
