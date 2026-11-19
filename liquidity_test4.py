"""Liquidity-entry test (long only). For every 15m PMO(35,20,10) zero-touch (up) in the last 14 days, mark whether the 1H and 4H frames show
"start or approach of liquidity entry" (definition agreed with the user), then print raw forward price numbers (no verdicts).
Frame OK = MACD ok OR PMO ok.  MACD ok: hist>0, or hist<0 and shrinking (rising).  PMO ok: blue>orange, or blue<orange, rising, gap shrinking.
Extra: 4H PMO blue higher than 2 candles ago.  Pass = 1H OK and 4H OK and 4H PMO rising.  Frames include the forming candle (price at the touch)."""
import json
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
    p, _ = S.pmo(c)
    te = S.tema(c, 50)
    m15 = [a - b for a, b in zip(S.ema(c, 12), S.ema(c, 26))]
    g15 = S.ema(m15, 9)
    _, snaps = LX.run(op, hi, lo, c)
    out = []
    for i in range(400, len(c)):
        if not (p[i - 1] <= 0 < p[i]):
            continue
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
        out.append({"sym": sym, "t": t[i], "f1": f1, "f4": f4, "loose": f1["ok"] and f4["ok"] and f4["rise2"], "pass": f1["green"] and f4["green"], "pass2": f1["green"] and f4["green"] and f4["rise2"], "btc": b, "x": extra, "up": up, "dn": dn, "first": first, "fwd": fwd})
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
    print("== اختبار قفزة الحجم (أكبر حجم 15د في آخر ساعتين مقارنة بوسيط 10 أيام)")
    V = lambda r: r["x"]["vm"]
    for lo_, hi_, nm in ((0, 2, "أقل من ×2"), (2, 5, "×2 إلى ×5"), (5, 10, "×5 إلى ×10"), (10, 25, "×10 إلى ×25"), (25, 1e9, "×25 فأكثر")):
        report(f"حجم {nm}", [r for r in rows if lo_ <= V(r) < hi_])
    print()
    for thr in (3, 5, 10):
        report(f"حجم ≥×{thr}: الكل", [r for r in rows if V(r) >= thr])
        report(f"حجم ≥×{thr} + الفريمات الكبرى (صارم)", [r for r in rows if V(r) >= thr and r["pass"]])
        report(f"حجم ≥×{thr} + ليس صارم", [r for r in rows if V(r) >= thr and not r["pass"]])
        report(f"حجم ≥×{thr} + قرب أوردر بلوك", [r for r in rows if V(r) >= thr and r["x"]["ob_near"]])
        report(f"حجم ≥×{thr} + بيتكوين OK", [r for r in rows if V(r) >= thr and r["btc"]])
        report(f"حجم ≥×{thr} + شمعة اللمس نفسها ≥×3", [r for r in rows if V(r) >= thr and r["x"]["vt"] >= 3])
        print()
    print("مرجع: كل اللمسات")
    report("كل اللمسات", rows)
    report("حجم أقل من ×3", [r for r in rows if V(r) < 3])
    json.dump(rows, open("liquidity_rows.json", "w"))


if __name__ == "__main__":
    main()
