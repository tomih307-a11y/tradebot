"""Follow-up of a batch of undated radar alerts. Alert fields: symbol, radar hour (UTC+3), hour-close price, alert price, quote time HH:MM:SS (radar), radar prob %, last-hour %, last-4h %, volume x, state.
Date located by matching a 1H candle (close at radar hour, price within 0.3%, hourly move within 1.5pp). Follow-up on 1-minute candles after the alert (the candle containing the alert is excluded).
Output in UTC+4."""
import json
import statistics as st
import time
from datetime import datetime, timedelta, timezone
import scanner as S

# (day index 0=Oct24,1=Oct25,2=Oct26), sym, close, alertprice, HH:MM:SS, prob, ret1h, ret4h, vol, drop?
A = [
(0,"BIRB_USDT",0.17171,0.18065,"08:02:49",33,-7.2,0.3,6.6,1),(0,"MAGMA_USDT",0.29989,0.29839,"08:02:49",30,-10.8,-22.7,2.0,1),(0,"IRYS_USDT",0.0858,0.08647,"09:02:55",46,-5.2,-16.6,0.9,1),
(0,"BMT_USDT",0.04004,0.03912,"11:03:08",27,-6.0,-2.4,3.7,1),(0,"HOOKED_USDT",0.000978,0.000995,"12:02:47",33,-7.3,-14.7,2.3,1),(0,"AGT_USDT",0.08,0.08178,"12:02:47",27,-5.1,7.8,1.2,1),
(0,"IN_USDT",0.04316,0.04282,"12:02:47",25,-11.6,-2.2,8.3,1),(0,"Q_USDT",0.074597,0.074853,"13:02:46",24,-5.7,-14.5,1.2,1),(0,"BIRB_USDT",0.11365,0.11268,"14:02:48",43,-29.6,-31.6,4.6,1),
(0,"IRYS_USDT",0.06426,0.06481,"15:03:05",43,-10.0,-21.1,3.5,1),(0,"JCT_USDT",0.003777,0.003687,"15:03:05",27,-20.7,-2.8,1.9,1),(0,"MAGMA_USDT",0.22083,0.22044,"16:02:47",43,-25.0,-21.7,5.2,1),
(0,"ORDER_USDT",0.0683,0.06964,"17:02:56",34,-5.4,-16.1,15.6,1),(0,"EPIC_USDT",0.1429,0.1428,"18:02:48",32,2.4,6.5,1.4,0),(0,"BMT_USDT",0.03748,0.0381,"18:02:48",29,1.6,4.9,4.2,0),
(0,"CT_USDT",0.15699,0.15634,"20:02:50",29,-2.8,-10.3,1.2,1),(0,"ORDER_USDT",0.06104,0.06134,"23:03:10",33,0.0,-9.4,2.7,1),(0,"JCT_USDT",0.007091,0.007159,"23:03:10",29,1.0,3.3,0.3,0),
(1,"HOOKED_USDT",0.001723,0.00168,"01:02:56",30,1.6,-22.6,1.2,1),(1,"ANIME_USDT",0.003377,0.003426,"02:02:46",26,-14.5,-6.9,16.2,1),(1,"IRYS_USDT",0.04037,0.04093,"03:03:08",28,-10.0,-8.5,1.0,1),
(1,"BIRB_USDT",0.08724,0.08508,"03:03:08",25,2.0,-5.5,0.9,0),(1,"BMT_USDT",0.03709,0.03735,"03:03:08",24,-2.3,-2.4,0.8,0),(1,"ORDER_USDT",0.05121,0.05159,"05:02:47",28,-8.3,-7.3,1.6,1),
(1,"JCT_USDT",0.00619,0.006277,"05:02:47",26,-4.6,-14.4,0.5,1),(1,"MAGMA_USDT",0.32945,0.32808,"09:02:59",39,-12.7,-11.4,2.3,1),(1,"HOOKED_USDT",0.001533,0.00154,"09:02:59",29,-1.1,0.7,0.8,0),
(1,"IRYS_USDT",0.03987,0.04023,"10:02:48",35,-10.0,6.0,1.3,1),(1,"EPIC_USDT",0.1231,0.1232,"10:02:48",30,-4.3,-17.3,1.8,1),(1,"CLAUDIA_USDT",0.002848,0.002866,"10:02:48",23,-7.3,-10.6,0.6,1),
(1,"JCT_USDT",0.006482,0.00632,"12:02:50",32,-11.1,6.3,1.8,1),(1,"ORDER_USDT",0.04955,0.04942,"14:02:48",27,-5.1,-10.4,1.3,1),(1,"APR_USDT",0.03711,0.03801,"14:02:48",23,0.9,2.5,8.2,0),
(1,"AGT_USDT",0.06585,0.06572,"15:03:05",25,-3.7,-15.1,0.8,1),(1,"HOOKED_USDT",0.002084,0.002095,"17:02:47",29,-3.6,6.3,1.3,1),(1,"Q_USDT",0.067381,0.068146,"18:02:56",27,-15.3,-20.0,2.6,1),
(1,"DARK_USDT",0.000748,0.000737,"21:02:48",32,-9.7,-15.8,1.1,1),(2,"HOOKED_USDT",0.001195,0.001195,"04:02:51",27,-8.9,-14.3,0.5,1),(2,"JCT_USDT",0.004074,0.004217,"06:02:46",42,-25.3,-34.3,1.8,1),
(2,"APR_USDT",0.03273,0.03403,"06:02:46",32,-2.7,-17.3,2.0,1),(2,"ORDER_USDT",0.04083,0.04063,"08:02:48",28,-1.9,-4.2,0.5,0),(2,"AGT_USDT",0.05746,0.05751,"08:02:48",26,-0.9,-5.7,0.3,0),
(2,"HUMA_USDT",0.01785,0.01788,"14:02:48",30,None,None,None,1),
]
UTC3 = timedelta(hours=3); UTC4 = timedelta(hours=4)
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + UTC4
now = int(time.time())


def fetch(sym, itv, step, a, b_end):
    out = {k: [] for k in ("time", "open", "high", "low", "close")}
    while a < b_end:
        b = min(a + step * 1000, b_end)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            raise SystemExit("fetch failed " + sym)
        for k in out:
            out[k] += d[k]
        a = b
    return out


day0 = datetime(2026, 10, 24, 0, 0, 0, tzinfo=timezone.utc)
syms = sorted({a[1] for a in A})
H1, M1 = {}, {}
t_lo, t_hi = int((day0 - timedelta(hours=6)).timestamp()), int((day0 + timedelta(days=5)).timestamp())
for s in syms:
    H1[s] = fetch(s, "Min60", 3600, t_lo, t_hi)
    M1[s] = fetch(s, "Min1", 60, t_lo, t_hi)

rows = []
for di, sym, close, aprice, hhmmss, prob, r1, r4, vol, drop in A:
    h = H1[sym]
    cands = []
    for i in range(4, len(h["time"])):
        close_utc4 = h["time"][i] + 3600
        d = dt4(close_utc4)
        if abs(h["close"][i] / close - 1) > 0.003:
            continue
        radar_close = datetime.fromtimestamp(close_utc4, timezone.utc) + UTC3
        ret1 = (h["close"][i] / h["open"][i] - 1) * 100
        ret4 = (h["close"][i] / h["open"][i - 3] - 1) * 100
        score = abs(ret1 - r1) + abs(ret4 - r4) if r1 is not None else 0
        if r1 is not None and score > 3:
            continue
        cands.append((score, abs((radar_close.date() - (day0 + timedelta(days=di)).date()).days), close_utc4, radar_close))
    if not cands:
        rows.append({"sym": sym, "found": False, "prob": prob, "t": hhmmss, "day": di}); continue
    cands.sort(key=lambda c: (c[1], c[0]))
    _, _, close_utc4, radar_close = cands[0]
    hh, mm, ss = (int(x) for x in hhmmss.split(":"))
    alert_utc = datetime.fromtimestamp(close_utc4, timezone.utc).replace(minute=mm, second=ss) if mm < 59 else None
    a0 = int(alert_utc.timestamp())
    m = M1[sym]
    start = (a0 // 60 + 1) * 60
    idx = [i for i in range(len(m["time"])) if m["time"][i] >= start and m["time"][i] <= a0 + 48 * 3600]
    res = {"sym": sym, "found": True, "prob": prob, "drop": drop, "vol": vol, "a0": a0, "price": aprice}
    for nm, hrs in (("2h", 2), ("6h", 6), ("24h", 24)):
        w = [i for i in idx if m["time"][i] <= a0 + hrs * 3600]
        if not w:
            res[nm] = None; continue
        pk = max(w, key=lambda i: m["high"][i])
        res[nm] = {"max": (m["high"][pk] / aprice - 1) * 100, "dd_before": (min(m["low"][i] for i in w if i <= pk) / aprice - 1) * 100, "min": (min(m["low"][i] for i in w) / aprice - 1) * 100}
    f10 = next((i for i in idx if m["high"][i] >= aprice * 1.10), None)
    m4 = next((i for i in idx if m["low"][i] <= aprice * 0.96), None)
    res["t10"] = (m["time"][f10] - a0) / 3600 if f10 is not None else None
    res["t4"] = (m["time"][m4] - a0) / 3600 if m4 is not None else None
    last24 = [i for i in idx if m["time"][i] <= a0 + 24 * 3600]
    res["close24"] = (m["close"][last24[-1]] / aprice - 1) * 100 if last24 else None
    rows.append(res)
json.dump(rows, open("alert_batch.json", "w"))
ok = [r for r in rows if r["found"]]
print(f"التنبيهات: {len(A)} · طُبقت على شموع: {len(ok)} · لم يُعثر: {len(A)-len(ok)}\n")
print(f"{'#':>2s} {'العملة':9s} {'وقت التنبيه (UTC+4)':20s} {'احتمال':>6s} {'سعر':>9s} | {'أعلى 2س':>8s} {'أعلى 6س':>8s} {'أعلى 24س':>9s} | {'أسوأ نزول قبل قمة 24س':>20s} | {'+10% بعد':>9s} {'-4% بعد':>8s}")
for n, r in enumerate(rows, 1):
    if not r["found"]:
        print(f"{n:2d} {r['sym'][:-5]:9s} لم يُعثر على الشمعة"); continue
    g = lambda w: f"{r[w]['max']:+.1f}%" if r.get(w) else "—"
    print(f"{n:2d} {r['sym'][:-5]:9s} {dt4(r['a0']):%d %b %H:%M:%S}   {r['prob']:5d}% {r['price']:9.6g} | {g('2h'):>8s} {g('6h'):>8s} {g('24h'):>9s} | {r['24h']['dd_before']:+19.1f}% | {('%.1f س' % r['t10']) if r['t10'] is not None else 'لم يصل':>9s} {('%.1f س' % r['t4']) if r['t4'] is not None else 'لم ينزل':>8s}")
n = len(ok)
h2 = sum(1 for r in ok if r["2h"] and r["2h"]["max"] >= 10)
h24 = sum(1 for r in ok if r["24h"] and r["24h"]["max"] >= 10)
print(f"\nالحصيلة ({n} تنبيهاً): وصل +10% خلال ساعتين: {h2} ({h2/n*100:.0f}%) مقابل متوسط احتمال التنبيهات {st.mean(r['prob'] for r in ok):.0f}% · وصل +10% خلال 24 ساعة: {h24} ({h24/n*100:.0f}%)")
dn2 = sum(1 for r in ok if r["t4"] is not None and r["t4"] <= 2); dn24 = sum(1 for r in ok if r["t4"] is not None and r["t4"] <= 24)
print(f"نزل -4% خلال ساعتين: {dn2} ({dn2/n*100:.0f}%) · خلال 24 ساعة: {dn24} ({dn24/n*100:.0f}%)")
print(f"وسيط أعلى سعر: ساعتان {st.median(r['2h']['max'] for r in ok):+.2f}% · 6 ساعات {st.median(r['6h']['max'] for r in ok):+.2f}% · 24 ساعة {st.median(r['24h']['max'] for r in ok):+.2f}% · وسيط أسوأ نزول قبل قمة 24س {st.median(r['24h']['dd_before'] for r in ok):+.2f}% · وسيط سعر بعد 24س {st.median(r['close24'] for r in ok if r['close24'] is not None):+.2f}%")
print("\nحسب احتمال التنبيه:")
for lo_, hi_, nm in ((0, 27, "≤27%"), (28, 33, "28–33%"), (34, 100, "≥34%")):
    g = [r for r in ok if lo_ <= r["prob"] <= hi_]
    if g:
        print(f"  {nm}: {len(g)} تنبيه · وصل +10% خلال ساعتين {sum(1 for r in g if r['2h'] and r['2h']['max']>=10)} · خلال 24س {sum(1 for r in g if r['24h'] and r['24h']['max']>=10)}")
print("حسب الحالة:")
for lab, f in (("بعد هبوط حاد", lambda r: r["drop"] == 1), ("هادئة قبل التنبيه", lambda r: r["drop"] == 0)):
    g = [r for r in ok if f(r)]
    print(f"  {lab}: {len(g)} · وصل +10% خلال ساعتين {sum(1 for r in g if r['2h'] and r['2h']['max']>=10)} · خلال 24س {sum(1 for r in g if r['24h'] and r['24h']['max']>=10)} · وسيط أعلى 24س {st.median(r['24h']['max'] for r in g):+.1f}%")
print("حسب حجم آخر ساعة (أضعاف المعتاد):")
for lab, f in (("≥3 أضعاف", lambda r: r["vol"] is not None and r["vol"] >= 3), ("<3 أضعاف", lambda r: r["vol"] is not None and r["vol"] < 3)):
    g = [r for r in ok if f(r)]
    if g:
        print(f"  {lab}: {len(g)} · وصل +10% خلال ساعتين {sum(1 for r in g if r['2h'] and r['2h']['max']>=10)} · خلال 24س {sum(1 for r in g if r['24h'] and r['24h']['max']>=10)} · وسيط أعلى 24س {st.median(r['24h']['max'] for r in g):+.1f}%")
print("\nمثال: وقف -4% وهدف +10% (قبل الرسوم)، الأول الذي يتحقق خلال 24 ساعة، وإلا السعر بعد 24س:")
for tgt_h in (2, 24):
    res = []
    for r in ok:
        t10 = r["t10"] if r["t10"] is not None and r["t10"] <= tgt_h else None
        t4 = r["t4"] if r["t4"] is not None and r["t4"] <= tgt_h else None
        if t10 is not None and (t4 is None or t10 < t4):
            res.append(10.0)
        elif t4 is not None:
            res.append(-4.0)
        else:
            res.append(r["close24"] if r["close24"] is not None else 0.0)
    print(f"  أفق {tgt_h} ساعة: {len(res)} صفقة · رابحة بالهدف {sum(1 for x in res if x == 10.0)} · خاسرة بالوقف {sum(1 for x in res if x == -4.0)} · المتوسط {st.mean(res):+.2f}% للصفقة · المجموع {sum(res):+.1f}%")
