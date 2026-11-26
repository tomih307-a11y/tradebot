"""Oct 14-17 channel signals with real post times (Telegram local = Riyadh UTC+3). 7-day horizon on 1m candles (5m fallback).
Per signal: price at post vs posted entry; was TP1 already reached in the 30 min before posting; TPs reached before the channel's own stop rule (close of its TF under the stop);
exit at that close; wick stop; worst drawdown; equal-tranche position result; 'any TP' (channel's definition) vs net-positive position."""
import time
from datetime import datetime, timezone
import scanner as S

now = int(time.time())
P = lambda mo, d, h, mi: int(datetime(2026, mo, d, h, mi, tzinfo=timezone.utc).timestamp()) - 3 * 3600
TF = {"15m": 900, "1H": 3600, "4H": 14400}
SIG = [
    ("BTW", P(10, 14, 15, 58), 2.17, 2.0615, [2.2351, 2.2965, 2.3581, 2.4196, 2.4810, 2.5426, 2.6040], "4H"),
    ("MITO", P(10, 14, 16, 11), 0.015260, 0.014497, [0.015718, 0.016150, 0.016583, 0.017015, 0.017447, 0.017880, 0.018312], "4H"),
    ("BTW", P(10, 16, 3, 12), 1.8685, 1.7938, [1.9246, 1.9806, 2.0367, 2.0927, 2.1488], "1H"),
    ("STBL", P(10, 16, 3, 19), 0.03868, 0.037133, [0.03984, 0.041001, 0.042161, 0.043322, 0.044482], "15m"),
    ("ZRO", P(10, 16, 3, 21), 2.1289, 2.0012, [2.1928, 2.537, 2.881, 3.2253, 3.5695, 3.9135, 4.2578], "4H"),
    ("WLD", P(10, 16, 11, 42), 0.5305, 0.5199, [0.5438, 0.557, 0.5703, 0.5836], "4H"),
    ("ZK", P(10, 16, 11, 43), 0.01807, 0.01735, [0.01852, 0.01965, 0.02078, 0.02191, 0.02304, 0.02417, 0.0253], "4H"),
    ("AKE", P(10, 16, 11, 54), 0.0313, 0.0297, [0.0322, 0.0333, 0.0344, 0.0354, 0.0365, 0.0376], "4H"),
    ("TIA", P(10, 16, 12, 19), 0.684, 0.663, [0.701, 0.714, 0.727, 0.74, 0.752], "4H"),
    ("SIREN", P(10, 17, 1, 11), 0.03697, 0.03586, [0.03789, 0.03882, 0.03974, 0.04067], "15m"),
    ("WLD", P(10, 17, 1, 14), 0.5138, 0.48811, [0.5266, 0.5459, 0.5652, 0.5844, 0.6037, 0.623, 0.6423], "1H"),
    ("DNA", P(10, 17, 2, 3), 0.28599, 0.27741, [0.29171, 0.29457, 0.30029, 0.30601, 0.31459], "15m"),
    ("Q", P(10, 17, 3, 14), 0.0544, 0.0517, [0.0558, 0.0595, 0.0631, 0.0668, 0.0705, 0.0742, 0.0779, 0.0816], "4H"),
]
r3 = lambda t: datetime.fromtimestamp(t + 3 * 3600, timezone.utc).strftime("%d %H:%M")


def get(url):
    for r in range(4):
        try:
            return S.get(url)["data"]
        except Exception:
            time.sleep(1.5 + r)
    return None


def kl(sym, itv, a, b, step):
    out = {k: [] for k in ("time", "high", "low", "close")}
    while a < b:
        e = min(a + step * 1000, b)
        d = get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={e}")
        if d and d.get("time"):
            for k in out:
                out[k] += d[k]
        a = e
    return out


def kl_spot(sym, itv, a, b, step):
    out = {k: [] for k in ("time", "high", "low", "close")}
    while a < b:
        e = min(a + step * 1000, b)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://api.mexc.com/api/v3/klines?symbol={sym}&interval={itv}&startTime={a*1000}&endTime={e*1000}&limit=1000"); break
            except Exception:
                time.sleep(1.5 + r)
        for x in d or []:
            out["time"].append(int(x[0]) // 1000); out["high"].append(float(x[2])); out["low"].append(float(x[3])); out["close"].append(float(x[4]))
        a = e
    return out


res = []
for name, t0, entry, stop, tps, tf in SIG:
    sym = name + "_USDT"
    end = min(now, t0 + 7 * 86400)
    fut = name == "TIA"
    if fut:
        m, step = kl(sym, "Min1", t0 - 1800, end, 60), 60
        if len(m["time"]) < 0.8 * (end - t0 + 1800) / 60:
            m, step = kl(sym, "Min5", t0 - 1800, end, 300), 300
    else:
        m, step = kl_spot(name + "USDT", "1m", t0 - 1800, end, 60), 60
        if len(m["time"]) < 0.8 * (end - t0 + 1800) / 60:
            m, step = kl_spot(name + "USDT", "5m", t0 - 1800, end, 300), 300
    if not m["time"]:
        print(f"== {name} {r3(t0)}: لا بيانات\n"); continue
    T = m["time"]
    pre = [i for i in range(len(T)) if T[i] < t0]
    post = [i for i in range(len(T)) if T[i] >= t0]
    px = m["close"][pre[-1]] if pre else m["close"][post[0]]
    late = bool(pre) and max(m["high"][i] for i in pre) >= tps[0]
    # channel's own stop: first closed TF candle after post with close < stop
    sec = TF[tf]
    bars = {}
    for i in post:
        b = T[i] // sec * sec
        bars.setdefault(b, []).append(i)
    tc = None
    for b in sorted(bars):
        if b + sec <= now and b + sec > t0:
            last = bars[b][-1]
            if T[last] + step >= b + sec and m["close"][last] < stop:
                tc = (b + sec, m["close"][last]); break
    wick = next((T[i] for i in post if m["low"][i] <= stop), None)
    cut = tc[0] if tc else 10 ** 12
    reach = [next((T[i] for i in post if m["high"][i] >= tp and T[i] < cut), None) for tp in tps]
    k = sum(x is not None for x in reach)
    lo = min(m["low"][i] for i in post)
    ex = (tc[1] / entry - 1) * 100 if tc else (m["close"][post[-1]] / entry - 1) * 100
    pos = (sum((tp / entry - 1) * 100 for tp, x in zip(tps, reach) if x) + (len(tps) - k) * ex) / len(tps)
    res.append((name, k, len(tps), tc is not None, pos, wick is not None))
    print(f"== {name} · نُشرت {r3(t0)} · دخول {entry} · وقف {stop} ({tf}) · بيانات {'عقود' if fut else 'سبوت'} {'1د' if step == 60 else '5د'}")
    print(f"   السعر لحظة النشر {px:.6g} ({(px/entry-1)*100:+.2f}% عن الدخول){' · TP1 كان قد تحقق قبل النشر!' if late else ''}")
    print(f"   الأهداف قبل وقف الإغلاق: {k}/{len(tps)} · أوقاتها: " + ", ".join(f"TP{i+1} {r3(x)}" for i, x in enumerate(reach) if x))
    print(f"   وقف بإغلاق {tf}: {('نعم ' + r3(tc[0]) + f' عند {tc[1]:.6g} ({ex:+.2f}%)') if tc else 'لا'} · وقف بالفتيل: {r3(wick) if wick else 'لا'} · أدنى سعر {(lo/entry-1)*100:+.2f}% · نتيجة المركز (شرائح متساوية) {pos:+.2f}%\n")
n = len(res)
print(f"== ملخص {n} توصية (أكتوبر 14-17)")
print(f"  بتعريفه (أي هدف بلغ): {sum(r[1] >= 1 for r in res)}/{n}")
print(f"  ضُرب وقف الإغلاق: {sum(r[3] for r in res)}/{n}")
print(f"  المركز الصافي موجب: {sum(r[4] > 0 for r in res)}/{n} · متوسط المركز {sum(r[4] for r in res)/n:+.2f}%")
