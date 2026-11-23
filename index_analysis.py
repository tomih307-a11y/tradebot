"""S&P 500 analysis from the MEXC perpetual SPX500_USDT (price source: Hyperliquid), cross-checked with SPY_USDT. PMO(35,20,10), MACD(12,26,9), TEMA50 on 15m/1H/4H/daily,
daily moving averages, ATR, swing levels, LuxAlgo-style order blocks, last daily candles, and the weekend move since the US cash close (Friday 21:00 UTC). Times UTC+4."""
import sys
import time
from datetime import datetime, timedelta, timezone
import luxalgo_ob as LX
import scanner as S

SYM = sys.argv[1] if len(sys.argv) > 1 else "SPX500_USDT"
now = int(time.time())
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)
SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}


def kl(itv, days):
    out = {k: [] for k in ("time", "open", "high", "low", "close")}
    a = now - days * 86400
    while a < now:
        b = min(a + SEC[itv] * 1000, now)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{SYM}?interval={itv}&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            raise SystemExit("fetch failed")
        for k in out:
            out[k] += d[k]
        a = b
    seen, keep = set(), []
    for i, t in enumerate(out["time"]):
        if t not in seen:
            seen.add(t); keep.append(i)
    return {k: [out[k][i] for i in keep] for k in out}


tk = next(x for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"] == SYM)
D = {"Min15": kl("Min15", 14), "Min60": kl("Min60", 60), "Hour4": kl("Hour4", 200), "Day1": kl("Day1", 400)}
d1 = D["Day1"]
price = float(tk["lastPrice"])
print(f"{SYM} · السعر الآن {price} · 24س {float(tk.get('riseFallRate') or 0)*100:+.2f}% · حجم 24س {float(tk['amount24'])/1e6:.1f}M$ · شموع يومية متاحة: {len(d1['close'])} (من {dt4(d1['time'][0]):%d %b %Y})")
print("\nآخر 10 شموع يومية (UTC):")
for i in range(len(d1["close"]) - 10, len(d1["close"])):
    o, h, l, c = d1["open"][i], d1["high"][i], d1["low"][i], d1["close"][i]
    print(f"  {datetime.fromtimestamp(d1['time'][i], timezone.utc):%a %d %b}: فتح {o:.1f} أعلى {h:.1f} أدنى {l:.1f} إغلاق {c:.1f} ({(c/o-1)*100:+.2f}%) · مدى {(h/l-1)*100:.2f}%")
c = d1["close"]
hi_all = max(d1["high"]); i_hi = d1["high"].index(hi_all)
print(f"\nأعلى سعر في الفترة: {hi_all:.1f} ({datetime.fromtimestamp(d1['time'][i_hi], timezone.utc):%d %b %Y}) · السعر الآن {(price/hi_all-1)*100:+.1f}% عنه")
for n in (5, 20, 60):
    print(f"  تغير آخر {n} يوماً: {(price/c[-n-1]-1)*100:+.1f}%", end="")
print()
for n in (20, 50, 100, 200):
    if len(c) > n:
        ma = sum(c[-n:]) / n
        print(f"  متوسط {n} يوماً: {ma:.1f} (السعر {(price/ma-1)*100:+.1f}% عنه)")
tr = [d1["high"][i] - d1["low"][i] if i == 0 else max(d1["high"][i] - d1["low"][i], abs(d1["high"][i] - c[i - 1]), abs(d1["low"][i] - c[i - 1])) for i in range(len(c))]
atr = S.ema(tr, 14)[-1]
print(f"  ATR يومي (14): {atr:.1f} نقطة = {atr/price*100:.2f}% من السعر → المدى اليومي المعتاد تقريباً {price-atr:.0f} إلى {price+atr:.0f}")

print("\nحالة المؤشرات (PMO 35,20,10 · مكدي 12,26,9 · TEMA50):")
for nm, k, per in (("15 دقيقة", "Min15", 900), ("ساعة", "Min60", 3600), ("4 ساعات", "Hour4", 14400), ("يومي", "Day1", 86400)):
    d = D[k]
    cl = d["close"]
    p, ps = S.pmo(cl)
    m = [a - b for a, b in zip(S.ema(cl, 12), S.ema(cl, 26))]
    g = S.ema(m, 9)
    te = S.tema(cl, 50)
    h = m[-1] - g[-1]
    print(f"  {nm}: PMO {p[-1]:+.2f} ({'فوق' if p[-1] > 0 else 'تحت'} الصفر) {'↑' if p[-1] > p[-2] else '↓'} {'فوق' if p[-1] > ps[-1] else 'تحت'} البرتقالي · مكدي {'أخضر' if h > 0 else 'أحمر'} {'↑' if h > m[-2] - g[-2] else '↓'} · السعر {'فوق' if cl[-1] > te[-1] else 'تحت'} TEMA50 ({(cl[-1]/te[-1]-1)*100:+.2f}%)")

print("\nقمم وقيعان بارزة (يومي، ±3 أيام) قريبة من السعر:")
lv = []
for i in range(3, len(c) - 3):
    if d1["high"][i] == max(d1["high"][i - 3:i + 4]):
        lv.append((d1["high"][i], "قمة", d1["time"][i]))
    if d1["low"][i] == min(d1["low"][i - 3:i + 4]):
        lv.append((d1["low"][i], "قاع", d1["time"][i]))
near = sorted([x for x in lv if abs(x[0] / price - 1) <= 0.07], key=lambda x: -x[0])
for p_, kind, t_ in near[:12]:
    print(f"  {p_:9.1f} {kind} ({datetime.fromtimestamp(t_, timezone.utc):%d %b}) · {(p_/price-1)*100:+.1f}% عن السعر")

print("\nأوردر بلوك (بمنطق LuxAlgo) الحية الآن:")
for nm, k in (("ساعة", "Min60"), ("4 ساعات", "Hour4"), ("يومي", "Day1")):
    d = D[k]
    if len(d["close"]) < 250:
        print(f"  {nm}: بيانات غير كافية ({len(d['close'])} شمعة)"); continue
    _, snaps = LX.run(d["open"], d["high"], d["low"], d["close"])
    for ob in snaps[-1]:
        print(f"  {nm}: {'صاعد' if ob['bias'] == 1 else 'هابط'} [{ob['lo']:.1f} – {ob['hi']:.1f}] · {(ob['lo']/price-1)*100:+.1f}% إلى {(ob['hi']/price-1)*100:+.1f}% عن السعر")

# weekend move since the US cash close (Friday 16:00 ET = 21:00 UTC in November)
h1 = D["Min60"]
fr = [i for i in range(len(h1["time"])) if datetime.fromtimestamp(h1["time"][i], timezone.utc).weekday() == 4 and datetime.fromtimestamp(h1["time"][i], timezone.utc).hour == 20]
if fr:
    i = fr[-1]
    print(f"\nمنذ إغلاق السوق الأمريكي (الجمعة {datetime.fromtimestamp(h1['time'][i], timezone.utc):%d %b} 21:00 UTC): الإغلاق {h1['close'][i]:.1f} ← الآن {price:.1f} ({(price/h1['close'][i]-1)*100:+.2f}%) · أعلى {max(h1['high'][i+1:]):.1f} · أدنى {min(h1['low'][i+1:]):.1f}")
