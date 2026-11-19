"""Check my PMO implementation against the numbers in the user's two TradingView screenshots (LUNC 15m): PMO(35,20,10) and PMO(4,20,10)."""
import json
import time
import urllib.request


def klines(sym, interval, days):
    now = int(time.time())
    url = f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={interval}&start={now - days * 86400}&end={now}"
    d = json.load(urllib.request.urlopen(url, timeout=60))["data"]
    return d["time"], d["close"]


def smooth(x, n):
    """DecisionPoint custom EMA: alpha = 2/n."""
    a, out, e = 2 / n, [], x[0]
    for v in x:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


def ema(x, n):
    a, out, e = 2 / (n + 1), [], x[0]
    for v in x:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


def pmo(close, l1, l2, sig):
    roc = [0.0] + [(close[i] / close[i - 1] - 1) * 100 for i in range(1, len(close))]
    p = smooth([10 * v for v in smooth(roc, l1)], l2)
    return p, ema(p, sig)


def up_crossings(p, t):
    return [t[i] for i in range(1, len(p)) if p[i - 1] <= 0 < p[i]]


t, c = klines("LUNC_USDT", "Min15", 14)
fmt = lambda x: time.strftime("%m-%d %H:%M", time.gmtime(x))
print(f"LUNC_USDT 15m: {len(c)} شمعة من {fmt(t[0])} إلى {fmt(t[-1])} UTC")
res = {}
for name, (l1, l2, s) in {"افتراضي 35-20-10": (35, 20, 10), "إعدادك 4-20-10": (4, 20, 10)}.items():
    p, sg = pmo(c, l1, l2, s)
    res[name] = (p, sg)
tgt = {"افتراضي 35-20-10": (1.78091552, 2.63830426), "إعدادك 4-20-10": (-0.53105232, -0.49814076)}
print("\nبحث عن الشمعة التي تطابق أرقام صورتيك معاً (الأزرق، البرتقالي):")
best = []
for i in range(300, len(c)):
    err = 0
    for name, (b, o) in tgt.items():
        p, sg = res[name]
        err += abs(p[i] - b) + abs(sg[i] - o)
    best.append((err, i))
best.sort()
for err, i in best[:4]:
    print(f"  {fmt(t[i])} UTC · إغلاق {c[i]:.8f} · خطأ كلي {err:.3f} · افتراضي ({res['افتراضي 35-20-10'][0][i]:.4f}, {res['افتراضي 35-20-10'][1][i]:.4f}) · إعدادك ({res['إعدادك 4-20-10'][0][i]:.4f}, {res['إعدادك 4-20-10'][1][i]:.4f})")
print("\nلحظات عبور PMO الأزرق للصفر صعوداً (آخر 4 أيام):")
for name in res:
    cr = [x for x in up_crossings(res[name][0], t) if x > t[-1] - 4 * 86400]
    print(f"  {name}: " + "، ".join(fmt(x) for x in cr))
open("verify.txt", "w", encoding="utf-8").write("done")
