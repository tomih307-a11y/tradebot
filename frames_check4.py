"""PMO(35,20,10) and MACD(12,26,9) on the 1H frame at a 15m PMO zero-touch, including the forming 1H candle, with bearish-cross detection in the last 3 hours.
Usage: python3 frames_check4.py SYMBOL:DD:HH:MM ... (time UTC+4, Nov 2026)"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

TZ = 4


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))["data"]


def kl(sym, itv, days):
    now = int(time.time())
    d = get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={now - days * 86400}&end={now}")
    return d["time"], d["close"]


def smooth(x, n):
    a, e, out = 2 / n, x[0], []
    for v in x:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


def ema(x, n):
    a, e, out = 2 / (n + 1), x[0], []
    for v in x:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


def pmo(c):
    roc = [0.0] + [(c[i] / c[i - 1] - 1) * 100 for i in range(1, len(c))]
    p = smooth([10 * v for v in smooth(roc, 35)], 20)
    return p, ema(p, 10)


arrow = lambda a, b: "↑" if a > b else "↓"
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)
for spec in sys.argv[1:]:
    sym, dd, hh, mm = spec.split(":")
    want = (datetime(2026, 11, int(dd), int(hh), int(mm), tzinfo=timezone.utc) - timedelta(hours=TZ)).timestamp()
    t, c = kl(sym, "Min15", 14)
    p15, _ = pmo(c)
    cross = min((i for i in range(1, len(c)) if p15[i - 1] <= 0 < p15[i]), key=lambda i: abs(t[i] - want))
    tc = t[cross] + 900
    t1, c1 = kl(sym, "Min60", 40)
    series = [c1[k] for k in range(len(t1)) if t1[k] + 3600 <= tc] + [c[cross]]       # closed 1H candles + the forming one
    p, s = pmo(series)
    m = [a - b for a, b in zip(ema(series, 12), ema(series, 26))]
    ms = ema(m, 9)
    h = [a - b for a, b in zip(m, ms)]
    n = len(series) - 1
    gap, gprev = p[n] - s[n], p[n - 1] - s[n - 1]
    pdown = [k for k in range(n - 2, n + 1) if p[k - 1] > s[k - 1] and p[k] <= s[k]]       # blue crossed below orange within last 3 hours
    mdown = [k for k in range(n - 2, n + 1) if m[k - 1] > ms[k - 1] and m[k] <= ms[k]]
    print(f"== {sym} · لمس 15د {dt(t[cross]):%d %H:%M}")
    print(f"   1س PMO: أزرق {p[n]:+.2f} ({'فوق' if p[n] > 0 else 'تحت'} الصفر) · برتقالي {s[n]:+.2f} · الفرق {gap:+.2f} (قبل ساعة {gprev:+.2f}) {arrow(gap, gprev)} · تقاطع هبوطي PMO في آخر 3 ساعات: {'نعم' if pdown else 'لا'} · الأزرق {'فوق' if gap > 0 else 'تحت'} البرتقالي")
    print(f"   1س مكدي: هيستوغرام {h[n]:+.2e} ({'أخضر' if h[n] > 0 else 'أحمر'}، {arrow(h[n], h[n - 1])}) · خط المكدي {'فوق' if m[n] > ms[n] else 'تحت'} الإشارة · تقاطع هبوطي للمكدي في آخر 3 ساعات: {'نعم' if mdown else 'لا'}")
