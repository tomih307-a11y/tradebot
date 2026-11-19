"""PMO(35,20,10) and MACD(12,26,9) on the 5m frame at the close of a 15m PMO zero-touch bar.
Usage: python3 frames_check3.py SYMBOL:DD:HH:MM ... (time UTC+4, Nov 2026)"""
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
    sec = {"Min5": 300, "Min15": 900}[itv]
    keep = [i for i, tt in enumerate(d["time"]) if tt + sec <= now]
    return [d["time"][i] for i in keep], [d["close"][i] for i in keep]


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


dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)
for spec in sys.argv[1:]:
    sym, dd, hh, mm = spec.split(":")
    want = (datetime(2026, 11, int(dd), int(hh), int(mm), tzinfo=timezone.utc) - timedelta(hours=TZ)).timestamp()
    t15, c15 = kl(sym, "Min15", 14)
    p15, _ = pmo(c15)
    cross = min((i for i in range(1, len(c15)) if p15[i - 1] <= 0 < p15[i]), key=lambda i: abs(t15[i] - want))
    tc = t15[cross] + 900
    t5, c5 = kl(sym, "Min5", 6)
    n = max(k for k in range(len(t5)) if t5[k] + 300 <= tc)
    p, s = pmo(c5[:n + 1])
    m = [a - b for a, b in zip(ema(c5[:n + 1], 12), ema(c5[:n + 1], 26))]
    ms = ema(m, 9)
    h = [a - b for a, b in zip(m, ms)]
    ups = [k for k in range(max(1, n - 11), n + 1) if p[k - 1] <= 0 < p[k]]            # last hour (12 candles)
    downs = [k for k in range(max(1, n - 11), n + 1) if p[k - 1] > 0 >= p[k]]
    mdown = [k for k in range(max(1, n - 11), n + 1) if m[k - 1] > ms[k - 1] and m[k] <= ms[k]]
    print(f"== {sym} · لمس 15د {dt(t15[cross]):%d %H:%M}")
    print(f"   5د PMO: أزرق {p[n]:+.3f} ({'فوق' if p[n] > 0 else 'تحت'} الصفر) · {'فوق' if p[n] > s[n] else 'تحت'} البرتقالي · {'يصعد' if p[n] > p[n - 1] else 'ينزل'} · عبور الصفر صعوداً في آخر ساعة: {len(ups)} · هبوطاً: {len(downs)}")
    print(f"   5د مكدي: هيستوغرام {h[n]:+.2e} ({'أخضر' if h[n] > 0 else 'أحمر'}، {'يكبر' if h[n] > h[n - 1] else 'يصغر'}) · تقاطع هبوطي للمكدي في آخر ساعة: {len(mdown)}")
