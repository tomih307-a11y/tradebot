"""MACD(12,26,9) and PMO(35,20,10) on the 4H frame AT the moment of a 15m PMO zero-touch, including the still-forming 4H candle.
Usage: python3 frames_check2.py SYMBOL:DD:HH:MM ...  (time UTC+4, Nov 2026)"""
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


def macd(c):
    m = [a - b for a, b in zip(ema(c, 12), ema(c, 26))]
    s = ema(m, 9)
    return m, s, [a - b for a, b in zip(m, s)]


arrow = lambda a, b: "↑" if a > b else "↓"
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)
for spec in sys.argv[1:]:
    sym, dd, hh, mm = spec.split(":")
    want = (datetime(2026, 11, int(dd), int(hh), int(mm), tzinfo=timezone.utc) - timedelta(hours=TZ)).timestamp()
    t, c = kl(sym, "Min15", 14)
    p, s = pmo(c)
    cross = min((i for i in range(1, len(c)) if p[i - 1] <= 0 < p[i]), key=lambda i: abs(t[i] - want))
    tc = t[cross] + 900
    t4, c4 = kl(sym, "Hour4", 90)
    done = [c4[k] for k in range(len(t4)) if t4[k] + 14400 <= tc]            # closed 4H candles before the touch
    series = done + [c[cross]]                                              # + the forming 4H candle (price at the touch)
    pp, ps = pmo(series)
    mm_, ms, mh = macd(series)
    n = len(series) - 1
    gap, gap_prev = pp[n] - ps[n], pp[n - 1] - ps[n - 1]
    print(f"== {sym} · لمس 15د {dt(t[cross]):%d %H:%M}")
    print(f"   4س PMO: أزرق {pp[n]:+.2f} · برتقالي {ps[n]:+.2f} · الفرق {gap:+.2f} (قبل شمعة: {gap_prev:+.2f}) {arrow(gap, gap_prev)} · الأزرق {arrow(pp[n], pp[n - 1])}")
    print(f"   4س مكدي: خط {mm_[n]:+.3e} · إشارة {ms[n]:+.3e} · هيستوغرام {mh[n]:+.3e} (قبل شمعة: {mh[n - 1]:+.3e}) {arrow(mh[n], mh[n - 1])} · {'أحمر' if mh[n] < 0 else 'أخضر'}")
