"""PMO(35,20,10) state on 15m trough, 1H, 4H and daily at the moment of a 15m PMO zero-touch (time in UTC+4, Nov 2026).
Usage: python3 frames_check.py SYMBOL:DD:HH:MM ..."""
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

TZ = 4
SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))["data"]


def kl(sym, itv, days):
    now = int(time.time())
    d = get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={now - days * 86400}&end={now}")
    keep = [i for i, tt in enumerate(d["time"]) if tt + SEC[itv] <= now]
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


def state(p, s, j):
    side = "فوق الصفر" if p[j] > 0 else "تحت الصفر"
    cross = "الأزرق فوق البرتقالي" if p[j] > s[j] else "الأزرق تحت البرتقالي"
    slope = "يصعد" if p[j] > p[j - 1] else "ينزل"
    since = next((j - k for k in range(j, 0, -1) if (p[k - 1] <= 0) != (p[k] <= 0)), None)
    return f"{p[j]:+.2f} ({side}، {cross}، {slope})" + (f" · منذ آخر عبور للصفر {since} شمعة" if since is not None else "")


dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)
for spec in sys.argv[1:]:
    sym, dd, hh, mm = spec.split(":")
    want = (datetime(2026, 11, int(dd), int(hh), int(mm), tzinfo=timezone.utc) - timedelta(hours=TZ)).timestamp()
    t, c = kl(sym, "Min15", 14)
    p, s = pmo(c)
    cross = min((i for i in range(1, len(c)) if p[i - 1] <= 0 < p[i]), key=lambda i: abs(t[i] - want))
    tc = t[cross] + 900
    print(f"== {sym} · لمس 15د عند {dt(t[cross]):%d %H:%M} · قاع PMO(15د) في آخر 12 ساعة: {min(p[max(cross - 48, 0):cross]):+.2f}")
    for itv, days, nm in (("Min60", 40, "ساعة"), ("Hour4", 80, "4 ساعات"), ("Day1", 260, "يومي")):
        tt, cc = kl(sym, itv, days)
        if len(cc) < 60:
            print(f"   {nm}: بيانات قليلة"); continue
        pp, ss = pmo(cc)
        j = max(k for k in range(len(tt)) if tt[k] + SEC[itv] <= tc)
        print(f"   {nm}: {state(pp, ss, j)}")
