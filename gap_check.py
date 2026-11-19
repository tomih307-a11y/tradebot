"""For each (coin, PMO zero-touch time in UTC+4): how long before the touch did price last touch a bullish order block that was alive then?
Usage: python3 gap_check.py SYMBOL:DD:HH:MM ... (November 2026, UTC+4)"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import luxalgo_ob as LX

TZ = 4


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))["data"]


def smooth(x, n):
    a, e, out = 2 / n, x[0], []
    for v in x:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)
for spec in sys.argv[1:]:
    sym, dd, hh, mm = spec.split(":")
    now = int(time.time())
    d = get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min15&start={now - 14 * 86400}&end={now}")
    keep = [i for i, tt in enumerate(d["time"]) if tt + 900 <= now]
    t, o, h, l, c = ([d[k][i] for i in keep] for k in ("time", "open", "high", "low", "close"))
    roc = [0.0] + [(c[i] / c[i - 1] - 1) * 100 for i in range(1, len(c))]
    pmo = smooth([10 * v for v in smooth(roc, 35)], 20)
    obs, snaps = LX.run(o, h, l, c)
    want = datetime(2026, 11, int(dd), int(hh), int(mm), tzinfo=timezone.utc) - timedelta(hours=TZ)
    cross = min((i for i in range(1, len(c)) if pmo[i - 1] <= 0 < pmo[i]), key=lambda i: abs(t[i] - want.timestamp()))
    last = None
    for i in range(cross, max(cross - 400, 0), -1):                     # walk back: last bar whose candle overlapped an alive bullish OB
        if any(ob["bias"] == 1 and l[i] <= ob["hi"] and h[i] >= ob["lo"] for ob in snaps[i]):
            last = i
            break
    if last is None:
        print(f"{sym}: لمس PMO {dt(t[cross]):%d %H:%M} · لم يلمس أوردر بلوك صاعد في آخر 100 ساعة")
    else:
        print(f"{sym}: لمس PMO {dt(t[cross]):%d %H:%M} · آخر لمسة لأوردر بلوك صاعد {dt(t[last]):%d %H:%M} · الفاصل {cross - last} شمعة = {(cross - last) * 15} دقيقة")
