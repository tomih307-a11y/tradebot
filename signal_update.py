"""Status update of the two channel signals (entry at the 25 Nov 21:00 UTC+4 hourly close): TPs reached with times, stop hit, highest/lowest since, current price. Times UTC+4."""
import time
from datetime import datetime, timedelta, timezone
import scanner as S

SIGNALS = [
    ("AT_USDT", 0.0544, 0.0517, [0.056, 0.0579, 0.0597, 0.0616, 0.0634, 0.0653]),
    ("MOODENG_USDT", 0.02792, 0.02652, [0.02876, 0.02971, 0.03066, 0.03161, 0.03255, 0.0335]),
]
T0 = int((datetime(2026, 11, 25, 21, 0, 0, tzinfo=timezone.utc) - timedelta(hours=4)).timestamp())
now = int(time.time())
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)


def fetch(sym):
    out = {k: [] for k in ("time", "high", "low", "close")}
    a = T0 - 60
    while a < now:
        b = min(a + 60000, now)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min1&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            raise SystemExit("fetch failed")
        for k in out:
            out[k] += d[k]
        a = b
    return out


print(f"التحديث: {dt4(now):%d %b %H:%M} (UTC+4) · مرّ {(now-T0)/3600:.1f} ساعة على الإشارة (25 Nov 21:00)\n")
for sym, entry, stop, tps in SIGNALS:
    m = fetch(sym)
    idx = [i for i in range(len(m["time"])) if m["time"][i] >= T0]
    last = m["close"][-1]
    hit, st_t = {}, None
    for i in idx:
        if st_t is None and m["low"][i] <= stop:
            st_t = m["time"][i]
        for n, tp in enumerate(tps, 1):
            if n not in hit and m["high"][i] >= tp and (st_t is None or m["time"][i] < st_t):
                hit[n] = m["time"][i]
    mx = max(idx, key=lambda i: m["high"][i]); mn = min(idx, key=lambda i: m["low"][i])
    print(f"== {sym[:-5]} · دخول {entry} · وقف {stop} · السعر الآن {last} ({(last/entry-1)*100:+.2f}%)")
    for n, tp in enumerate(tps, 1):
        print(f"   TP{n} {tp} ({(tp/entry-1)*100:+.2f}%): " + (f"بُلغ {dt4(hit[n]):%d %b %H:%M} (بعد {(hit[n]-T0)/3600:.1f}س)" if n in hit else "لم يُبلغ"))
    print(f"   الوقف: " + (f"ضُرب {dt4(st_t):%d %b %H:%M} (بعد {(st_t-T0)/3600:.1f}س)" if st_t else "لم يُضرب"))
    print(f"   أعلى سعر منذ الدخول {m['high'][mx]:.6g} ({(m['high'][mx]/entry-1)*100:+.2f}%) {dt4(m['time'][mx]):%d %b %H:%M} · أدنى سعر {m['low'][mn]:.6g} ({(m['low'][mn]/entry-1)*100:+.2f}%) {dt4(m['time'][mn]):%d %b %H:%M}\n")
