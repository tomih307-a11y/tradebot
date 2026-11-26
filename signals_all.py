"""Status of all channel signals so far (Gulf Digital Elite): AT, MOODENG (both 25 Nov 21:00 UTC+4), EVAA (25 Nov 22:00 UTC+4). Entry = hourly close at the signal time. 1m candles. Times UTC+4."""
import time
from datetime import datetime, timedelta, timezone
import scanner as S

U = lambda y, mo, d, h, mi=0: int((datetime(y, mo, d, h, mi, tzinfo=timezone.utc) - timedelta(hours=4)).timestamp())
SIGNALS = [
    ("AT_USDT", U(2026, 11, 25, 21), 0.0544, 0.0517, [0.056, 0.0579, 0.0597, 0.0616, 0.0634, 0.0653]),
    ("MOODENG_USDT", U(2026, 11, 25, 14), 0.02792, 0.02652, [0.02876, 0.02971, 0.03066, 0.03161, 0.03255, 0.0335]),
    ("EVAA_USDT", U(2026, 11, 25, 22), 2.785, 2.646, [2.869, 2.963, 3.058, 3.153, 3.247, 3.342]),
    ("ELSA_USDT", U(2026, 11, 26, 13), 0.09577, 0.09098, [0.09864, 0.1019, 0.10516, 0.10841, 0.11167, 0.11492]),
]
now = int(time.time())
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)


def fetch(sym, t0):
    out = {k: [] for k in ("time", "high", "low", "close")}
    a = t0 - 60
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


print(f"التحديث: {dt4(now):%d %b %H:%M} (UTC+4)\n")
def find_t0(sym, entry):
    """Signal time unknown: the last 10 closed 1H candles; pick the one whose close is nearest to the entry price."""
    h = int(now // 3600) * 3600
    d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min60&start={h-11*3600}&end={now}")["data"]
    c = [(abs(d["close"][i] / entry - 1), d["time"][i]) for i in range(len(d["time"])) if d["time"][i] + 3600 <= now]
    return min(c)[1] + 3600 if c else h


for sym, t0, entry, stop, tps in SIGNALS:
    if t0 is None:
        t0 = find_t0(sym, entry)
    m = fetch(sym, t0)
    idx = [i for i in range(len(m["time"])) if m["time"][i] >= t0]
    if not idx:
        print(f"== {sym[:-5]}: لا شموع بعد"); continue
    last = m["close"][-1]
    hit, st_t = {}, None
    for i in idx:
        if st_t is None and m["low"][i] <= stop:
            st_t = m["time"][i]
        for n, tp in enumerate(tps, 1):
            if n not in hit and m["high"][i] >= tp and (st_t is None or m["time"][i] < st_t):
                hit[n] = m["time"][i]
    mx = max(idx, key=lambda i: m["high"][i]); mn = min(idx, key=lambda i: m["low"][i])
    print(f"== {sym[:-5]} · إشارة {dt4(t0):%d %b %H:%M} (مرّ {(now-t0)/3600:.1f}س) · دخول {entry} · وقف {stop} · الآن {last} ({(last/entry-1)*100:+.2f}%)")
    print("   الأهداف: " + " | ".join(f"TP{n} {'✔ ' + dt4(hit[n]).strftime('%H:%M') + ' (' + str(round((hit[n]-t0)/60)) + 'د)' if n in hit else '✘'}" for n in range(1, 7)))
    print("   الوقف: " + (f"ضُرب {dt4(st_t):%d %b %H:%M} (بعد {(st_t-t0)/60:.0f} دقيقة)" if st_t else "لم يُضرب"))
    print(f"   أعلى سعر {m['high'][mx]:.6g} ({(m['high'][mx]/entry-1)*100:+.2f}%) {dt4(m['time'][mx]):%d %H:%M} · أدنى {m['low'][mn]:.6g} ({(m['low'][mn]/entry-1)*100:+.2f}%) {dt4(m['time'][mn]):%d %H:%M}\n")
