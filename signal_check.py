"""Follow-up of channel signals without dates. For each signal: find the recent 1H candles whose close is within 0.5% of the entry zone, then follow from the next minute (1m candles):
which TPs were reached (and when), whether the stop was hit, highest price and %, lowest price, current price. Times UTC+4."""
import time
from datetime import datetime, timedelta, timezone
import scanner as S

SIGNALS = [
    ("MOODENG_USDT", 0.02792, 0.02652, [0.02876, 0.02971, 0.03066, 0.03161, 0.03255, 0.0335], "مغلقة"),
    ("AT_USDT", 0.0544, 0.0517, [0.056, 0.0579, 0.0597, 0.0616, 0.0634, 0.0653], "نشطة"),
]
now = int(time.time())
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)


def fetch(sym, itv, step, a, b_end):
    out = {k: [] for k in ("time", "high", "low", "close")}
    while a < b_end:
        b = min(a + step * 1000, b_end)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            raise SystemExit("fetch failed " + sym)
        for k in out:
            out[k] += d[k]
        a = b
    seen, keep = set(), []
    for i, t in enumerate(out["time"]):
        if t not in seen:
            seen.add(t); keep.append(i)
    return {k: [out[k][i] for i in keep] for k in out}


for sym, entry, stop, tps, status in SIGNALS:
    h1 = fetch(sym, "Min60", 3600, now - 21 * 86400, now)
    m1 = fetch(sym, "Min1", 60, now - 21 * 86400, now)
    last = m1["close"][-1]
    print(f"\n==== {sym[:-5]} · الدخول {entry} · الوقف {stop} ({(stop/entry-1)*100:+.2f}%) · الحالة المكتوبة: {status} · السعر الآن {last} ({(last/entry-1)*100:+.2f}% عن الدخول)")
    cands = []
    for i in range(len(h1["time"])):
        if abs(h1["close"][i] / entry - 1) <= 0.005:
            cands.append(h1["time"][i] + 3600)
    # keep one candidate per 6h cluster
    pick, lastc = [], 0
    for t in cands:
        if t - lastc >= 6 * 3600:
            pick.append(t); lastc = t
    print(f"  شموع ساعة أغلقت قرب سعر الدخول (±0.5%) خلال 21 يوماً: {len(cands)} · لحظات متميزة: {len(pick)}")
    rows = []
    for t0 in pick:
        idx = [i for i in range(len(m1["time"])) if m1["time"][i] >= t0]
        if not idx:
            continue
        hit = {}
        st_t = None
        mx = (0, None)
        for i in idx:
            if st_t is None and m1["low"][i] <= stop:
                st_t = m1["time"][i]
            for n, tp in enumerate(tps, 1):
                if n not in hit and m1["high"][i] >= tp and st_t is None:
                    hit[n] = m1["time"][i]
            if m1["high"][i] > mx[0]:
                mx = (m1["high"][i], m1["time"][i])
        lo = min(m1["low"][i] for i in idx)
        rows.append((t0, hit, st_t, mx, lo))
        tptxt = ", ".join(f"TP{n} بعد {(t-t0)/3600:.1f}س" for n, t in sorted(hit.items())) or "لا هدف"
        sttxt = f"ضُرب الوقف بعد {(st_t-t0)/3600:.1f}س" if st_t else "لم يُضرب الوقف"
        print(f"  • {dt4(t0):%d %b %H:%M}: {tptxt} | {sttxt} | أعلى سعر {mx[0]:.6g} ({(mx[0]/entry-1)*100:+.1f}%) {dt4(mx[1]):%d %b %H:%M} | أدنى {lo:.6g} ({(lo/entry-1)*100:+.1f}%)")
