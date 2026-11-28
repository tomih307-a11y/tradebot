"""PORTAL: two posts at entry 0.01557 (1H template stop -5%; 4H template stop -10%). Claimed TP1 0.01604 at 08:58 PM Riyadh (date missing). Spot + futures 1m, last 72h:
hourly closes within 1.2% of entry, rising crossings of TP1, and the path after the best start candidate."""
import time
from datetime import datetime, timedelta, timezone
import scanner as S

now = int(time.time())
u4 = lambda x: (datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)).strftime("%d %H:%M")
E, STOP, TP1 = 0.01557, 0.01479, 0.01604


def fut(a, b):
    out = {k: [] for k in ("time", "high", "low", "close")}
    while a < b:
        e = min(a + 60000, b)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/PORTAL_USDT?interval=Min1&start={a}&end={e}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d:
            for k in out:
                out[k] += d[k]
        a = e
    return out


def spot(a, b):
    out = {k: [] for k in ("time", "high", "low", "close")}
    while a < b:
        e = min(a + 60000, b)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://api.mexc.com/api/v3/klines?symbol=PORTALUSDT&interval=1m&startTime={a*1000}&endTime={e*1000}&limit=1000"); break
            except Exception:
                time.sleep(1.5 + r)
        for x in d or []:
            out["time"].append(int(x[0]) // 1000); out["high"].append(float(x[2])); out["low"].append(float(x[3])); out["close"].append(float(x[4]))
        a = e
    return out


print(f"الآن: {u4(now)} (UTC+4) · دخول {E} · وقف {STOP} · TP1 {TP1}\n")
for kind, fn in (("سبوت", spot), ("عقود", fut)):
    m = fn(now - 72 * 3600, now)
    T = m["time"]
    if len(T) < 100:
        print(f"{kind}: بيانات غير كافية\n"); continue
    hc = {}
    for i, t in enumerate(T):
        hc[t // 3600 * 3600] = i
    near = [(h + 3600, m["close"][i]) for h, i in sorted(hc.items()) if abs(m["close"][i] / E - 1) <= 0.012]
    print(f"{kind}: إغلاقات ساعة قريبة من الدخول (±1.2%): " + ", ".join(f"{u4(t)}={p:.5f}" for t, p in near[-10:]))
    # rising crossings of TP1
    cr, below_since = [], None
    for i in range(len(T)):
        if m["high"][i] >= TP1:
            if below_since is None or T[i] - below_since > 3600:
                cr.append(T[i])
            below_since = T[i]
    print(f"  لحظات عبور TP1 صعوداً: " + ", ".join(u4(x) for x in cr[-8:]))
    print(f"  أعلى سعر خلال 72 ساعة {max(m['high']):.5f} ({u4(T[m['high'].index(max(m['high']))])}) · أدنى {min(m['low']):.5f} ({u4(T[m['low'].index(min(m['low']))])}) · آخر سعر {m['close'][-1]:.5f} ({(m['close'][-1]/E-1)*100:+.1f}% عن الدخول)")
    # path after the last close near entry that precedes the last TP1 crossing
    if near and cr:
        starts = [t for t, _ in near if t <= cr[-1]]
        if starts:
            t0 = starts[-1]
            post = [i for i in range(len(T)) if T[i] >= t0]
            lo = min(m["low"][i] for i in post); hi = max(m["high"][i] for i in post)
            wick = next((T[i] for i in post if m["low"][i] <= STOP), None)
            print(f"  أقرب بداية قبل آخر عبور: {u4(t0)} · أعلى بعدها {(hi/E-1)*100:+.1f}% · أدنى {(lo/E-1)*100:+.1f}% · وقف فتيل: {u4(wick) if wick else 'لا'}")
    print()
