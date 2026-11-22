"""Locate a radar alert without a date: find the 1H candle (close time HH:00 UTC+4) whose close matches, then follow up from the alert price."""
import sys
import time
from datetime import datetime, timedelta, timezone
import scanner as S

SYM, HOUR_CLOSE, HOUR, ALERT_PRICE, SECOND = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
EXP1, EXP4 = float(sys.argv[6]), float(sys.argv[7])      # radar: last-hour % and last-4-hours %     # SECOND = HH:MM:SS of the price quote
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)
now = int(time.time())


def kl(itv, step, start, end):
    out = {k: [] for k in ("time", "open", "high", "low", "close")}
    a = start
    while a < end:
        b = min(a + step * 1000, end)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{SYM}?interval={itv}&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            raise SystemExit("fetch failed")
        for k in out:
            out[k] += d[k]
        a = b
    idx = sorted(set(range(len(out["time"]))), key=lambda i: out["time"][i])
    seen, keep = set(), []
    for i in idx:
        if out["time"][i] not in seen:
            seen.add(out["time"][i]); keep.append(i)
    return {k: [out[k][i] for i in keep] for k in out}


h1 = kl("Min60", 3600, now - 120 * 86400, now)
cands = []
for i in range(1, len(h1["time"])):
    close_t = h1["time"][i] + 3600
    if dt(close_t).hour == HOUR % 24 and dt(close_t).minute == 0 and abs(h1["close"][i] / HOUR_CLOSE - 1) < 0.003:
        ret = (h1["close"][i] / h1["open"][i] - 1) * 100
        ret4 = (h1["close"][i] / h1["open"][i - 3] - 1) * 100 if i >= 3 else 0.0
        cands.append((close_t, h1["close"][i], ret, ret4))
print(f"{SYM}: شموع الساعة التي يغلق عند {HOUR}:00 (UTC+4) بسعر قريب من {HOUR_CLOSE}: {len(cands)}")
for ct, cl, ret, ret4 in cands:
    print(f"  {dt(ct):%d %b %Y %H:%M} إغلاق {cl} · حركة الساعة {ret:+.1f}% · 4 ساعات {ret4:+.1f}%")
if not cands:
    raise SystemExit("لا تطابق")
ct = min(cands, key=lambda c: abs(c[2] - EXP1) + abs(c[3] - EXP4))[0]
hh, mm, ss = (int(x) for x in SECOND.split(":"))
alert = dt(ct).replace(hour=hh, minute=mm, second=ss) - timedelta(hours=4)
a0 = int(alert.timestamp())
print(f"\\nالتنبيه المختار: {dt(a0):%d %b %Y %H:%M:%S} (UTC+4) · سعر التنبيه {ALERT_PRICE}")
d5 = kl("Min5", 300, a0 - 600, now)
bars = sorted({d5['time'][i]: (d5['time'][i], d5['high'][i], d5['low'][i], d5['close'][i]) for i in range(len(d5['time']))}.values())
after = [b for b in bars if b[0] + 300 > a0]
print(f"السعر الحالي {after[-1][3]} ({(after[-1][3]/ALERT_PRICE-1)*100:+.2f}% عن سعر التنبيه) · حتى {dt(now):%d %b %H:%M}\\n")
for nm, hrs in (("ساعتان", 2), ("6 ساعات", 6), ("24 ساعة", 24), ("3 أيام", 72), ("حتى الآن", 10_000)):
    w = [b for b in after if b[0] <= a0 + hrs * 3600]
    if not w:
        continue
    pk = max(w, key=lambda b: b[1])
    dd = min(b[2] for b in w if b[0] <= pk[0])
    print(f"خلال {nm}: أعلى سعر {pk[1]:.6g} عند {dt(pk[0]):%d %b %H:%M} = {(pk[1]/ALERT_PRICE-1)*100:+.2f}% · أدنى سعر قبل القمة {dd:.6g} ({(dd/ALERT_PRICE-1)*100:+.2f}%) · أدنى سعر في النافذة {min(b[2] for b in w):.6g} ({(min(b[2] for b in w)/ALERT_PRICE-1)*100:+.2f}%)")
f10 = next((b for b in after if b[1] >= ALERT_PRICE * 1.10), None)
print("\\nأول وصول إلى +10%:", f"{dt(f10[0]):%d %b %H:%M} (بعد {(f10[0]-a0)/3600:.1f} ساعة)" if f10 else "لم يصل")
m4 = next((b for b in after if b[2] <= ALERT_PRICE * 0.96), None)
print("أول نزول إلى -4%:", f"{dt(m4[0]):%d %b %H:%M} (بعد {(m4[0]-a0)/3600:.1f} ساعة)" if m4 else "لم ينزل")
