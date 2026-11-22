"""Radar alert follow-up: from the alert price at the alert time, farthest price reached (highest high), when, the profit %, and the worst dip before it. Times are UTC+4."""
import sys
import time
from datetime import datetime, timedelta, timezone
import scanner as S

SYM, ALERT_PRICE = "JCT_USDT", 0.007159
ALERT = datetime(2026, 10, 24, 23, 3, 10, tzinfo=timezone.utc) - timedelta(hours=3)      # radar time is UTC+3
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)
now = int(time.time())
a0 = int(ALERT.timestamp())


def kl(itv, step, start, end):
    out = {k: [] for k in ("time", "high", "low", "close")}
    a = start
    while a < end:
        b = min(a + step * 1000, end)
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{SYM}?interval={itv}&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r); d = None
        if d is None:
            raise SystemExit("fetch failed")
        for k in out:
            out[k] += d[k]
        a = b
    return out


d5 = kl("Min5", 300, a0 - 600, now)
bars = [(d5["time"][i], d5["high"][i], d5["low"][i], d5["close"][i]) for i in range(len(d5["time"]))]
bars = sorted({b[0]: b for b in bars}.values())
after = [b for b in bars if b[0] + 300 > a0]                       # candles overlapping/after the alert time
print(f"{SYM} · تنبيه {dt(a0):%d %b %H:%M:%S} (UTC+4) بسعر {ALERT_PRICE} · شموع 5 دقائق: {len(after)} حتى {dt(now):%d %b %H:%M}")
print(f"السعر الحالي {after[-1][3]} ({(after[-1][3]/ALERT_PRICE-1)*100:+.2f}% عن سعر التنبيه)\n")
for nm, hrs in (("ساعتان", 2), ("6 ساعات", 6), ("24 ساعة", 24), ("3 أيام", 72), ("7 أيام", 168), ("30 يوماً (حتى الآن)", 10_000)):
    w = [b for b in after if b[0] <= a0 + hrs * 3600]
    if not w:
        continue
    pk = max(w, key=lambda b: b[1])
    dd = min(b[2] for b in w if b[0] <= pk[0])                       # worst low before the peak
    lo_all = min(b[2] for b in w)
    print(f"خلال {nm}: أعلى سعر {pk[1]:.6g} عند {dt(pk[0]):%d %b %H:%M} = {(pk[1]/ALERT_PRICE-1)*100:+.2f}% · أدنى سعر قبل القمة {dd:.6g} ({(dd/ALERT_PRICE-1)*100:+.2f}%) · أدنى سعر في النافذة {lo_all:.6g} ({(lo_all/ALERT_PRICE-1)*100:+.2f}%)")
first10 = next((b for b in after if b[1] >= ALERT_PRICE * 1.10), None)
print("\nأول وصول إلى +10%:", f"{dt(first10[0]):%d %b %H:%M} (بعد {(first10[0]-a0)/3600:.1f} ساعة)" if first10 else "لم يصل خلال الفترة")
first_m4 = next((b for b in after if b[2] <= ALERT_PRICE * 0.96), None)
print("أول نزول إلى -4%:", f"{dt(first_m4[0]):%d %b %H:%M} (بعد {(first_m4[0]-a0)/3600:.1f} ساعة)" if first_m4 else "لم ينزل خلال الفترة")
