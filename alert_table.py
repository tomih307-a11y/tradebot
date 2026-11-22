"""For each radar alert: alert price vs current price, and highest / lowest price AFTER the alert using 1-minute candles (the candle containing the alert time is excluded). Radar time is UTC+3; output in UTC+4."""
import time
from datetime import datetime, timedelta, timezone
import scanner as S

ALERTS = [("ARIA_USDT", "2026-10-26 18:02:47", 0.07046), ("JCT_USDT", "2026-10-25 00:03:10", 0.007159), ("Q_USDT", "2026-10-25 19:02:56", 0.068146), ("AGT_USDT", "2026-10-26 09:02:48", 0.05751)]
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)
now = int(time.time())
print(f"{'العملة':10s} {'وقت التنبيه (UTC+4)':22s} {'سعر التنبيه':>11s} {'السعر الآن':>11s} {'الفرق':>9s} | {'أعلى سعر بعد التنبيه':>26s} | {'أدنى سعر بعد التنبيه':>26s}")
for sym, ts, price in ALERTS:
    a0 = int((datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) - timedelta(hours=4)).timestamp())
    start = (a0 // 60 + 1) * 60                                    # first full minute after the alert
    hi, lo, last = None, None, None
    a = start
    while a < now:
        b = min(a + 60 * 1000, now)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min1&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            raise SystemExit("fetch failed " + sym)
        for i in range(len(d["time"])):
            if d["time"][i] < start:
                continue
            if hi is None or d["high"][i] > hi[0]:
                hi = (d["high"][i], d["time"][i])
            if lo is None or d["low"][i] < lo[0]:
                lo = (d["low"][i], d["time"][i])
            last = d["close"][i]
        a = b
    print(f"{sym[:-5]:10s} {dt(a0):%d %b %H:%M:%S}   {price:11.6g} {last:11.6g} {(last/price-1)*100:+8.2f}% | {hi[0]:.6g} ({(hi[0]/price-1)*100:+.2f}%) {dt(hi[1]):%d %b %H:%M} | {lo[0]:.6g} ({(lo[0]/price-1)*100:+.2f}%) {dt(lo[1]):%d %b %H:%M}")
