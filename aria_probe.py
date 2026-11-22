import time
from datetime import datetime, timedelta, timezone
import scanner as S
now = int(time.time())
print("now UTC+4:", (datetime.fromtimestamp(now, timezone.utc) + timedelta(hours=4)).strftime("%d %b %H:%M"))
out = {k: [] for k in ("time", "open", "close")}
a = now - 150 * 86400
while a < now:
    b = min(a + 3600 * 1000, now)
    d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/ARIA_USDT?interval=Min60&start={a}&end={b}")["data"]
    for k in out:
        out[k] += d[k]
    a = b
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)
print("candles", len(out["time"]), "last candle open", dt(out["time"][-1]).strftime("%d %b %H:%M"), "close", out["close"][-1])
for i in range(len(out["time"])):
    ret = (out["close"][i] / out["open"][i] - 1) * 100
    if ret <= -15:
        print("hourly drop", round(ret, 1), "hour close", dt(out["time"][i] + 3600).strftime("%d %b %H:%M"), "close price", out["close"][i], "open", out["open"][i])
print("--- last 8 hourly candles")
for i in range(len(out["time"]) - 8, len(out["time"])):
    print(dt(out["time"][i]).strftime("%d %H:%M"), "open", out["open"][i], "close", out["close"][i])
