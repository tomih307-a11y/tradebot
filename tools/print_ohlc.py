"""Raw MEXC REST candles for BTC_USDT 1h around the biggest PMO differences (for comparison with TradingView)."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tradebot.config import load_config                  # noqa: E402
from tradebot.mexc.client import MexcClient              # noqa: E402

cfg = load_config("config.toml")
c = MexcClient(cfg.mexc)
now = int(time.time())
L = ["# شموع MEXC الخام (BTC_USDT، ساعة، وقت الفتح UTC)", "", "| وقت الفتح | O | H | L | C | الحجم |", "|---|---|---|---|---|---|"]
d = c.klines("BTC_USDT", "Min60", now - 40 * 3600, now)
for i, t in enumerate(d["time"]):
    ts = datetime.fromtimestamp(int(t), timezone.utc)
    if ts.strftime("%m-%d") in ("10-11", "10-12") and (t + 3600) <= now:
        L.append(f"| {ts:%m-%d %H:%M} | {d['open'][i]} | {d['high'][i]} | {d['low'][i]} | {d['close'][i]} | {d['vol'][i]} |")
Path("reports/tv").mkdir(parents=True, exist_ok=True)
Path("reports/tv/ohlc.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
