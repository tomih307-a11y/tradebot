"""Print the last N closed-candle PMO/SIGNAL values (UTC open time) for BTC_USDT on Min15 and Min60,
using exactly the code the backtest uses. Compare with the TradingView table (tools/pmo_table.pine)."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.config import load_config                          # noqa: E402
from tradebot.data.klines import drop_unclosed, fetch_range      # noqa: E402
from tradebot.indicators import pmo                              # noqa: E402
from tradebot.mexc.client import TF_SECONDS, MexcClient          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC_USDT")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--bars", type=int, default=1500)
    ap.add_argument("--out", default="reports/tv")
    a = ap.parse_args()
    cfg = load_config("config.toml")
    ind = cfg.pmo_zero_cross.indicator
    client = MexcClient(cfg.mexc)
    now = time.time()
    lines = [f"# قيم PMO/SIGNAL من كود الباكتست — {a.symbol}", f"(وقت التشغيل: {datetime.fromtimestamp(now, timezone.utc):%Y-%m-%d %H:%M} UTC؛ الشموع المغلقة فقط؛ الوقت = وقت فتح الشمعة UTC)", ""]
    for tf in ("Min15", "Min60", "Hour4"):
        step = TF_SECONDS[tf]
        end = int(now // step * step)
        k = drop_unclosed(fetch_range(client, a.symbol, tf, end - a.bars * step, end), tf, now)
        p, s, _ = pmo(k["close"], ind.roc_smooth_len, ind.roc_scale, ind.pmo_smooth_len, ind.signal_len)
        lines += [f"## {tf} (آخر {a.n} شمعة مغلقة، الأحدث أولاً)", "| وقت الفتح UTC | PMO | SIGNAL |", "|---|---|---|"]
        for i in range(len(p) - 1, len(p) - 1 - a.n, -1):
            t = datetime.fromtimestamp(int(k["time"][i]), timezone.utc).strftime("%m-%d %H:%M")
            lines.append(f"| {t} | {p[i]:.4f} | {s[i]:.4f} |")
        lines.append("")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pmo_values.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
