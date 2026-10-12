"""Compare TradingView's PMO table values (screenshot, 2026-10-12) with PMO computed from MEXC REST candles,
using both the `close` series and the `realClose` series that MEXC returns."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.config import load_config                  # noqa: E402
from tradebot.indicators import pmo                      # noqa: E402
from tradebot.mexc.client import TF_SECONDS, MexcClient  # noqa: E402

# (UTC open "MM-DD HH:MM") -> (PMO, SIGNAL) read from the TradingView table
TV = {
    "Min15": {
        "10-12 06:30": (0.0955, 0.0931), "10-12 06:15": (0.0995, 0.0926), "10-12 06:00": (0.1102, 0.0910),
        "10-12 05:45": (0.1125, 0.0867), "10-12 05:30": (0.1078, 0.0810), "10-12 05:15": (0.1069, 0.0751),
        "10-12 05:00": (0.0913, 0.0680), "10-12 04:45": (0.0662, 0.0628), "10-12 04:30": (0.0424, 0.0620),
        "10-12 04:15": (0.0302, 0.0664), "10-12 04:00": (0.0398, 0.0744), "10-12 03:45": (0.0537, 0.0821),
        "10-12 03:30": (0.0662, 0.0885), "10-12 03:15": (0.0797, 0.0934), "10-12 03:00": (0.0909, 0.0965),
        "10-12 02:45": (0.0969, 0.0977), "10-12 02:30": (0.0959, 0.0979)},
    "Min60": {
        "10-12 05:00": (0.3080, 0.2750), "10-12 04:00": (0.2921, 0.2676), "10-12 03:00": (0.2874, 0.2622),
        "10-12 02:00": (0.2913, 0.2566), "10-12 01:00": (0.2883, 0.2489), "10-12 00:00": (0.2838, 0.2401),
        "10-11 23:00": (0.2702, 0.2304), "10-11 22:00": (0.2645, 0.2215), "10-11 21:00": (0.2664, 0.2120),
        "10-11 20:00": (0.2699, 0.1999), "10-11 19:00": (0.2620, 0.1844), "10-11 18:00": (0.2493, 0.1671),
        "10-11 17:00": (0.2277, 0.1489), "10-11 16:00": (0.1864, 0.1314), "10-11 15:00": (0.1387, 0.1191),
        "10-11 14:00": (0.1109, 0.1148), "10-11 13:00": (0.1123, 0.1157)},
}


def fetch(client, sym, tf, bars):
    step = TF_SECONDS[tf]
    now = time.time()
    end = int(now // step * step)
    parts = []
    cur = end - bars * step
    while cur <= end:
        stop = min(cur + 1999 * step, end)
        d = client.klines(sym, tf, cur, stop)
        parts.append(d)
        cur = stop + step
    out = {}
    for key in ("time", "open", "high", "low", "close", "realOpen", "realHigh", "realLow", "realClose"):
        out[key] = np.concatenate([np.asarray(p.get(key, []), dtype=float) for p in parts])
    order = np.argsort(out["time"], kind="stable")
    keep = np.concatenate(([True], np.diff(out["time"][order]) > 0))
    idx = order[keep]
    out = {k: v[idx] for k, v in out.items()}
    m = (out["time"] + step) <= now                      # closed candles only
    return {k: v[m] for k, v in out.items()}


def main():
    cfg = load_config("config.toml")
    ind = cfg.pmo_zero_cross.indicator
    client = MexcClient(cfg.mexc)
    L = ["# مقارنة TradingView مع قيم بايثون (BTC_USDT)", ""]
    for tf, step in (("Min15", 900), ("Min60", 3600)):
        k = fetch(client, "BTC_USDT", tf, 1500)
        L.append(f"## {tf}")
        L.append(f"- الفرق بين close و realClose: أقصى {np.max(np.abs(k['close'] - k['realClose'])):.4f} · متوسط {np.mean(np.abs(k['close'] - k['realClose'])):.4f} (آخر 200 شمعة: "
                 f"{np.mean(np.abs(k['close'][-200:] - k['realClose'][-200:])):.4f})")
        res = {}
        for name in ("close", "realClose"):
            p, s, _ = pmo(k[name], ind.roc_smooth_len, ind.roc_scale, ind.pmo_smooth_len, ind.signal_len)
            res[name] = (p, s)
        L += ["", "| وقت الفتح | TV PMO | بايثون(close) | فرق | بايثون(realClose) | فرق | TV SIGNAL | بايثون(close) | فرق | بايثون(realClose) | فرق |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        worst = {"close": 0.0, "realClose": 0.0}
        sums = {"close": [], "realClose": []}
        for i, t in enumerate(k["time"]):
            ts = datetime.fromtimestamp(int(t), timezone.utc).strftime("%m-%d %H:%M")
            if ts in TV[tf]:
                tp, tsg = TV[tf][ts]
                row = [ts, f"{tp:.4f}"]
                for name in ("close", "realClose"):
                    p = res[name][0][i]
                    row += [f"{p:.4f}", f"{abs(p - tp):.4f}"]
                    worst[name] = max(worst[name], abs(p - tp), abs(res[name][1][i] - tsg))
                    sums[name] += [abs(p - tp), abs(res[name][1][i] - tsg)]
                row.append(f"{tsg:.4f}")
                for name in ("close", "realClose"):
                    s_ = res[name][1][i]
                    row += [f"{s_:.4f}", f"{abs(s_ - tsg):.4f}"]
                L.append("| " + " | ".join(row) + " |")
        L.append("")
        for name in ("close", "realClose"):
            L.append(f"- سلسلة **{name}**: أقصى فرق مطلق {worst[name]:.4f} · متوسط الفرق {np.mean(sums[name]):.4f}")
        L.append("")
    out = Path("reports/tv")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tv_diag.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
