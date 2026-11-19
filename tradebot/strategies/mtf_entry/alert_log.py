"""Forward alert log (read-only). Appends new T1-T3 alerts with features, resolves old ones as 'clean winner' (docs/FORWARD_ALERTS.md).
Usage: python -m tradebot.strategies.mtf_entry.alert_log DIR   (DIR holds alerts.csv; summary.md is written there)"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from .channel_logic import atr14
from .early_signs import WARM, features
from .hunter_tiers import alerts, tier_of
from .immediate_move import AHEAD, alert_features, clean_label

COLS = ["bar_time", "sym", "tier", "px", "L", "P", "A", "U", "M", "turn24", "ret24", "atr1h", "logged_at", "resolved", "clean", "bars_to_5", "mae", "mfe24"]
RESOLVE_AFTER = (AHEAD + 1) * 900


def load_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def merge_new(rows: list[dict], new: list[dict]) -> list[dict]:
    """Append alerts not already logged (key: bar_time, sym, tier)."""
    seen = {(r["bar_time"], r["sym"], r["tier"]) for r in rows}
    return rows + [r for r in new if (r["bar_time"], r["sym"], r["tier"]) not in seen]


def summarize(rows: list[dict]) -> str:
    done = [r for r in rows if r["resolved"] == "1"]
    lines = [f"# سجل التنبيهات الأمامي · محسوم {len(done)} من {len(rows)}", ""]
    if not done:
        return "\n".join(lines + ["لا تنبيهات محسومة بعد."])
    def rate(sel):
        return f"{sum(r['clean'] == '1' for r in sel) / len(sel) * 100:.1f}% ({len(sel)})" if sel else "-"
    lines.append(f"فائز نظيف: الكل {rate(done)} · T1 {rate([r for r in done if r['tier'] == '0'])} · T2 {rate([r for r in done if r['tier'] == '1'])} · "
                 f"T3 {rate([r for r in done if r['tier'] == '2'])} · L {rate([r for r in done if r['L'] == '1'])} · T1+L {rate([r for r in done if r['tier'] == '0' and r['L'] == '1'])}")
    lines.append(f"آخر تحديث {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC · القواعد: docs/FORWARD_ALERTS.md")
    return "\n".join(lines)


def main():
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import MexcClient

    from .paper import NON_CRYPTO

    d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
    path = d / "alerts.csv"
    rows = load_rows(path)
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    kb = drop_unclosed(fetch_range(cl, "BTC_USDT", "Min15", int(now // 900 * 900) - 6 * 96 * 900, now), "Min15", now)
    tb, cb = np.asarray(kb["time"], float), np.asarray(kb["close"], float)
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 300_000]
    new, pending = [], {}
    for r in rows:
        if r["resolved"] != "1" and now - float(r["bar_time"]) - 900 >= RESOLVE_AFTER:
            pending.setdefault(r["sym"], []).append(r)
    for tk in tick:
        sym = tk["symbol"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - 12 * 86400, now), "Min15", now)
            o, h, l, c, v, amt, t = (np.asarray(k[x], float) for x in ("open", "high", "low", "close", "vol", "amount", "time"))
            n = len(c)
            if n < WARM + 20:
                continue
            f, _, valid = features(o, h, l, c, v, t, np.array([0.0]), np.array([0.0]), np.zeros(n))
            tier = tier_of({x: f[x] for x in ("F1", "F2", "F3")}); tier[:WARM] = -1; tier[~valid] = -1
            a15 = atr14(h, l, c)
            for i, kk in alerts(tier):
                if kk > 2 or i < n - 4 or np.isnan(a15[i]):
                    continue
                ft = alert_features(c, amt, a15, i)
                ib = int(np.searchsorted(tb, t[i], side="right")) - 1
                m = bool(ib >= 96 and cb[ib] / cb[ib - 96] - 1 >= 0)
                new.append(dict(bar_time=int(t[i]), sym=sym, tier=kk, px=c[i], L=int(ft["L"]), P=int(ft["P"]), A=int(ft["A"]), U=int(ft["U"]), M=int(m),
                                turn24=round(ft["turn"]), ret24=round(ft["ret24"], 4), atr1h=round(ft["atr1h"], 4), logged_at=now, resolved="", clean="", bars_to_5="", mae="", mfe24=""))
            for r in pending.get(sym, []):
                idx = np.where(t == float(r["bar_time"]))[0]
                if len(idx):
                    lab = clean_label(h, l, c, int(idx[0]))
                    if lab is not None:
                        r.update(resolved="1", clean=int(lab[0]), bars_to_5="" if np.isnan(lab[1]) else int(lab[1]), mae=round(lab[2], 2),
                                 mfe24=round((h[int(idx[0]) + 1:int(idx[0]) + 1 + AHEAD].max() / c[int(idx[0])] - 1) * 100, 2))
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
    rows = merge_new(rows, [{k: str(v) for k, v in r.items()} for r in new])
    save_rows(path, rows)
    (d / "summary.md").write_text(summarize(rows) + "\n", encoding="utf-8")
    print(summarize(rows), f"· جديد هذه الجولة: {len(new)}")


if __name__ == "__main__":
    main()
