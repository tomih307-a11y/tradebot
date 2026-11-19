"""Forward log of the quality scanner (rules: docs/QUALITY_SCAN.md, docs/FORWARD_ALERTS.md). Read-only.
Usage: python -m tradebot.strategies.mtf_entry.quality_log DIR   (DIR holds quality.csv; quality_summary.md is written there)"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from .ob_entry import hourly
from .quality_scan import COOL, components, setup_risk, trade
from .trail_exit import boot99

COLS = ["bar_time", "sym", "dir", "tier", "px", "dist", "comps", "turn24", "logged_at", "resolved", "R"]
RESOLVE_AFTER = 49 * 3600


def load_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def add_new(rows: list[dict], new: list[dict]) -> list[dict]:
    """Append signals not logged yet (same bar/sym/dir/tier) and respecting the 8-hour per-(sym,dir,tier) cooldown (BASE rows exempt)."""
    keys = {(r["bar_time"], r["sym"], r["dir"], r["tier"]) for r in rows}
    out = list(rows)
    for r in sorted(new, key=lambda x: float(x["bar_time"])):
        k = (r["bar_time"], r["sym"], r["dir"], r["tier"])
        if k in keys:
            continue
        if r["tier"] != "BASE" and any(x["sym"] == r["sym"] and x["dir"] == r["dir"] and x["tier"] == r["tier"]
                                       and abs(float(x["bar_time"]) - float(r["bar_time"])) < COOL * 3600 for x in out):
            continue
        out.append(r); keys.add(k)
    return out


def summarize(rows: list[dict]) -> str:
    done = [r for r in rows if r["resolved"] == "1"]
    lines = [f"# سجل ماسح الجودة الأمامي · محسوم {len(done)} من {len(rows)}", ""]
    for d in ("L", "S"):
        base = [float(r["R"]) for r in done if r["dir"] == d and r["tier"] == "BASE"]
        lines.append(f"{'لونج' if d == 'L' else 'شورت'} · أساس: " + (f"{np.mean(base):+.3f} ({len(base)})" if base else "-"))
        for t in ("A", "B"):
            sel = [r for r in done if r["dir"] == d and r["tier"] == t]
            if sel:
                R = np.array([float(r["R"]) for r in sel])
                lo, hi = boot99(np.array([int(float(r["bar_time"]) // 86400) for r in sel]), R) if len(sel) >= 5 else (float("nan"), float("nan"))
                lines.append(f"  {t}: {len(sel)} إشارة · متوسط R {R.mean():+.3f} · حد 99% [{lo:+.3f}, {hi:+.3f}] · رابحة {(R > 0).mean() * 100:.0f}%")
            else:
                lines.append(f"  {t}: لا إشارات محسومة بعد")
    lines.append(f"آخر تحديث {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC")
    return "\n".join(lines)


def main():
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import MexcClient

    from .paper import NON_CRYPTO

    d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
    path = d / "quality.csv"
    rows = load_rows(path)
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 1_000_000]
    pending = {}
    for r in rows:
        if r["resolved"] != "1" and now - float(r["bar_time"]) >= RESOLVE_AFTER:
            pending.setdefault((r["sym"], r["dir"]), []).append(r)
    new = []
    for tk in tick:
        sym = tk["symbol"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - 22 * 96 * 900, now), "Min15", now)
            t, o15, h15, l15, c15 = (np.asarray(k[x], float) for x in ("time", "open", "high", "low", "close"))
            if len(c15) < 1200:
                continue
            th0, o0, h0, l0, c0 = hourly(t, o15, h15, l15, c15)
            for dr, (o, h, l, c) in (("L", (o0, h0, l0, c0)), ("S", (1 / o0, 1 / l0, 1 / h0, 1 / c0))):
                comp, ob_lo, atr = components(th0, o, h, l, c)
                score = comp.sum(axis=1)
                n = len(c)
                for i in range(n - 3, n):
                    if np.isnan(atr[i]):
                        continue
                    tier = ("A" if score[i] >= 6 else ("B" if score[i] == 5 else None)) if comp[i, 2] else None
                    dist = setup_risk(c[i], ob_lo[i], atr[i]) if tier else atr[i]
                    for tr_ in ([tier] if tier else []) + (["BASE"] if (int(th0[i] // 3600) % 7 == 0) else []):
                        new.append(dict(bar_time=int(th0[i]), sym=sym, dir=dr, tier=tr_, px=round(float(c[i]), 10), dist=round(float(dist if tr_ != "BASE" else atr[i]), 10),
                                        comps="".join("1" if v else "0" for v in comp[i]), turn24=round(float(tk.get("amount24") or 0)), logged_at=now, resolved="", R=""))
                for r in pending.get((sym, dr), []):
                    idx = np.where(th0 == float(r["bar_time"]))[0]
                    if len(idx):
                        rr = trade(o, h, l, c, int(idx[0]), float(r["dist"]))
                        if rr is not None:
                            r.update(resolved="1", R=round(rr, 4))
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
    rows = add_new(rows, [{k: str(v) for k, v in r.items()} for r in new])
    save_rows(path, rows)
    (d / "quality_summary.md").write_text(summarize(rows) + "\n", encoding="utf-8")
    print(summarize(rows), f"· جديد هذه الجولة: {len(new)}")


if __name__ == "__main__":
    main()
