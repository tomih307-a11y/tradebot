"""Live scan: contracts where, on the 15m chart, PMO has JUST crossed above its signal line with both lines above zero (long watchlist).
Each candidate is flagged for: 1H PMO > 0, 4H PMO > 0, price inside a still-valid bullish 4H order block (same code as the bot).
Closed candles only. Output: <out>/scan.json and <out>/scan.md. This is a candidate LIST, not advice: the walk-forward tests found no edge.
Usage: python tools/scan_now.py --out reports/scan --top 400 --fresh 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.config import load_config                                        # noqa: E402
from tradebot.data.klines import drop_unclosed, fetch_range                      # noqa: E402
from tradebot.indicators import pmo                                             # noqa: E402
from tradebot.mexc.client import TF_SECONDS, MexcClient                         # noqa: E402
from tradebot.strategies.multi_agent.rules import RulesAgent, cross_above_zero  # noqa: E402


def masked_pmo(close, ind):
    p, sg, _ = pmo(close, ind.roc_smooth_len, ind.roc_scale, ind.pmo_smooth_len, ind.signal_len)
    p, sg = p.copy(), sg.copy()
    p[: int(ind.warmup_bars)] = np.nan
    sg[: int(ind.warmup_bars)] = np.nan
    return p, sg


def recent(client, sym, tf, bars, now):
    step = TF_SECONDS[tf]
    return drop_unclosed(fetch_range(client, sym, tf, int(now // step * step) - bars * step, int(now)), tf, now)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/scan")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--top", type=int, default=400)
    ap.add_argument("--fresh", type=int, default=4, help="cross must be within the last N closed 15m bars (4 = last hour)")
    ap.add_argument("--bars", type=int, default=720)
    a = ap.parse_args()
    cfg = load_config(a.config)
    client = MexcClient(cfg.mexc)
    ind = cfg.pmo_zero_cross.indicator
    now = int(time.time())
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    tick = tick[: a.top]
    live = {t["symbol"]: float(t["lastPrice"]) for t in tick}
    turn = {t["symbol"]: float(t.get("amount24") or 0) for t in tick}
    agent = RulesAgent(cfg)
    rows, scanned = [], 0
    for rank, t in enumerate(tick, 1):
        sym = t["symbol"]
        try:
            k15 = recent(client, sym, "Min15", a.bars, now)
            if len(k15["close"]) < int(ind.warmup_bars) + 60:
                continue
            scanned += 1
            p, sg = masked_pmo(k15["close"], ind)
            c = cross_above_zero(p, sg, 1, a.fresh)
            if c is None:
                continue
            n = len(p)
            row = {"symbol": sym, "rank": rank, "turnover_usdt": turn[sym], "price": live[sym], "age_bars": n - 1 - c,
                   "cross_close": datetime.fromtimestamp(int(k15["time"][c]) + 900, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                   "since_cross_pct": float((k15["close"][-1] / k15["close"][c] - 1) * 100), "pmo15": float(p[-1]), "sig15": float(sg[-1])}
            k1, k4 = recent(client, sym, "Min60", a.bars, now), recent(client, sym, "Hour4", a.bars, now)
            for name, k in (("pmo1h", k1), ("pmo4h", k4)):
                ph, _ = masked_pmo(k["close"], ind) if len(k["close"]) > int(ind.warmup_bars) else (np.array([np.nan]), None)
                row[name] = None if not np.isfinite(ph[-1]) else float(ph[-1])
            ob = agent.cond_price_in_ob_4h({"k4h": k4, "price": live[sym]}) if len(k4["close"]) >= int(cfg.agents.rules.min_bars) else None
            row["in_bull_ob_4h"] = bool(ob and ob["side"] == "long")
            row["ob_zone"] = list(ob["zone"]) if row["in_bull_ob_4h"] else None
            row["h1_ok"] = bool(row["pmo1h"] is not None and row["pmo1h"] > 0)
            row["h4_ok"] = bool(row["pmo4h"] is not None and row["pmo4h"] > 0)
            rows.append(row)
        except Exception as e:
            print(f"  {sym}: {e}", flush=True)
    for r in rows:
        r["score"] = int(r["h1_ok"]) + int(r["h4_ok"]) + int(r["in_bull_ob_4h"])
    rows.sort(key=lambda r: (-r["score"], r["age_bars"], r["rank"]))
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "scan.json").write_text(json.dumps({"generated": stamp, "scanned": scanned, "fresh_bars": a.fresh, "rows": rows}, ensure_ascii=False, indent=1))
    L = [f"# مسح الآن: تقاطع PMO للأعلى فوق الصفر على 15د ({stamp})", "",
         f"- عقود مفحوصة: {scanned} (أعلى {a.top} سيولة) · مرشحون: {len(rows)} · الحداثة: التقاطع خلال آخر {a.fresh} شموع 15د مغلقة", "",
         "| العقد | عمر التقاطع (شمعة) | السعر | منذ التقاطع | PMO 15د | PMO 1س | PMO 4س | داخل OB صاعد 4س | ترتيب السيولة |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda v: "–" if v is None else f"{v:+.2f}"
        L.append(f"| {r['symbol']} | {r['age_bars']} | {r['price']:.6g} | {r['since_cross_pct']:+.2f}% | {r['pmo15']:+.2f} | {f(r['pmo1h'])} | {f(r['pmo4h'])} | "
                 f"{'نعم ' + str([round(z, 6) for z in r['ob_zone']]) if r['in_bull_ob_4h'] else 'لا'} | {r['rank']} |")
    (Path(a.out) / "scan.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
