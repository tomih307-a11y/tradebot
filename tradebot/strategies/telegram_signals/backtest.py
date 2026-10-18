"""Replay a channel's DATED history (Telegram Desktop 'Export chat history' -> result.json) against real MEXC candles.

For every parsed signal: entry = open of the first 1m candle after (message time + delay), skipped if price already
ran away; then the same paper simulation as live paper mode. Chronological portfolio rules: margin = margin_pct of the
CURRENT balance, at most max_concurrent open trades. Reports every signal that did / did not trade and why.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import paper
from .parser import parse_signal, parse_update


def fetch_candles(client, symbol: str, start: int, end: int) -> list[tuple]:
    """1m candles (ts, o, h, l, c) in [start, end]; empty list if MEXC no longer keeps them that far back."""
    out, cur = [], start
    while cur <= end:
        stop = min(cur + 1999 * 60, end)
        d = client.klines(symbol, "Min1", cur, stop)
        for i, t in enumerate(d.get("time", [])):
            out.append((int(t), float(d["open"][i]), float(d["high"][i]), float(d["low"][i]), float(d["close"][i])))
        cur = stop + 60
    return sorted(set(out))


def simulate_signal(sig, candles, cfg, delay_sec: int):
    """-> (status, Position|None). Unit-free: returns the position simulated with margin=1 USDT."""
    t = cfg.telegram_signals
    cfg_ts = {"fee_pct": t.paper_fee_pct, "slip_pct": t.paper_slippage_pct, "breakeven_after_tp1": bool(t.breakeven_after_tp1)}
    start = sig.ts + delay_sec
    nxt = [c for c in candles if c[0] >= start]
    if not nxt:
        return "no_candles", None
    first = nxt[0]
    s = 1 if sig.side == "long" else -1
    fill = first[1] * (1 + s * t.paper_slippage_pct / 100.0)
    if abs(fill - sig.entry) / sig.entry * 100 > float(t.max_entry_deviation_pct):
        return "price_ran_away", None
    if s * (fill - sig.tps[0]) >= 0 or s * (fill - sig.stop) <= 0:
        return "beyond_tp1_or_stop", None
    lev = sig.leverage if (t.leverage_mode == "signal" and sig.leverage) else int(t.fixed_leverage)
    pos = paper.open_position(sig, fill, 100.0, 1.0, lev, first[0], t.stop_mode, t.paper_fee_pct, t.maintenance_margin)  # margin = 1.0
    for c in nxt[1:]:
        paper.step(pos, c, cfg_ts)
        if pos.closed:
            break
    if not pos.closed:
        pos.exit_reason = "still_open"
    return "traded", pos


def run(cfg, client, messages: list[dict], delay_sec: int, out_dir: str, horizon_days: int = 14) -> dict:
    t = cfg.telegram_signals
    sigs = []
    for m in messages:
        sg = parse_signal(m["text"], m["id"], m["ts"])
        if sg and sg.ts:
            sigs.append(sg)
    rows, sims = [], []
    for sg in sigs:
        cand = fetch_candles(client, sg.symbol, sg.ts, sg.ts + horizon_days * 86400)
        status, pos = simulate_signal(sg, cand, cfg, delay_sec)
        sims.append((sg, status, pos))
    # chronological portfolio: balance compounding + concurrency cap
    balance = float(t.paper_balance_usdt)
    open_trades = []                                            # (exit_ts, pnl_per_unit_margin, margin)
    equity_curve, results = [], []
    for sg, status, pos in sorted(sims, key=lambda x: x[0].ts):
        entry_ts = pos.opened_ts if pos else sg.ts
        still = []
        for ex_ts, pnl_u, mg in open_trades:                    # settle trades that closed before this entry
            if ex_ts <= entry_ts:
                balance += pnl_u * mg
            else:
                still.append((ex_ts, pnl_u, mg))
        open_trades = still
        if status != "traded":
            results.append((sg, status, None, 0.0, balance))
            continue
        if len(open_trades) >= int(t.max_concurrent):
            results.append((sg, "max_concurrent_full", None, 0.0, balance))
            continue
        margin = balance * float(t.margin_pct_of_balance) / 100.0
        exit_ts = pos.exit_ts or (sg.ts + horizon_days * 86400)
        open_trades.append((exit_ts, pos.realized, margin))     # realized is per 1 USDT margin
        results.append((sg, "traded", pos, pos.realized * margin, balance))
    for ex_ts, pnl_u, mg in open_trades:
        balance += pnl_u * mg
    iso = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M") if ts else ""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traded = [r for r in results if r[1] == "traded"]
    wins = [r for r in traded if r[3] > 0]
    with open(out / "signals_backtest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signal_time_utc", "symbol", "side", "entry", "stop", "tp1", "leverage", "status", "exit_reason", "tp_hit", "pnl_usdt", "balance_before"])
        for sg, st, pos, pnl, bal in results:
            w.writerow([iso(sg.ts), sg.symbol, sg.side, sg.entry, sg.stop, sg.tps[0], sg.leverage, st,
                        pos.exit_reason if pos else "", pos.tp_hit if pos else "", f"{pnl:.4f}", f"{bal:.4f}"])
    reasons = {}
    for r in results:
        reasons[r[1]] = reasons.get(r[1], 0) + 1
    exits = {}
    for r in traded:
        exits[r[2].exit_reason] = exits.get(r[2].exit_reason, 0) + 1
    L = ["# اختبار توصيات القناة على أسعار MEXC الحقيقية", ""]
    L.append(f"- التوصيات المفهومة: {len(sigs)} · الفترة: {iso(sigs[0].ts) if sigs else '-'} → {iso(sigs[-1].ts) if sigs else '-'} UTC")
    L.append(f"- تأخر التنفيذ بعد الرسالة: {delay_sec} ثانية · هامش {t.margin_pct_of_balance}% من الرصيد · أقصى {t.max_concurrent} صفقات · وقف: {t.stop_mode}")
    L.append(f"- الرصيد: {t.paper_balance_usdt}$ → **{balance:.2f}$**")
    L.append(f"- صفقات نُفّذت: {len(traded)} · رابحة: {len(wins)} ({len(wins) / len(traded) * 100:.0f}%)" if traded else "- لا صفقات نُفّذت")
    L.append("- حالة كل التوصيات: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    L.append("- أسباب الخروج: " + ", ".join(f"{k}={v}" for k, v in sorted(exits.items())))
    (out / "signals_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return {"balance": balance, "traded": len(traded), "signals": len(sigs)}
