"""Live scan for SMC setups (paper: notifications only, no orders). A setup is listed while its CHoCH is on the last 2 closed 15m bars."""
from __future__ import annotations

import logging
import time

import numpy as np

from tradebot.data.klines import drop_unclosed, fetch_range
from tradebot.indicators import atr_rma
from tradebot.mexc.client import MexcClient

from . import model as M

log = logging.getLogger("smc_strategy")


def scan_once(client, top=300, now=None):
    now = int(time.time()) if now is None else now
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    out = []
    for rank, t in enumerate(tick[:top], 1):
        sym = t["symbol"]
        try:
            k15 = drop_unclosed(fetch_range(client, sym, "Min15", int(now // 900 * 900) - 900 * 900, now), "Min15", now)
            k4 = drop_unclosed(fetch_range(client, sym, "Hour4", int(now // 14400 * 14400) - 300 * 14400, now), "Hour4", now)
        except Exception as e:
            log.warning("%s: %s", sym, e)
            continue
        if len(k15["close"]) < 200 or len(k4["close"]) < 60:
            continue
        o, h, l, c = (np.asarray(k15[f], float) for f in ("open", "high", "low", "close"))
        atr = atr_rma(h, l, c, 14)
        if not np.isfinite(atr[-1]):
            continue
        for side in ("long", "short"):
            for su in M.live_setups(o, h, l, c, atr, k15["time"].astype(int), k4, side):
                out.append({"symbol": sym, "rank": rank, **su, "price": float(t["lastPrice"])})
    return sorted(out, key=lambda x: (x["age"], -x["rr"]))


def line(s):
    return (f"{s['symbol']} {'لونق' if s['side'] == 'long' else 'شورت'} · دخول≈{s['entry']:.6g} · وقف {s['stop']:.6g} · هدف {s['target']:.6g} · "
            f"R:R {s['rr']:.1f} · عمر الإشارة {s['age']} شمعة · سيولة #{s['rank']}")


def scan_loop(cfg, once=False):
    from tradebot.strategies.telegram_signals.trader import Notifier

    client, nt, seen = MexcClient(cfg.mexc), Notifier(cfg), set()
    nt.tag = "[SMC] "
    while True:
        try:
            for s in scan_once(client):
                key = (s["symbol"], s["side"], s["choch_i"])
                if key not in seen:
                    seen.add(key)
                    nt.send("🎯 " + line(s))
        except Exception:
            log.exception("scan failed")
        if once:
            return
        time.sleep(60)
