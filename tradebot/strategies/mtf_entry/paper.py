"""Paper runner: scans top-liquidity contracts, opens a paper position when a 15m bar touches the entry zone in an aligned direction.
Open target + trailing stop (chandelier: extreme since entry -/+ 3 x ATR(14) of the 1H frame, only ratchets in favour); the stop is checked every loop.
Manual close:  python -m tradebot.strategies.mtf_entry.paper close SYMBOL     |     ... status
Limits: 10% of balance margin per trade, max 2 concurrent. Never sends real orders."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from tradebot.data.klines import drop_unclosed, fetch_range
from tradebot.mexc.client import MexcClient, TF_SECONDS

from . import model as M

log = logging.getLogger("mtf_entry")
STATE = Path("state/mtf_paper.json")
START_BALANCE, MARGIN_PCT, MAX_OPEN, LEVERAGE, FEE, SLIP, TOP = 25.0, 0.10, 2, 5, 0.0004, 0.0005, 60
NON_CRYPTO = {"CAD_USDT", "US30_USDT", "NAS100_USDT", "SPX500_USDT", "SPY_USDT", "QQQ_USDT", "IGV_USDT", "XLI_USDT", "SOXL_USDT", "SOXS_USDT",
              "KORU_USDT", "GGLL_USDT", "MUU_USDT", "XAU_USDT", "XAUT_USDT", "SILVER_USDT", "UKOIL_USDT", "USOIL_USDT", "EUR_USDT", "GBP_USDT", "JPY_USDT"}
BARS = {"Day1": 300, "Hour4": 300, "Min60": 300, "Min30": 300, "Min15": 300}


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"balance": START_BALANCE, "open": [], "closed": [], "seen": {}}


def save(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))


def fetch(client, sym: str, tf: str, now: int) -> dict:
    s = TF_SECONDS[tf]
    return drop_unclosed(fetch_range(client, sym, tf, int(now // s * s) - BARS[tf] * s, now), tf, now)


def analyse(client, sym: str, now: int):
    ks = {tf: fetch(client, sym, tf, now) for tf, _ in M.FRAMES}
    states = {tf: M.tf_state(np.asarray(k["close"], float)) for tf, k in ks.items()}
    d, score = M.direction(states)
    z = ks[M.ZONE_TF]
    lean = d or (1 if score > 0 else -1 if score < 0 else next((v for v in (states[tf] for tf, _ in M.FRAMES) if v), 0))
    zone = M.entry_zone(*(np.asarray(z[f], float) for f in ("high", "low", "close")), lean)  # zone shown for the lean too; entries need d != 0
    k15 = ks["Min15"]
    hit = bool(len(k15["low"]) and M.touches(zone, float(k15["low"][-1]), float(k15["high"][-1])))
    return d, score, zone, hit, int(k15["time"][-1]) if len(k15["time"]) else 0, states


def manage(client, st: dict, px: dict, now: int, nt) -> None:
    """Update the trailing stop of every open paper position with the latest price; close it when the stop is hit."""
    for pos in list(st["open"]):
        price = px.get(pos["symbol"])
        if not price:
            continue
        d = 1 if pos["side"] == "long" else -1
        ext = pos.get("extreme", pos["entry"])
        ext = max(ext, price) if d == 1 else min(ext, price)
        pos["extreme"] = ext
        k = fetch(client, pos["symbol"], M.ZONE_TF, now)
        atr = M.last_atr(*(np.asarray(k[f], float) for f in ("high", "low", "close")))
        if atr == atr:
            pos["stop"] = M.ratchet(d, pos.get("stop"), M.trail_stop(d, ext, atr))
        if pos.get("stop") is not None and (price <= pos["stop"] if d == 1 else price >= pos["stop"]):
            pnl = close_pos(st, pos, price, now)
            nt.send(f"🛑 ورقي {pos['symbol']} أُغلقت بالوقف المتحرك @ {price:.6g} · {pnl:+.3f} USDT")


def open_pos(st: dict, sym: str, d: int, price: float, now: int) -> dict:
    margin = st["balance"] * MARGIN_PCT
    px = price * (1 + SLIP * d)
    qty = margin * LEVERAGE / px
    st["balance"] -= margin * LEVERAGE * FEE
    pos = {"symbol": sym, "side": "long" if d == 1 else "short", "entry": px, "qty": qty, "margin": margin, "t": now}
    st["open"].append(pos)
    return pos


def close_pos(st: dict, pos: dict, price: float, now: int) -> float:
    d = 1 if pos["side"] == "long" else -1
    px = price * (1 - SLIP * d)
    pnl = (px - pos["entry"]) * pos["qty"] * d - px * pos["qty"] * FEE
    st["balance"] += pnl
    st["open"].remove(pos)
    st["closed"].append({**pos, "exit": px, "pnl": pnl, "t_exit": now})
    return pnl


def run_loop(cfg, once: bool = False) -> None:
    from tradebot.strategies.telegram_signals.trader import Notifier

    assert cfg.trading.dry_run, "mtf_entry يعمل ورقياً فقط"
    client, nt = MexcClient(cfg.mexc), Notifier(cfg)
    nt.tag = "[MTF] "
    while True:
        try:
            now, st = int(time.time()), load()
            tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
            tick.sort(key=lambda t: -float(t.get("amount24") or 0))
            manage(client, st, {t["symbol"]: float(t["lastPrice"]) for t in tick}, now, nt)
            for t in tick[:TOP]:
                sym = t["symbol"]
                try:
                    d, score, zone, hit, bar_t, _ = analyse(client, sym, now)
                except Exception as e:
                    log.warning("%s: %s", sym, e)
                    continue
                if d == 0 or not hit or st["seen"].get(sym) == bar_t or any(p["symbol"] == sym for p in st["open"]):
                    continue
                if len(st["open"]) >= MAX_OPEN:
                    continue
                st["seen"][sym] = bar_t
                p = open_pos(st, sym, d, float(t["lastPrice"]), now)
                nt.send(f"🎯 ورقي {sym} {'لونق' if d == 1 else 'شورت'} @ {p['entry']:.6g} · نتيجة {score:+.2f} · منطقة {zone[0]:.6g}-{zone[1]:.6g} · هدف مفتوح + وقف متحرك")
            save(st)
        except Exception:
            log.exception("scan failed")
        if once:
            return
        time.sleep(60)


def snapshot(client, top: int = TOP) -> list[dict]:
    """Read-only live scan: direction, score, zone and whether the last closed 15m bar touches it, for the top contracts by liquidity."""
    now = int(time.time())
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    out = []
    for rank, t in enumerate(tick[:top], 1):
        try:
            d, score, zone, hit, bar_t, states = analyse(client, t["symbol"], now)
        except Exception as e:
            log.warning("%s: %s", t["symbol"], e)
            continue
        out.append({"symbol": t["symbol"], "rank": rank, "dir": d, "score": score, "zone": zone, "hit": hit,
                    "price": float(t["lastPrice"]), "states": states})
    return out


def early_snapshot(client, top: int = TOP) -> list[dict]:
    """Read-only: contracts whose HTF bias is set, price is at/near the 1H zone, move not yet extended, 15m PMO about to cross / just crossed."""
    now = int(time.time())
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    out = []
    for rank, t in enumerate(tick[:top], 1):
        sym = t["symbol"]
        if sym.endswith("STOCK_USDT") or sym in NON_CRYPTO:
            continue
        try:
            ks = {tf: fetch(client, sym, tf, now) for tf in ("Day1", "Hour4", "Min60", "Min15")}
            states = {tf: M.tf_state(np.asarray(ks[tf]["close"], float)) for tf in ("Day1", "Hour4")}
            a = lambda tf, f: np.asarray(ks[tf][f], float)
            su = M.early_setup(states, a("Min60", "high"), a("Min60", "low"), a("Min60", "close"), M.pmo_gap(a("Min15", "close")), float(t["lastPrice"]))
        except Exception as e:
            log.warning("%s: %s", sym, e)
            continue
        if su:
            out.append({"symbol": sym, "rank": rank, "price": float(t["lastPrice"]), **su})
    return out


def status(client) -> None:
    st = load()
    px = {t["symbol"]: float(t["lastPrice"]) for t in client.tickers()}
    print(f"الرصيد: {st['balance']:.2f} USDT")
    for p in st["open"]:
        d = 1 if p["side"] == "long" else -1
        print(f"{p['symbol']} {p['side']} دخول {p['entry']:.6g} الآن {px.get(p['symbol'], float('nan')):.6g} · غير محقق {(px.get(p['symbol'], p['entry']) - p['entry']) * p['qty'] * d:+.3f} · وقف {p.get('stop') or float('nan'):.6g}")
    print(f"صفقات مغلقة: {len(st['closed'])}")


if __name__ == "__main__":
    c = MexcClient(__import__("tradebot.config", fromlist=["load_config"]).load_config("config.toml").mexc)
    if len(sys.argv) >= 2 and sys.argv[1] == "early":
        rows = early_snapshot(c, int(sys.argv[2]) if len(sys.argv) > 2 else TOP)
        print(f"فرص مبكرة: {len(rows)}")
        inside = lambda r: r["zone"][0] <= r["price"] <= r["zone"][1]
        for r in sorted(rows, key=lambda r: (not inside(r), r["turn"][0] != "imminent", r["turn"][1])):
            kind = f"يتقاطع خلال ~{r['turn'][1]:.1f} شمعة 15د" if r["turn"][0] == "imminent" else f"تقاطع قبل {r['turn'][1]} شمعة"
            print(f"{'داخل المنطقة ' if inside(r) else 'قرب المنطقة '}{'لونق ' if r['dir'] == 1 else 'شورت'} {r['symbol']:<14} سعر {r['price']:.6g} · منطقة {r['zone'][0]:.6g}-{r['zone'][1]:.6g} · "
                  f"بعد {r['away_atr']:+.2f} ATR عن المنطقة · امتداد {r['ext_atr']:+.2f} ATR · {kind}")
    elif len(sys.argv) >= 2 and sys.argv[1] == "scan":
        rows = snapshot(c, int(sys.argv[2]) if len(sys.argv) > 2 else TOP)
        def lean(r):  # always a side: sign of the weighted score, ties broken by the highest frames
            if r["score"]:
                return 1 if r["score"] > 0 else -1
            return next((v for v in (r["states"][tf] for tf, _ in M.FRAMES) if v), 0)

        print(f"العقود الممسوحة: {len(rows)}  ·  مؤكد: {sum(bool(r['dir']) for r in rows)}  ·  ميل ضعيف: {sum(not r['dir'] for r in rows)}")
        for r in sorted(rows, key=lambda r: (-bool(r["dir"]), -abs(r["score"]))):
            d = lean(r)
            if d == 0:
                continue
            h = np.asarray
            z = f"{r['zone'][0]:.6g}-{r['zone'][1]:.6g}" if r["zone"] else "-"
            tag = "مؤكد" if r["dir"] else "ميل ضعيف"
            print(f"{'لونق ' if d == 1 else 'شورت'} {tag:<8} {r['symbol']:<14} نتيجة {r['score']:+.2f} · سعر {r['price']:.6g} · منطقة {z}{' 🎯داخلها' if r['hit'] else ''} · "
                  + " ".join(f"{k[:4]}{'▲' if v == 1 else '▼' if v == -1 else '•'}" for k, v in r["states"].items()))
    elif len(sys.argv) >= 2 and sys.argv[1] == "status":
        status(c)
    elif len(sys.argv) == 3 and sys.argv[1] == "close":
        st = load()
        pos = next((p for p in st["open"] if p["symbol"] == sys.argv[2]), None)
        if not pos:
            sys.exit("لا صفقة مفتوحة بهذا الاسم")
        price = next(float(t["lastPrice"]) for t in c.tickers() if t["symbol"] == sys.argv[2])
        print(f"أُغلقت {sys.argv[2]}: {close_pos(st, pos, price, int(time.time())):+.3f} USDT")
        save(st)
