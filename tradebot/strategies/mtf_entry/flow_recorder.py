"""Trade-flow recorder (public MEXC WebSocket, no keys). Aggregates every executed trade per symbol per minute:
who hit the market (taker side T), whether it opened/closed positions (O), USDT value, count, first/last price.
Why: aggressive buying + new positions is the likeliest early 'spark', and MEXC offers NO history of it — only the live stream.
Usage:  python -m tradebot.strategies.mtf_entry.flow_recorder OUT_DIR MINUTES [MIN_TURNOVER_USDT]
Flag meaning (T: 1/2, O: 1/2/3) is NOT assumed here: all six combinations are stored raw and verified later against price/OI."""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

WS_URL = "wss://contract.mexc.com/edge"
REST = "https://contract.mexc.com"
FIELDS = ["minute", "symbol", "n", "p_open", "p_close", "p_high", "p_low"] + [f"v_T{t}_O{o}" for t in (1, 2) for o in (1, 2, 3)]


class Aggregator:
    """Per-minute, per-symbol accumulator (pure logic, no network)."""

    def __init__(self, csize: dict[str, float]):
        self.csize, self.cur, self.done = csize, {}, []
        self.lock = threading.Lock()

    def add(self, symbol: str, p: float, v: float, T: int, O: int, t_ms: int) -> None:
        minute = int(t_ms // 60000 * 60)
        usdt = p * v * self.csize.get(symbol, 1.0)
        with self.lock:
            key = (minute, symbol)
            if key not in self.cur:
                self.cur[key] = {"n": 0, "p_open": p, "p_close": p, "p_high": p, "p_low": p, **{f"v_T{a}_O{b}": 0.0 for a in (1, 2) for b in (1, 2, 3)}}
            r = self.cur[key]
            r["n"] += 1
            r["p_close"] = p
            r["p_high"], r["p_low"] = max(r["p_high"], p), min(r["p_low"], p)
            k = f"v_T{T}_O{O}"
            if k in r:
                r[k] += usdt

    def flush(self, now_s: float, keep_current: bool = True) -> list[dict]:
        """Return finished minutes (older than the current one); with keep_current=False also the open one."""
        cur_min = int(now_s // 60 * 60)
        out = []
        with self.lock:
            for key in [k for k in self.cur if (k[0] < cur_min or not keep_current)]:
                r = self.cur.pop(key)
                out.append({"minute": key[0], "symbol": key[1], **{k: (round(v, 2) if k.startswith("v_") else v) for k, v in r.items()}})
        return out


def write_part(rows: list[dict], out_dir: str, tag: str) -> str | None:
    if not rows:
        return None
    os.makedirs(out_dir, exist_ok=True)
    first = datetime.fromtimestamp(min(r["minute"] for r in rows), timezone.utc).strftime("%Y%m%d_%H%M")
    path = os.path.join(out_dir, f"flow_{first}_{tag}.csv.gz")
    buf = io.StringIO()
    w = csv.DictWriter(buf, FIELDS)
    w.writeheader()
    w.writerows(rows)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return path


def rest_json(path: str):
    with urllib.request.urlopen(urllib.request.Request(REST + path, headers={"User-Agent": "flow-recorder"}), timeout=20) as r:
        return json.loads(r.read().decode()).get("data")


def universe(min_turnover: float):
    detail = rest_json("/api/v1/contract/detail") or []
    csize = {c["symbol"]: float(c.get("contractSize") or 1.0) for c in detail if c.get("quoteCoin") == "USDT"}
    tick = [t for t in rest_json("/api/v1/contract/ticker") or [] if str(t.get("symbol", "")).endswith("_USDT") and float(t.get("amount24") or 0) >= min_turnover]
    return [t["symbol"] for t in sorted(tick, key=lambda t: -float(t.get("amount24") or 0))], csize


def run(out_dir: str, minutes: float, min_turnover: float = 300_000, flush_every: int = 600):
    import websocket

    syms, csize = universe(min_turnover)
    agg, stats, stop_at = Aggregator(csize), {"msgs": 0, "deals": 0, "subs_ok": 0, "reconnects": 0, "parts": 0}, time.time() + minutes * 60
    print(f"{len(syms)} عقداً · مدة {minutes:.0f} دقيقة", flush=True)

    def on_message(ws, raw):
        try:
            m = json.loads(raw)
        except Exception:
            return
        stats["msgs"] += 1
        ch = m.get("channel", "")
        if ch == "rs.sub.deal":
            stats["subs_ok"] += 1
        elif ch == "push.deal":
            for d in m.get("data") or []:
                try:
                    agg.add(m["symbol"], float(d["p"]), float(d["v"]), int(d["T"]), int(d["O"]), int(d["t"]))
                    stats["deals"] += 1
                except Exception:
                    pass

    def pinger(ws):
        while ws.keep_running and time.time() < stop_at:
            time.sleep(15)
            try:
                ws.send(json.dumps({"method": "ping"}))
            except Exception:
                return

    def on_open(ws):
        def subscribe():
            for s in syms:
                try:
                    ws.send(json.dumps({"method": "sub.deal", "param": {"symbol": s}}))
                except Exception:
                    return
                time.sleep(0.02)
        threading.Thread(target=subscribe, daemon=True).start()
        threading.Thread(target=pinger, args=(ws,), daemon=True).start()

    def flusher():
        last = time.time()
        while time.time() < stop_at:
            time.sleep(5)
            if time.time() - last >= flush_every:
                last = time.time()
                p = write_part(agg.flush(time.time()), out_dir, f"p{stats['parts']}")
                stats["parts"] += 1
                print(f"[{time.strftime('%H:%M:%S')}] part={p} {stats}", flush=True)

    threading.Thread(target=flusher, daemon=True).start()
    while time.time() < stop_at:
        ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_message)
        threading.Thread(target=lambda: (time.sleep(max(stop_at - time.time(), 0)), ws.close()), daemon=True).start()
        ws.run_forever(ping_interval=0)
        if time.time() < stop_at:
            stats["reconnects"] += 1
            time.sleep(3)
    p = write_part(agg.flush(time.time(), keep_current=False), out_dir, f"p{stats['parts']}final")
    print(f"done part={p} {stats}", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    run(out, float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 300_000)
