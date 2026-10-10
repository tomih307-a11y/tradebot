"""Live runner: after every 15m close, refresh the 3 timeframes and run the shared engine."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from tradebot.data.klines import fetch_range
from tradebot.mexc.client import (CLOSE_LONG, CLOSE_SHORT, OPEN_LONG, OPEN_SHORT, TF_SECONDS,
                                  MexcClient)

from . import model as M
from .backtest import BacktestExecutor
from .costs import CostModel
from .engine import PMOEngine
from .features import build_features
from .model import Trade
from .notify import Notifier

log = logging.getLogger("pmo.live")


class LiveExecutor:
    """DRY_RUN: simulate fills/stops locally (no orders). Real: market orders + exchange Stop orders."""

    def __init__(self, cfg, client: MexcClient, notifier: Notifier):
        self.cfg, self.client, self.notifier = cfg, client, notifier
        self.dry = bool(cfg.trading.dry_run)
        self.sim = BacktestExecutor(cfg)
        self.positions: dict[tuple[str, str], dict] = {}   # (symbol, side) -> exchange position
        self._pending: dict[str, dict] = {}
        self._detail: dict[str, dict] = {}

    # ── exchange snapshot, refreshed once per bar before the engine runs ──
    def refresh_positions(self) -> None:
        if self.dry:
            return
        self.positions = {}
        for p in self.client.open_positions():
            side = "long" if int(p["positionType"]) == 1 else "short"
            self.positions[(p["symbol"], side)] = p

    def _detail_of(self, symbol):
        if symbol not in self._detail:
            self._detail[symbol] = self.client.contract_detail(symbol)
        return self._detail[symbol]

    def _round_price(self, symbol, px):
        unit = float(self._detail_of(symbol).get("priceUnit") or 0)
        return round(round(px / unit) * unit, 12) if unit else px

    def _vol(self, symbol, price):
        d = self._detail_of(symbol)
        notional = self.cfg.trading.trade_size_usdt * self.cfg.trading.leverage
        vol = int(notional / (price * float(d["contractSize"])))
        vol = max(vol, int(d.get("minVol", 1)))
        return vol

    # ── Executor protocol ──
    def open_position(self, symbol, side, ref_price, f, i):
        if self.dry:
            return self.sim.open_position(symbol, side, ref_price, f, i)
        vol = self._vol(symbol, ref_price)
        code = OPEN_LONG if side == "long" else OPEN_SHORT
        self.client.submit_market_order(symbol, code, vol, self.cfg.trading.leverage,
                                        self.cfg.trading.margin_type, ref_price, f"pmo{int(time.time())}{symbol}"[:32])
        for _ in range(10):
            self.refresh_positions()
            p = self.positions.get((symbol, side))
            if p:
                self._pending[symbol] = {"vol": int(p["holdVol"]), "position_id": int(p["positionId"])}
                return float(p["holdAvgPrice"])
            time.sleep(0.5)
        raise RuntimeError(f"order sent but no position seen for {symbol}")

    def place_stop(self, tr: Trade) -> None:
        if self.dry:
            self.notifier.send(f"[DRY_RUN] وقف ابتدائي {tr.symbol} @ {tr.stop:.6g} (لم يُرسل)")
            return
        pend = self._pending.pop(tr.symbol)
        tr.vol, tr.position_id = pend["vol"], pend["position_id"]
        res = self.client.place_stop(tr.position_id, tr.vol, self._round_price(tr.symbol, tr.stop))
        tr.stop_order_id = int(res) if isinstance(res, (int, str)) and str(res).isdigit() else self._find_stop_id(tr)

    def _find_stop_id(self, tr):
        for o in self.client.open_stop_orders(tr.symbol):
            if int(o.get("positionId", -1)) == tr.position_id:
                return int(o.get("id") or o.get("orderId"))
        return None

    def move_stop(self, tr: Trade, new_stop, f, i):
        if self.dry:
            self.notifier.send(f"[DRY_RUN] تعديل وقف {tr.symbol} → {new_stop:.6g} (لم يُرسل)")
            return
        last = float(f.close[i])
        wrong_side = (tr.s == 1 and new_stop >= last) or (tr.s == -1 and new_stop <= last)
        if wrong_side:                                   # cannot rest a stop through the market -> flatten now
            log.warning("%s: new stop %s is beyond price %s; closing at market", tr.symbol, new_stop, last)
            self._market_close(tr, last)
            return
        if tr.stop_order_id is None:
            tr.stop_order_id = self._find_stop_id(tr)
        self.client.change_stop(tr.stop_order_id, self._round_price(tr.symbol, new_stop))

    def _market_close(self, tr, ref):
        code = CLOSE_LONG if tr.side == "long" else CLOSE_SHORT
        self.client.submit_market_order(tr.symbol, code, tr.vol, self.cfg.trading.leverage,
                                        self.cfg.trading.margin_type, ref, f"pmoc{int(time.time())}{tr.symbol}"[:32])

    def close_position(self, tr: Trade, ref_price, reason):
        if self.dry:
            return self.sim.close_position(tr, ref_price, reason)
        if tr.stop_order_id is not None:
            try:
                self.client.cancel_stop(tr.stop_order_id)
            except Exception as e:
                log.warning("cancel stop failed %s: %s", tr.symbol, e)
        self._market_close(tr, ref_price)
        return self.sim.close_position(tr, ref_price, reason)       # estimated fill price

    def stop_hit(self, tr: Trade, f, i):
        if self.dry:
            return self.sim.stop_hit(tr, f, i)
        if (tr.symbol, tr.side) in self.positions:
            return None
        o = float(f.open[i])                                         # position gone => exchange stop fired
        return o if (tr.s == 1 and o <= tr.stop) or (tr.s == -1 and o >= tr.stop) else tr.stop


# ───────────────────────── state file ─────────────────────────
def save_state(path, engine: PMOEngine, last_ts: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"last_ts": last_ts, "next_id": engine.next_id, "last_exit_ts": engine.last_exit_ts,
           "open": [t.to_dict() for t in engine.open.values()]}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(p)


def load_state(path, engine: PMOEngine) -> int | None:
    p = Path(path)
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    engine.next_id = doc["next_id"]
    engine.last_exit_ts = {k: int(v) for k, v in doc["last_exit_ts"].items()}
    engine.open = {d["symbol"]: Trade.from_dict(d) for d in doc["open"]}
    return int(doc["last_ts"])


def reconcile(engine: PMOEngine, ex: LiveExecutor, client: MexcClient, notifier: Notifier) -> None:
    """Match saved trades against real MEXC positions and stop orders (non-DRY_RUN only)."""
    if ex.dry:
        return
    ex.refresh_positions()
    stops = {}
    for o in client.open_stop_orders():
        stops.setdefault(int(o.get("positionId", -1)), o)
    for sym, tr in sorted(engine.open.items()):
        pos = ex.positions.get((sym, tr.side))
        if pos is None:
            notifier.send(f"⚠️ {sym}: لا يوجد مركز على MEXC؛ سيُغلق في أول معالجة (الوقف نُفِّذ أثناء التوقف)")
            continue
        tr.position_id, tr.vol = int(pos["positionId"]), int(pos["holdVol"])
        o = stops.get(tr.position_id)
        if o is None:
            res = client.place_stop(tr.position_id, tr.vol, ex._round_price(sym, tr.stop))
            tr.stop_order_id = ex._find_stop_id(tr)
            notifier.send(f"🔧 {sym}: أمر الوقف مفقود — أُعيد إرساله @ {tr.stop:.6g}")
        else:
            tr.stop_order_id = int(o.get("id") or o.get("orderId"))
            px = float(o.get("stopLossPrice", tr.stop))
            if not math.isclose(px, tr.stop, rel_tol=1e-9):
                client.change_stop(tr.stop_order_id, ex._round_price(sym, tr.stop))
                notifier.send(f"🔧 {sym}: وقف MEXC {px:.6g} ≠ المحفوظ {tr.stop:.6g} — صُحِّح")
    known = {(s, t.side) for s, t in engine.open.items()}
    for (sym, side), pos in sorted(ex.positions.items()):
        if (sym, side) not in known and sym in set(engine.cfg.trading.symbols):
            notifier.send(f"⚠️ مركز غير مُدار على MEXC: {sym} {side} — لن تلمسه الاستراتيجية")


# ───────────────────────── main loop ─────────────────────────
def fetch_features(cfg, client: MexcClient, symbols, now: float):
    s = cfg.pmo_zero_cross
    n = s.live.history_bars
    out = {}
    for sym in symbols:
        k = {}
        for key, tf in (("15", s.tf_entry), ("1h", s.tf_1h), ("4h", s.tf_4h)):
            step = TF_SECONDS[tf]
            end = int(now // step * step)
            k[key] = fetch_range(client, sym, tf, end - n * step, end)
        out[sym] = build_features(sym, k["15"], k["1h"], k["4h"], s, now=now,
                                  liquidity_window=s.backtest.liquidity_window_bars)
    return out


def run_live(cfg, once: bool = False) -> None:
    s = cfg.pmo_zero_cross
    live = s.live
    client = MexcClient(cfg.mexc)
    cells = json.loads(Path(live.cells_file).read_text()) if Path(live.cells_file).exists() else {}
    notifier = Notifier(cfg, cells)
    ex = LiveExecutor(cfg, client, notifier)
    engine = PMOEngine(cfg, ex, CostModel(cfg), on_event=notifier.handle)
    last_ts = load_state(live.state_file, engine)
    symbols = list(cfg.trading.symbols)
    all_syms = list(dict.fromkeys(symbols + [cfg.trading.btc_symbol]))
    reconcile(engine, ex, client, notifier)
    notifier.send(f"▶️ بدء التشغيل الحي — DRY_RUN={cfg.trading.dry_run} · مفتوحة مستعادة: {len(engine.open)}")
    bar = TF_SECONDS[s.tf_entry]
    while True:
        now = time.time()
        due = (now // bar) * bar + live.close_margin_sec
        if now < due:
            time.sleep(due - now)
        now = time.time()
        ts = int(now // bar * bar - bar)                          # open time of the last closed 15m bar
        try:
            feats = fetch_features(cfg, client, all_syms, now)
            if any(f.index_of(ts) is None for f in feats.values()):
                log.warning("bar %s not yet available for all symbols; retrying", ts)
                time.sleep(3)
                continue
            btc = feats.get(cfg.trading.btc_symbol)
            trade_feats = {k: v for k, v in feats.items() if k in symbols}
            if live.min_quote_volume_24h > 0:
                trade_feats = {k: v for k, v in trade_feats.items()
                               if (v.liq24[v.index_of(ts)] or 0) >= live.min_quote_volume_24h or k in engine.open}
            ex.refresh_positions()
            pending = [t for t in (btc.ts if btc is not None else []) if (last_ts is not None and last_ts < t < ts)]
            for t in pending:                                      # catch-up after downtime: no new entries
                engine.process_bar(int(t), trade_feats, btc, entries=False)
            if last_ts is None or ts > last_ts:
                engine.process_bar(ts, trade_feats, btc, entries=True)
                last_ts = ts
                save_state(live.state_file, engine, last_ts)
        except Exception:
            log.exception("cycle failed; will retry next bar")
        if once:
            return
        time.sleep(max(1, bar - live.close_margin_sec - 1))
