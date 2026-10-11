"""PMO zero-cross sequence engine — shared verbatim by backtest and live.

process_bar() runs section 7 in this exact order for one closed 15m bar:
  A) stop-hit check inside the bar, using the stop in force from the previous bar
  B) refresh higher-timeframe view (1H / 4H) of each open trade
  C) stage transitions
  D) failure exits
  E) trailing-stop update
  F) new entry signals (priority + max-open cap)
"""
from __future__ import annotations

import math
from typing import Callable, Protocol

from . import model as M
from .features import SymbolFeatures
from .model import Trade


class Executor(Protocol):
    def open_position(self, symbol: str, side: str, ref_price: float, f: SymbolFeatures, i: int) -> float: ...
    def place_stop(self, tr: Trade) -> None: ...
    def move_stop(self, tr: Trade, new_stop: float, f: SymbolFeatures, i: int) -> None: ...
    def close_position(self, tr: Trade, ref_price: float, reason: str) -> float: ...
    def stop_hit(self, tr: Trade, f: SymbolFeatures, i: int) -> float | None: ...


class PMOEngine:
    BAR_SEC = 900

    def __init__(self, cfg, executor: Executor, costs=None, on_event: Callable | None = None,
                 trade_from_ts: int | None = None, trace: bool = False):
        self.cfg = cfg
        self.s = cfg.pmo_zero_cross
        self.tr_cfg = cfg.trading
        self.ex = executor
        self.costs = costs
        self.on_event = on_event or (lambda kind, trade, info: None)
        self.trade_from_ts = trade_from_ts
        self.open: dict[str, Trade] = {}
        self.closed: list[Trade] = []
        self.rejected: list[dict] = []
        self.last_exit_ts: dict[str, int] = {}
        self.next_id = 1
        self.trace_on = trace
        self.trace: list[tuple] = []        # (ts, step, symbol) when trace=True

    # ───────────────────────── main entry point ─────────────────────────
    def process_bar(self, ts: int, feats: dict[str, SymbolFeatures], btc: SymbolFeatures | None = None,
                    entries: bool = True) -> None:
        views: dict[str, tuple[SymbolFeatures, int]] = {}
        for sym in sorted(self.open):
            f = feats.get(sym)
            i = f.index_of(ts) if f is not None else None
            if i is not None:
                views[sym] = (f, i)

        # A) stop hit with the stop from the previous bar
        for sym in sorted(views):
            tr = self.open[sym]
            f, i = views[sym]
            self._t(ts, "A", sym)
            self._track(tr, f, i)
            fill = self.ex.stop_hit(tr, f, i)
            if fill is not None:
                self._exit(tr, ts, fill, M.STOP_REASON_BY_STAGE[tr.stop_stage])
        live = {s: v for s, v in views.items() if s in self.open}

        # B) refresh HTF view (values are already aligned to last CLOSED candles in `feats`)
        for sym in sorted(live):
            self._t(ts, "B", sym)
            tr = self.open[sym]
            f, i = live[sym]
            if tr.stage == M.WAIT_1H and f.closed1h[i]:
                self._set_deadline(tr, ts, f, i)

        # C) stage transitions
        for sym in sorted(live):
            self._t(ts, "C", sym)
            self._transitions(self.open[sym], ts, *live[sym])

        # D) failure exits
        for sym in sorted(live):
            self._t(ts, "D", sym)
            tr = self.open[sym]
            f, i = live[sym]
            why = self._failure(tr, ts, f, i)
            if why:
                tr.fail_detail = why[1]
                fill = self.ex.close_position(tr, float(f.close[i]), why[0])
                self._exit(tr, ts, fill, why[0])
        live = {s: v for s, v in live.items() if s in self.open}

        # E) trailing stop update
        for sym in sorted(live):
            self._t(ts, "E", sym)
            self._update_stop(self.open[sym], ts, *live[sym])

        # F) new entries
        self._t(ts, "F", "*")
        if entries:
            self._entries(ts, feats, btc)

    # ───────────────────────── helpers ─────────────────────────
    def _t(self, ts, step, sym):
        if self.trace_on:
            self.trace.append((ts, step, sym))

    def _track(self, tr: Trade, f: SymbolFeatures, i: int) -> None:
        if f.ts[i] <= tr.entry_ts:
            return
        hi, lo = float(f.high[i]), float(f.low[i])
        k = M.STAGE_RANK[tr.stage]                     # the bar belongs to the stage in force before this bar's transitions
        pb = tr.phase_best[k]
        if tr.s == 1:
            tr.best_price, tr.worst_price = max(tr.best_price, hi), min(tr.worst_price, lo)
            tr.phase_best[k] = hi if pb is None else max(pb, hi)
        else:
            tr.best_price, tr.worst_price = min(tr.best_price, lo), max(tr.worst_price, hi)
            tr.phase_best[k] = lo if pb is None else min(pb, lo)

    def _bars(self, tr: Trade, ts: int) -> int:
        return int((ts - tr.entry_ts) // self.BAR_SEC)

    def _set_stage(self, tr: Trade, stage: str, ts: int, price: float) -> None:
        tr.stage = stage
        if M.STAGE_RANK[stage] > M.STAGE_RANK[tr.max_stage]:
            tr.max_stage = stage
        tr.transitions.append([stage, self._bars(tr, ts), price])

    def _set_deadline(self, tr: Trade, ts: int, f: SymbolFeatures, i: int) -> None:
        """WAIT_DEADLINE = bar_no + min(cap, max(MIN_WAIT_1H, ceil(ETA_1H * 4 * WAIT_ETA_MULT))); inf ETA -> MIN."""
        e = self.s.exits
        eta = M.eta_candles(float(f.pmo1h[i]), float(f.slope1h[i]), tr.s)
        wait = e.min_wait_1h
        if not math.isinf(eta):
            wait = max(wait, math.ceil(eta * 4 * e.wait_eta_mult - 1e-9))
        wait = min(wait, e.max_wait_1h_cap)
        bar_no = int(ts // self.BAR_SEC)
        tr.wait_deadline = bar_no + int(wait)
        tr.deadline_log.append([bar_no, tr.wait_deadline])

    # ───────────────────────── C) transitions ─────────────────────────
    def _transitions(self, tr: Trade, ts: int, f: SymbolFeatures, i: int) -> None:
        s = tr.s
        c = float(f.close[i])
        if tr.stage == M.WAIT_1H and f.closed1h[i] and s * f.pmo1h[i] >= 0:
            self._set_stage(tr, M.WAIT_4H, ts, c)
            tr.h1_cross_bars, tr.h1_cross_price = self._bars(tr, ts), c
            self.on_event("stage_1h", tr, {"ts": ts, "price": c})
            if s * f.pmo4h[i] >= 0:                      # 4H already across -> RIDE immediately
                self._set_stage(tr, M.RIDE, ts, c)
                tr.h4_cross_bars, tr.h4_cross_price = self._bars(tr, ts), c
                self.on_event("stage_4h", tr, {"ts": ts, "price": c})
        elif tr.stage == M.WAIT_4H and f.closed4h[i] and s * f.pmo4h[i] >= 0:
            self._set_stage(tr, M.RIDE, ts, c)
            tr.h4_cross_bars, tr.h4_cross_price = self._bars(tr, ts), c
            self.on_event("stage_4h", tr, {"ts": ts, "price": c})

    # ───────────────────────── D) failures ─────────────────────────
    def _failure(self, tr: Trade, ts: int, f: SymbolFeatures, i: int):
        e, s = self.s.exits, tr.s
        if tr.stage == M.WAIT_1H:
            if e.fail_1h_slope and f.closed1h[i] and s * f.slope1h[i] < 0:
                return M.FAIL_1H, "1h_slope"
            if e.fail_1h_pmo15_back and s * f.pmo15[i] < 0:
                return M.FAIL_1H, "pmo15_back"
            if e.fail_1h_timeout and tr.wait_deadline is not None and int(ts // self.BAR_SEC) > tr.wait_deadline:
                return M.FAIL_1H, "timeout"
        elif tr.stage == M.WAIT_4H:
            if e.fail_4h_slope and f.closed4h[i] and s * f.slope4h[i] < 0:
                return M.FAIL_4H, "4h_slope"
            if e.fail_4h_pmo1h_back and s * f.pmo1h[i] < 0:
                return M.FAIL_4H, "pmo1h_back"
        return None

    # ───────────────────────── E) trailing stop ─────────────────────────
    def _update_stop(self, tr: Trade, ts: int, f: SymbolFeatures, i: int) -> None:
        if ts <= tr.entry_ts:           # trade opened on this very bar: starts from the project stop
            return
        s, t = tr.s, self.s.trail
        c = float(f.close[i])
        tr.extreme_close = max(tr.extreme_close, c) if s == 1 else min(tr.extreme_close, c)

        base = {M.WAIT_1H: 0, M.WAIT_4H: 1, M.RIDE: 2}[tr.stage]
        tr.trail_stage = max(tr.trail_stage, base)
        if tr.stage == M.RIDE and f.closed1h[i] and s * f.slope1h[i] < 0:
            tr.trail_stage = 3                                   # momentum weakened: tighten for good

        cands = []
        if tr.trail_stage == 1:
            fee = self.tr_cfg.taker_fee_pct / 100.0
            be = tr.entry_price * (1 + s * 2 * fee)               # break-even incl. round-trip fees
            if not t.breakeven_requires_protective or s * (c - be) > 0:
                cands.append(be)       # (by default only once price is beyond it; else it would be hit at once)
            cands.append(self._trail(tr, f.atr15[i], t.mult_1))
        elif tr.trail_stage == 2:
            cands.append(self._trail(tr, f.atr1h[i], t.mult_2))
        elif tr.trail_stage == 3:
            cands.append(self._trail(tr, f.atr15[i], t.mult_3))
        cands = [x for x in cands if x is not None and not math.isnan(x)]
        if not cands:
            return
        new = max([tr.stop] + cands) if s == 1 else min([tr.stop] + cands)   # never moves backwards
        if new != tr.stop:
            old = tr.stop
            tr.stop, tr.stop_stage = new, tr.trail_stage
            tr.n_stop_moves += 1
            self.ex.move_stop(tr, new, f, i)
            self.on_event("stop_move", tr, {"ts": ts, "old": old, "new": new, "price": c})

    @staticmethod
    def _trail(tr: Trade, atr: float, mult: float):
        if math.isnan(atr):
            return None
        return tr.extreme_close - tr.s * atr * mult

    # ───────────────────────── exits ─────────────────────────
    def _exit(self, tr: Trade, ts: int, fill: float, reason: str) -> None:
        tr.exit_ts, tr.exit_price, tr.exit_reason = ts, float(fill), reason
        tr.bars_held = self._bars(tr, ts)
        s, e = tr.s, tr.entry_price
        tr.gross_ret_pct = s * (fill - e) / e * 100.0
        best = max(tr.best_price, fill) if s == 1 else min(tr.best_price, fill)
        worst = min(tr.worst_price, fill) if s == 1 else max(tr.worst_price, fill)
        tr.mfe_pct = s * (best - e) / e * 100.0
        tr.mae_pct = s * (worst - e) / e * 100.0
        tr.mfe_retained = (tr.gross_ret_pct / tr.mfe_pct) if tr.mfe_pct > 0 else None
        self._phases(tr, fill)
        if self.costs is not None:
            self.costs.apply(tr)
        self.open.pop(tr.symbol, None)
        self.closed.append(tr)
        self.last_exit_ts[tr.symbol] = ts
        self.on_event("exit", tr, {"ts": ts, "price": fill, "reason": reason})

    @staticmethod
    def _phases(tr: Trade, fill: float) -> None:
        """Favorable price move (% of entry) per phase: entry->1H cross, 1H->4H cross, 4H cross->exit."""
        e, s = tr.entry_price, tr.s
        h1, h4 = tr.h1_cross_price, tr.h4_cross_price
        b0 = e
        b1 = h1 if h1 is not None else fill
        b2 = h4 if h4 is not None else (fill if h1 is not None else b1)
        b3 = fill if h4 is not None else b2
        tr.phase_moves = [s * (b1 - b0) / e * 100.0, s * (b2 - b1) / e * 100.0, s * (b3 - b2) / e * 100.0]
        starts = [e, h1, h4]
        tr.phase_mfe = [None if (tr.phase_best[k] is None or starts[k] is None)
                        else max(0.0, s * (tr.phase_best[k] - starts[k]) / e * 100.0) for k in range(3)]

    def force_close_all(self, ts: int, feats: dict[str, SymbolFeatures]) -> None:
        for sym in sorted(self.open):
            tr = self.open[sym]
            f = feats[sym]
            i = f.index_of(ts)
            if i is None:
                i = len(f.ts) - 1
            self._exit(tr, int(f.ts[i]), float(f.close[i]), M.END_OF_DATA)

    # ───────────────────────── F) entries ─────────────────────────
    def _entries(self, ts: int, feats: dict[str, SymbolFeatures], btc: SymbolFeatures | None) -> None:
        if self.trade_from_ts is not None and ts < self.trade_from_ts:
            return
        cands = []
        for sym in sorted(feats):
            f = feats[sym]
            if sym in self.open:
                continue                                  # open trade: any new cross is ignored
            if self.last_exit_ts.get(sym, -1) >= ts:
                continue                                  # cross must come AFTER the exit candle
            i = f.index_of(ts)
            if i is None or i < 1:
                continue
            p0, p1 = f.pmo15[i - 1], f.pmo15[i]
            if math.isnan(p0) or math.isnan(p1):
                continue
            if p0 < 0 and p1 >= 0:
                side = "long"
            elif p0 > 0 and p1 <= 0:
                side = "short"
            else:
                continue
            s = M.side_sign(side)
            h1 = M.frame_state(f.pmo1h[i], f.signal1h[i], f.slope1h[i], s)
            h4 = M.frame_state(f.pmo4h[i], f.signal4h[i], f.slope4h[i], s)
            if math.isnan(f.pmo1h[i]) or math.isnan(f.pmo4h[i]):
                self._reject(ts, sym, side, "insufficient_data")
                continue
            stage = M.WAIT_1H if s * f.pmo1h[i] < 0 else (M.WAIT_4H if s * f.pmo4h[i] < 0 else M.RIDE)
            why = self._filter_reject(s, h1, h4, btc, ts)
            if why:
                self._reject(ts, sym, side, why)
                continue
            liq = 0.0 if math.isnan(f.liq24[i]) else float(f.liq24[i])
            cands.append({"sym": sym, "side": side, "f": f, "i": i, "stage": stage,
                          "h1": h1, "h4": h4, "eta": h1["eta"], "liq": liq,
                          "n_with": M.n_slopes_with(h1, h4)})

        # priority: (1) more larger-frame slopes WITH the trade, (2) initial stage, (3) lower ETA_1H, (4) liquidity
        cands.sort(key=lambda c: (-c["n_with"], M.ENTRY_PRIORITY[c["stage"]], c["eta"], -c["liq"], c["sym"]))
        slots = self.tr_cfg.max_open_trades - len(self.open)
        for k, c in enumerate(cands):
            if k >= slots:
                self._reject(ts, c["sym"], c["side"], "max_open_full")
                continue
            self._open(ts, c, btc)

    def _filter_reject(self, s: int, h1: dict, h4: dict, btc, ts: int):
        fl = self.s.filters
        if fl.filter_1h_slope and h1["slope_dir"] != "with":
            return "FILTER_1H_SLOPE"
        if fl.filter_1h_signal:
            ok = h1["pmo"] > h1["signal"] if s == 1 else h1["pmo"] < h1["signal"]
            if not ok:
                return "FILTER_1H_SIGNAL"
        if fl.filter_eta_max and not (h1["eta"] <= fl.eta_max_value):
            return "FILTER_ETA_MAX"
        if fl.filter_4h_slope and h4["slope_dir"] != "with":
            return "FILTER_4H_SLOPE"
        if fl.filter_btc:
            j = btc.index_of(ts) if btc is not None else None
            if j is None or math.isnan(btc.pmo15[j]) or s * btc.pmo15[j] < 0:
                return "FILTER_BTC"
        return None

    def _reject(self, ts, sym, side, reason):
        self.rejected.append({"ts": ts, "symbol": sym, "side": side, "reason": reason})
        self.on_event("reject", None, {"ts": ts, "symbol": sym, "side": side, "reason": reason})

    def _open(self, ts: int, c: dict, btc: SymbolFeatures | None) -> None:
        f, i, side, s = c["f"], c["i"], c["side"], M.side_sign(c["side"])
        close = float(f.close[i])
        fill = self.ex.open_position(c["sym"], side, close, f, i)
        sl = self.tr_cfg.stop_loss_pct / 100.0
        stop0 = fill * (1 - s * sl)
        snap = {"h1": c["h1"], "h4": c["h4"], "slope_class": M.slope_class(c["h1"], c["h4"]),
                "n_slopes_with": c["n_with"]}
        j = btc.index_of(ts) if btc is not None else None
        if j is not None:
            snap["btc15"] = {"pmo": float(btc.pmo15[j]), "signal": float(btc.signal15[j]), "slope": float(btc.slope15[j])}
            snap["btc1h"] = {"pmo": float(btc.pmo1h[j]), "signal": float(btc.signal1h[j]), "slope": float(btc.slope1h[j])}
        trail0 = {M.WAIT_1H: 0, M.WAIT_4H: 1, M.RIDE: 2}[c["stage"]]
        tr = Trade(id=self.next_id, symbol=c["sym"], side=side, entry_ts=ts, entry_price=fill,
                   initial_stage=c["stage"], stage=c["stage"], max_stage=c["stage"], stop=stop0,
                   initial_stop=stop0, trail_stage=trail0, extreme_close=close,
                   best_price=fill, worst_price=fill, snap=snap)
        if c["stage"] != M.WAIT_1H:
            tr.h1_cross_bars, tr.h1_cross_price = 0, fill
        if c["stage"] == M.RIDE:
            tr.h4_cross_bars, tr.h4_cross_price = 0, fill
        tr.transitions.append([c["stage"], 0, fill])
        if c["stage"] == M.WAIT_1H:
            self._set_deadline(tr, ts, f, i)
        self.next_id += 1
        self.open[tr.symbol] = tr
        self.ex.place_stop(tr)
        self.on_event("entry", tr, {"ts": ts, "price": fill, "liq": c["liq"]})
