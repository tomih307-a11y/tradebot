"""Backtest driver: replays closed 15m bars through the shared PMOEngine."""
from __future__ import annotations

import numpy as np

from . import model as M
from .costs import CostModel
from .engine import PMOEngine
from .features import SymbolFeatures, build_features


class BacktestExecutor:
    def __init__(self, cfg):
        bt = cfg.pmo_zero_cross.backtest
        self.slip = bt.slippage_pct / 100.0
        self.slip_stop = bool(bt.slippage_on_stop)

    def open_position(self, symbol, side, ref_price, f, i):
        s = M.side_sign(side)
        return ref_price * (1 + s * self.slip)              # buy higher / sell lower

    def place_stop(self, tr):
        pass

    def move_stop(self, tr, new_stop, f, i):
        pass

    def close_position(self, tr, ref_price, reason):
        return ref_price * (1 - tr.s * self.slip)

    def stop_hit(self, tr, f, i):
        """Checked against the stop in force from the previous bar. Gap -> open price."""
        o, lo, hi = float(f.open[i]), float(f.low[i]), float(f.high[i])
        if tr.s == 1:
            px = o if o <= tr.stop else (tr.stop if lo <= tr.stop else None)
        else:
            px = o if o >= tr.stop else (tr.stop if hi >= tr.stop else None)
        if px is None:
            return None
        if self.slip_stop:
            px *= (1 - tr.s * self.slip)
        return px


def run_backtest(cfg, feats: dict[str, SymbolFeatures], btc: SymbolFeatures | None, start_ts: int,
                 end_ts: int, funding_hist=None, trace=False) -> PMOEngine:
    """feats must already be built from closed candles. Entries only for bars in [start_ts, end_ts]."""
    engine = PMOEngine(cfg, BacktestExecutor(cfg), CostModel(cfg, funding_hist),
                       trade_from_ts=start_ts, trace=trace)
    all_feats = dict(feats)
    allts = np.unique(np.concatenate([f.ts for f in feats.values()])) if feats else np.array([], dtype=np.int64)
    timeline = [int(t) for t in allts[(allts >= start_ts) & (allts <= end_ts)]]
    for ts in timeline:
        engine.process_bar(ts, all_feats, btc)
    if engine.open and timeline:
        engine.force_close_all(timeline[-1], all_feats)
    return engine


def load_features(cfg, client, symbols: list[str], start_ts: int, end_ts: int, now=None):
    """Download/cache candles (with warm-up padding before start) and build features."""
    from tradebot.data.klines import get_klines
    from tradebot.mexc.client import TF_SECONDS
    s = cfg.pmo_zero_cross
    pad_bars = s.indicator.warmup_bars + 50
    data_dir = s.backtest.data_dir
    out, btc = {}, None
    names = list(dict.fromkeys(list(symbols) + [cfg.trading.btc_symbol]))
    every = s.backtest.progress_every
    for n_done, sym in enumerate(names, 1):
        if n_done % every == 0:
            print(f"  downloaded {n_done}/{len(names)} symbols", flush=True)
        k = {}
        for key, tf in (("15", s.tf_entry), ("1h", s.tf_1h), ("4h", s.tf_4h)):
            t0 = start_ts - pad_bars * TF_SECONDS[tf]
            k[key] = get_klines(client, data_dir, sym, tf, t0, end_ts, now=now)
        if len(k["15"]["time"]) == 0 or len(k["1h"]["time"]) == 0 or len(k["4h"]["time"]) == 0:
            continue
        out[sym] = build_features(sym, k["15"], k["1h"], k["4h"], s, liquidity_window=s.backtest.liquidity_window_bars)
    btc = out.get(cfg.trading.btc_symbol)
    trade_syms = {n: out[n] for n in symbols if n in out}
    return trade_syms, btc
