"""Fees + funding applied to a finished trade (percent of notional)."""
from __future__ import annotations

import bisect


class CostModel:
    BAR_SEC = 900

    def __init__(self, cfg, funding_hist: dict[str, list[tuple[int, float]]] | None = None):
        self.fee = cfg.trading.taker_fee_pct
        self.notional = cfg.trading.trade_size_usdt * cfg.trading.leverage
        bt = cfg.pmo_zero_cross.backtest
        self.interval = int(bt.funding_interval_hours * 3600)
        self.fixed = bt.funding_fixed_pct / 100.0
        self.use_hist = bt.use_mexc_funding_history
        self.hist = funding_hist or {}

    def funding_rates(self, symbol: str, t0: int, t1: int) -> list[float]:
        """Rates (fractions) of every funding settlement strictly inside (t0, t1)."""
        h = self.hist.get(symbol) if self.use_hist else None
        if h and h[0][0] <= t0:
            times = [x[0] for x in h]
            lo, hi = bisect.bisect_right(times, t0), bisect.bisect_left(times, t1)
            return [h[k][1] for k in range(lo, hi)]
        first = (t0 // self.interval + 1) * self.interval
        return [self.fixed for _ in range(first, t1, self.interval)]

    def apply(self, tr) -> None:
        t0, t1 = tr.entry_ts + self.BAR_SEC, tr.exit_ts + self.BAR_SEC
        rates = self.funding_rates(tr.symbol, t0, t1)
        tr.funding_pct = tr.s * sum(rates) * 100.0           # long pays positive funding
        tr.fees_pct = 2 * self.fee
        tr.net_ret_pct = tr.gross_ret_pct - tr.fees_pct - tr.funding_pct
        tr.pnl_usdt = self.notional * tr.net_ret_pct / 100.0
