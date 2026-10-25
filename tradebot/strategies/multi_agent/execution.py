"""Agent 6 — automatic trading (execution). The only agent that touches the exchange; it wraps SignalTrader, so it opens with the
configured margin % / concurrency limits, in paper or real mode per trading.dry_run.
To replace it with your own agent, keep these four methods and pass your class to Orchestrator(execution=...)."""
from __future__ import annotations

from tradebot.strategies.telegram_signals.parser import Signal

from .types import Recommendation


class ExecutionAgent:
    def __init__(self, cfg, trader):
        self.cfg, self.trader, self._seq = cfg, trader, 0

    def can_open(self) -> bool:
        return len(self.trader.open) < int(self.cfg.telegram_signals.max_concurrent)

    def open(self, rec: Recommendation, price: float, now: int) -> str:
        self._seq += 1
        sig = Signal(msg_id=-self._seq, ts=now, symbol=rec.symbol, side=rec.side, entry=price, stop=rec.stop,
                     tps=list(rec.tps), leverage=None)
        return self.trader.try_open(sig, now)

    def move_stop(self, pos, price: float) -> None:
        self.trader.ex.move_stop(pos, price)

    def close(self, pos, now: int) -> None:
        self.trader.ex.close(pos, now)
