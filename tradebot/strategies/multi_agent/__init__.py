"""Five cooperating agents: scanner, technical analyst, news analyst, entry timing, exit manager.
Trades through the same SignalTrader (paper by default; real orders when trading.dry_run = false)."""
from __future__ import annotations


def run_live_cli(cfg, args) -> None:
    import logging

    from tradebot.mexc.client import MexcClient
    from tradebot.strategies.telegram_signals.trader import build

    from .orchestrator import Orchestrator

    logging.getLogger("multi_agent").setLevel(logging.INFO)
    logging.getLogger("telegram_signals").setLevel(logging.INFO)
    trader = build(cfg)
    Orchestrator(cfg, MexcClient(cfg.mexc), trader).run(once=args.once)


def run_backtest_cli(cfg, args) -> None:
    raise SystemExit("لا باكتست لهذه الاستراتيجية: الأخبار التاريخية غير متاحة. قيّمها بالتشغيل الورقي.")
