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
    reader = None
    if cfg.agents.channel.enabled and cfg.telegram_signals.channel:
        from tradebot.strategies.telegram_signals.source import make_reader
        reader = make_reader(cfg.telegram_signals.channel)
    Orchestrator(cfg, MexcClient(cfg.mexc), trader, reader).run(once=args.once)


def run_backtest_cli(cfg, args) -> None:
    raise SystemExit("لا باكتست لهذه الاستراتيجية: الأخبار التاريخية غير متاحة. قيّمها بالتشغيل الورقي.")
