"""Telegram-channel signals strategy (paper mode only; real execution needs the owner's explicit approval)."""
from __future__ import annotations


def run_live_cli(cfg, args) -> None:
    import logging

    from .source import PublicChannelReader
    from .trader import build

    logging.getLogger("telegram_signals").setLevel(logging.INFO)
    if not cfg.telegram_signals.channel:
        raise SystemExit("ضع اسم القناة في config.toml: [telegram_signals] channel = \"اسم_القناة\"")
    trader = build(cfg)
    trader.run(PublicChannelReader(cfg.telegram_signals.channel), once=args.once)


def run_backtest_cli(cfg, args) -> None:
    from tradebot.mexc.client import MexcClient

    from .backtest import run
    from .source import load_export

    if not getattr(args, "history", None):
        raise SystemExit("مرّر ملف التصدير: --history result.json (من Telegram Desktop: Export chat history، صيغة JSON)")
    msgs = load_export(args.history)
    run(cfg, MexcClient(cfg.mexc), msgs, args.delay, args.out or cfg.pmo_zero_cross.backtest.reports_dir + "/telegram")
