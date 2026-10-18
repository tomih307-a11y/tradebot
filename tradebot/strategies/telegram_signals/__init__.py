"""Telegram-channel signals strategy (paper mode only; real execution needs the owner's explicit approval)."""
from __future__ import annotations


def run_live_cli(cfg, args) -> None:
    import logging

    from .source import make_reader
    from .trader import build

    logging.getLogger("telegram_signals").setLevel(logging.INFO)
    if not cfg.telegram_signals.channel:
        raise SystemExit("ضع اسم القناة في config.toml: [telegram_signals] channel = \"اسم_القناة\"")
    trader = build(cfg)
    trader.run(make_reader(cfg.telegram_signals.channel), once=args.once)


def run_backtest_cli(cfg, args) -> None:
    from tradebot.mexc.client import MexcClient

    from .backtest import run
    from .source import load_export

    if not getattr(args, "history", None):
        raise SystemExit("مرّر ملف التصدير: --history result.json (من Telegram Desktop: Export chat history، صيغة JSON)")
    msgs = load_export(args.history)
    run(cfg, MexcClient(cfg.mexc), msgs, args.delay, args.out or cfg.pmo_zero_cross.backtest.reports_dir + "/telegram")


def tg_login() -> None:
    """One-time interactive login (phone + code typed HERE on your PC; nothing is sent anywhere else)."""
    import os

    from telethon.sync import TelegramClient
    aid, ah = os.environ.get("TG_API_ID", ""), os.environ.get("TG_API_HASH", "")
    if not (aid and ah):
        raise SystemExit("ضع TG_API_ID و TG_API_HASH أولاً (من my.telegram.org)")
    with TelegramClient("state/tg_session", int(aid), ah) as c:
        me = c.get_me()
        print("تم تسجيل الدخول:", getattr(me, "first_name", ""))
        for d in c.get_dialogs(limit=300):
            if d.is_channel:
                print(d.id, "|", d.name)
