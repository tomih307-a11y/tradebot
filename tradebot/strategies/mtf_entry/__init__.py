"""Multi-timeframe PMO+MACD entry-zone strategy (paper only; no stop, open target by design — the user closes manually)."""
from __future__ import annotations


def run_backtest_cli(cfg, args) -> None:
    raise SystemExit("mtf_entry: لا اختبار تاريخي بطلب المستخدم. استخدم: live")


def run_live_cli(cfg, args) -> None:
    from .paper import run_loop

    run_loop(cfg, once=getattr(args, "once", False))
