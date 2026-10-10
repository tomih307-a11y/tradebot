"""PMO zero-cross sequence strategy (selectable via [general] strategy = "pmo_zero_cross")."""
from __future__ import annotations


def run_backtest_cli(cfg, args) -> None:
    from tradebot.config import parse_time
    from tradebot.data.klines import get_funding
    from tradebot.mexc.client import MexcClient

    from .backtest import load_features, run_backtest
    from .report import write_report

    bt = cfg.pmo_zero_cross.backtest
    start = parse_time(args.start or bt.start)
    end = parse_time(args.end or bt.end)
    symbols = args.symbols.split(",") if args.symbols else (list(bt.symbols) if bt.has("symbols") else list(cfg.trading.symbols))
    client = MexcClient(cfg.mexc)
    feats, btc = load_features(cfg, client, symbols, start, end)
    if not feats:
        raise SystemExit("No candles were downloaded for any symbol (check network access to MEXC / dates / symbols).")
    print(f"loaded {len(feats)} symbols: " + ", ".join(f"{k}={len(v.ts)} bars" for k, v in sorted(feats.items())))
    funding = {s: get_funding(client, bt.data_dir, s, refresh=True) for s in feats} if bt.use_mexc_funding_history else {}

    def run(c):
        return run_backtest(c, feats, btc, start, end, funding)

    base = run(cfg)
    info = write_report(cfg, run, base, start, end, bt.reports_dir, symbols=sorted(feats))
    print(f"report: {info['report']}  trades: {info['trades']}")


def run_live_cli(cfg, args) -> None:
    from .live import run_live
    run_live(cfg, once=args.once)
