"""PMO zero-cross sequence strategy (selectable via [general] strategy = "pmo_zero_cross")."""
from __future__ import annotations


def run_backtest_cli(cfg, args) -> None:
    from tradebot.config import parse_time
    from tradebot.data.klines import get_funding
    from tradebot.mexc.client import MexcClient

    from .backtest import load_features, run_backtest
    from .report import write_report

    from .report import _override
    if args.max_open:
        cfg = _override(cfg, **{"trading.max_open_trades": args.max_open})
    bt = cfg.pmo_zero_cross.backtest
    start = parse_time(args.start or bt.start)
    end = parse_time(args.end or bt.end)
    raw = args.symbols if args.symbols else (bt.symbols if bt.has("symbols") else list(cfg.trading.symbols))
    client = MexcClient(cfg.mexc)
    if raw == "all" or raw == ["all"]:
        symbols = client.all_usdt_contracts()
        print(f"MEXC USDT perpetual contracts: {len(symbols)}", flush=True)
    else:
        symbols = raw.split(",") if isinstance(raw, str) else list(raw)
    feats, btc = load_features(cfg, client, symbols, start, end)
    if not feats:
        raise SystemExit("No candles were downloaded for any symbol (check network access to MEXC / dates / symbols).")
    print(f"loaded {len(feats)} symbols: " + ", ".join(f"{k}={len(v.ts)} bars from {int(v.ts[0])}" for k, v in sorted(feats.items())))
    use_hist = bt.use_mexc_funding_history and len(feats) <= bt.funding_history_max_symbols
    if bt.use_mexc_funding_history and not use_hist:
        print(f"{len(feats)} symbols > {bt.funding_history_max_symbols}: using the fixed funding rate", flush=True)
    funding = {s: get_funding(client, bt.data_dir, s, refresh=True) for s in feats} if use_hist else {}

    def run(c):
        return run_backtest(c, feats, btc, start, end, funding)

    base = run(cfg)
    info = write_report(cfg, run, base, start, end, args.out or bt.reports_dir, symbols=sorted(feats))
    print(f"report: {info['report']}  trades: {info['trades']}")


def run_live_cli(cfg, args) -> None:
    from .live import run_live
    run_live(cfg, once=args.once)
