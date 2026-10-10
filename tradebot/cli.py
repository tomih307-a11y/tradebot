"""python -m tradebot <backtest|live|compare-tv> [options]"""
from __future__ import annotations

import argparse
import logging

from tradebot.config import load_config
from tradebot.strategies import load_strategy


def compare_tv(cfg, args) -> int:
    """Compare our PMO/SIGNAL (last N values) with a TradingView CSV export: columns time,pmo,signal."""
    import csv
    import time as _t

    import numpy as np

    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.indicators import pmo
    from tradebot.mexc.client import TF_SECONDS, MexcClient
    s = cfg.pmo_zero_cross
    ind = s.indicator
    client = MexcClient(cfg.mexc)
    now = _t.time()
    step = TF_SECONDS[args.tf]
    k = drop_unclosed(fetch_range(client, args.symbol, args.tf, int(now // step * step) - 1500 * step, int(now)), args.tf, now)
    p, sg, _ = pmo(k["close"], ind.roc_smooth_len, ind.roc_scale, ind.pmo_smooth_len, ind.signal_len)
    ours = {int(t): (a, b) for t, a, b in zip(k["time"], p, sg)}
    rows = list(csv.DictReader(open(args.file)))[-args.n:]
    worst = 0.0
    for r in rows:
        t = int(r["time"])
        a, b = ours[t]
        d = max(abs(a - float(r["pmo"])), abs(b - float(r["signal"])))
        worst = max(worst, d)
        print(f"{t} pmo ours={a:.4f} tv={float(r['pmo']):.4f}  signal ours={b:.4f} tv={float(r['signal']):.4f}  diff={d:.4f}")
    print(f"max diff = {worst:.5f}  ->  {'OK' if worst <= args.tol else 'FAIL'} (tol {args.tol})")
    return 0 if worst <= args.tol else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tradebot")
    ap.add_argument("--config", default="config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--start"); b.add_argument("--end"); b.add_argument("--symbols", help="comma separated")
    l = sub.add_parser("live")
    l.add_argument("--once", action="store_true", help="process one bar then exit")
    c = sub.add_parser("compare-tv")
    c.add_argument("--file", required=True); c.add_argument("--symbol", default="BTC_USDT")
    c.add_argument("--tf", default="Min15"); c.add_argument("--n", type=int, default=20)
    c.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    logging.basicConfig(level=getattr(logging, cfg.general.log_level), format="%(asctime)s %(name)s %(message)s")
    if args.cmd == "compare-tv":
        return compare_tv(cfg, args)
    strat = load_strategy(cfg.general.strategy)
    (strat.run_backtest_cli if args.cmd == "backtest" else strat.run_live_cli)(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
