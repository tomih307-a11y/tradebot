"""SMC strategy (docs/SMC_STRATEGY.md): 4H bias + order block in discount/premium + 15m liquidity sweep and CHoCH, stop beyond the sweep,
target at the nearest opposing 4H swing (reward/risk >= 2). `backtest` runs the registered universe validation; `live` scans for setups (paper: it never sends orders)."""
from __future__ import annotations


def run_backtest_cli(cfg, args) -> None:
    from tradebot.mexc.client import MexcClient

    from .backtest import run_universe

    out = getattr(args, "out", None) or cfg.pmo_zero_cross.backtest.reports_dir + "/smc_model"
    res = run_universe(cfg, MexcClient(cfg.mexc), days=150, out_dir=out)
    print(open(f"{out}/smc_model.md", encoding="utf-8").read())
    print({k: v["passed"] for k, v in res["sides"].items()})


def run_live_cli(cfg, args) -> None:
    from .scan import scan_loop

    scan_loop(cfg, once=getattr(args, "once", False))
