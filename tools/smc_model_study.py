"""Run the registered SMC strategy validation on MEXC. Usage: python tools/smc_model_study.py --out reports/smc_model --days 150"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.config import load_config                                 # noqa: E402
from tradebot.mexc.client import MexcClient                              # noqa: E402
from tradebot.strategies.smc_strategy.backtest import run_universe       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/smc_model")
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--universe", type=int, default=5000)
    ap.add_argument("--config", default="config.toml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run_universe(cfg, MexcClient(cfg.mexc), days=a.days, universe=a.universe, out_dir=a.out, log=lambda m: print(m, flush=True))
    print(open(Path(a.out) / "smc_model.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
