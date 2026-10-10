"""Test #6: identical results on identical data (trades, report and CSV byte-for-byte)."""
from pathlib import Path

from conftest import ROOT, ovr
from synth import synth_klines
from tradebot.config import load_config, parse_time
from tradebot.strategies.pmo_zero_cross.backtest import run_backtest
from tradebot.strategies.pmo_zero_cross.features import build_features
from tradebot.strategies.pmo_zero_cross.report import write_report


def _world(cfg):
    s = cfg.pmo_zero_cross
    feats = {}
    for i, name in enumerate(["BTC_USDT", "AAA_USDT", "BBB_USDT"]):
        k15, k1, k4 = synth_klines(10 + i, 150, start=1_700_006_400 // 14400 * 14400)
        feats[name] = build_features(name, k15, k1, k4, s, liquidity_window=s.backtest.liquidity_window_bars)
    start = int(feats["BTC_USDT"].ts[0]) + 70 * 86400
    end = int(feats["BTC_USDT"].ts[-1])
    return feats, start, end


def test_backtest_twice_identical(tmp_path):
    cfg = load_config(ROOT / "config.toml")
    feats, start, end = _world(cfg)
    btc = feats["BTC_USDT"]
    run = lambda c: run_backtest(c, feats, btc, start, end)
    outs = []
    for n in (1, 2):
        eng = run(cfg)
        info = write_report(cfg, run, eng, start, end, tmp_path / f"r{n}")
        outs.append((eng, tmp_path / f"r{n}"))
    (e1, d1), (e2, d2) = outs
    assert len(e1.closed) > 20
    assert [t.to_dict() for t in e1.closed] == [t.to_dict() for t in e2.closed]
    for name in ("pmo_report.md", "pmo_trades.csv", "pmo_rejected.csv", "last_backtest_cells.json"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes(), name
