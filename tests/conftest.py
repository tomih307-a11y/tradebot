import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.config import Section, load_config                      # noqa: E402
from tradebot.strategies.pmo_zero_cross.backtest import BacktestExecutor  # noqa: E402
from tradebot.strategies.pmo_zero_cross.costs import CostModel         # noqa: E402
from tradebot.strategies.pmo_zero_cross.engine import PMOEngine        # noqa: E402
from tradebot.strategies.pmo_zero_cross.features import SymbolFeatures  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def ovr(cfg, **kw):
    """ovr(cfg, **{'pmo_zero_cross.trail.mult_1': 2.0})"""
    d = cfg.to_dict()
    for path, v in kw.items():
        cur = d
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = v
    return Section(d, "root")


@pytest.fixture
def cfg():
    c = load_config(ROOT / "config.toml")
    return ovr(c, **{"pmo_zero_cross.backtest.slippage_pct": 0.0, "trading.max_open_trades": 5})


class FB:
    """Scripted feature builder in FAVORABLE units: values are multiplied by side sign `s`
    so one scenario covers long (s=+1) and short (s=-1) mirrored exactly."""

    def __init__(self, symbol="X", s=1, n=40):
        self.s, self.n, self.symbol = s, n, symbol
        z = lambda v: np.full(n, float(v))
        self.a = {"pmo15": z(-1), "signal15": z(0), "slope15": z(0),
                  "pmo1h": z(-1), "signal1h": z(0), "slope1h": z(1), "atr1h": z(2),
                  "pmo4h": z(-1), "signal4h": z(0), "slope4h": z(1), "atr15": z(1), "liq24": z(1e6)}
        for k in self.a:
            if k.startswith(("pmo", "slope", "signal")):
                self.a[k] = self.a[k] * s
        self.px = z(100)
        self.closed1h = np.zeros(n, dtype=bool)
        self.closed4h = np.zeros(n, dtype=bool)
        self.band = 0.1
        self.lows = None
        self.opens = None

    def set(self, name, sl, v):
        self.a[name][sl] = v * (self.s if name.startswith(("pmo", "slope", "signal")) else 1)
        return self

    def price(self, sl, x):
        self.px[sl] = x
        return self

    def cross_at(self, k):
        """15m cross in the trade direction at bar k, stays favorable afterwards."""
        self.set("pmo15", slice(0, k), -1).set("pmo15", slice(k, None), 1)
        return self

    def build(self) -> SymbolFeatures:
        s, n = self.s, self.n
        c = 100 + s * (self.px - 100)
        o = np.concatenate(([c[0]], c[:-1]))
        if self.opens:                               # explicit open (gap) in favorable units
            for i, v in self.opens.items():
                o[i] = 100 + s * (v - 100)
        hi, lo = np.maximum(o, c) + self.band, np.minimum(o, c) - self.band
        if self.lows:                                # {bar: (low_in_fav_units)} adverse wick
            for i, adv in self.lows.items():
                p = 100 + s * (adv - 100)
                if s == 1:
                    lo[i] = min(lo[i], p)
                else:
                    hi[i] = max(hi[i], p)
        ts = np.arange(n, dtype=np.int64) * 900
        a = self.a
        return SymbolFeatures(
            symbol=self.symbol, ts=ts, open=o, high=hi, low=lo, close=c, liq24=a["liq24"],
            atr15=a["atr15"], pmo15=a["pmo15"], signal15=a["signal15"], slope15=a["slope15"],
            pmo1h=a["pmo1h"], signal1h=a["signal1h"], slope1h=a["slope1h"], atr1h=a["atr1h"],
            closed1h=self.closed1h, j1h=np.zeros(n, dtype=int),
            pmo4h=a["pmo4h"], signal4h=a["signal4h"], slope4h=a["slope4h"],
            closed4h=self.closed4h, j4h=np.zeros(n, dtype=int))


def make_engine(cfg, **kw):
    return PMOEngine(cfg, BacktestExecutor(cfg), CostModel(cfg), **kw)


def step(eng, feats, k, btc=None):
    eng.process_bar(k * 900, feats if isinstance(feats, dict) else {feats.symbol: feats}, btc)


def fav(s, x):
    """price in favorable units -> real price"""
    return 100 + s * (x - 100)
