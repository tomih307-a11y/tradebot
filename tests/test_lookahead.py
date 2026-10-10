"""Test #2: higher-timeframe values at each 15m bar come from CLOSED candles only."""
import numpy as np
import pytest

from conftest import ROOT
from synth import synth_klines
from tradebot.config import load_config
from tradebot.data.klines import drop_unclosed
from tradebot.strategies.pmo_zero_cross.features import build_features

CFG = load_config(ROOT / "config.toml").pmo_zero_cross


def _cut(k, tf_sec, close_limit):
    m = (k["time"] + tf_sec) <= close_limit
    return {f: v[m] for f, v in k.items()}


FIELDS = ["pmo15", "signal15", "slope15", "atr15", "pmo1h", "signal1h", "slope1h", "atr1h",
          "pmo4h", "signal4h", "slope4h", "closed1h", "closed4h"]


@pytest.fixture(scope="module")
def data():
    k15, k1, k4 = synth_klines(3, 160)
    return k15, k1, k4, build_features("S", k15, k1, k4, CFG)


def test_aligned_candle_is_closed_before_the_15m_bar_ends(data):
    k15, k1, k4, f = data
    close15 = f.ts + 900
    for j, step, name in ((f.j1h, 3600, "1h"), (f.j4h, 14400, "4h")):
        k = k1 if name == "1h" else k4
        ok = j >= 0
        assert np.all(k["time"][j[ok]] + step <= close15[ok])           # closed no later than this bar
        nxt = j[ok] + 1
        has = nxt < len(k["time"])
        assert np.all(k["time"][nxt[has]] + step > close15[ok][has])    # and it is the LAST such candle


def test_hour_value_changes_only_on_the_bar_that_closes_it(data):
    _, _, _, f = data
    for arr, closed in ((f.pmo1h, f.closed1h), (f.pmo4h, f.closed4h)):
        chg = np.zeros(len(arr), dtype=bool)
        chg[1:] = (arr[1:] != arr[:-1]) & ~(np.isnan(arr[1:]) & np.isnan(arr[:-1]))
        assert not np.any(chg & ~closed)
    assert f.closed1h.sum() > 0 and f.closed4h.sum() > 0
    on_hour = ((f.ts + 900) % 3600 == 0)
    assert np.array_equal(f.closed1h[f.j1h >= 0], on_hour[f.j1h >= 0])


@pytest.mark.parametrize("cut", [9999, 11003, 12001, 13500, 15000])
def test_truncating_the_future_changes_nothing(data, cut):
    """Features at bars <= cut computed from data that ENDS at the cut equal those from the full data."""
    k15, k1, k4, full = data
    cut_close = int(full.ts[cut]) + 900
    f = build_features("S", _cut(k15, 900, cut_close), _cut(k1, 3600, cut_close), _cut(k4, 14400, cut_close), CFG)
    assert len(f.ts) == cut + 1
    for name in FIELDS:
        a, b = getattr(f, name), getattr(full, name)[:cut + 1]
        assert np.array_equal(a, b, equal_nan=True), name


def test_poisoning_the_future_changes_nothing(data):
    k15, k1, k4, full = data
    cut = 12345
    cut_close = int(full.ts[cut]) + 900
    def poison(k, sec):
        k = {f: v.copy() for f, v in k.items()}
        m = (k["time"] + sec) > cut_close
        for f in ("open", "high", "low", "close"):
            k[f][m] *= 7.0
        return k
    f = build_features("S", poison(k15, 900), poison(k1, 3600), poison(k4, 14400), CFG)
    for name in FIELDS:
        assert np.array_equal(getattr(f, name)[:cut + 1], getattr(full, name)[:cut + 1], equal_nan=True), name


def test_unclosed_candles_are_dropped_live(data):
    k15, *_ = data
    now = int(k15["time"][-1]) + 400                      # last candle is only 400s old
    d = drop_unclosed(k15, "Min15", now)
    assert len(d["time"]) == len(k15["time"]) - 1
    assert d["time"][-1] + 900 <= now
