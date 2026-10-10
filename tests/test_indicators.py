"""Section 1 formulas + test #1 (TradingView comparison, runs when you supply the reference files)."""
import csv
from pathlib import Path

import numpy as np
import pytest

from tradebot.indicators import atr_rma, csf, ema, pmo, roc

DATA = Path(__file__).parent / "data"


def ref_pmo(close, n1=35, scale=10, n2=20, sig=10):
    """Literal, loop-by-loop transcription of the spec (independent of the numpy code)."""
    r = [None] + [100 * (close[t] - close[t - 1]) / close[t - 1] for t in range(1, len(close))]
    def smooth(x, n):
        out, prev = [None] * len(x), None
        for t, v in enumerate(x):
            if v is None:
                continue
            prev = v if prev is None else prev + (v - prev) * (2 / n)
            out[t] = prev
        return out
    p = smooth([None if v is None else scale * v for v in smooth(r, n1)], n2)
    e, prev = [None] * len(p), None
    for t, v in enumerate(p):
        if v is None:
            continue
        prev = v if prev is None else prev + (v - prev) * (2 / (sig + 1))
        e[t] = prev
    return p, e


def test_formulas_match_spec_reference():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
    p, s, sl = pmo(c, 35, 10, 20, 10)
    rp, rs = ref_pmo(list(c))
    assert np.allclose(p[1:], rp[1:], atol=1e-12) and np.allclose(s[1:], rs[1:], atol=1e-12)
    assert np.allclose(sl[2:], p[2:] - p[1:-1], atol=1e-15)


def test_csf_uses_2_over_n_not_2_over_n_plus_1():
    out = csf(np.array([1.0, 2.0, 2.0]), 35)
    assert out[1] == pytest.approx(1 + (2 - 1) * 2 / 35)
    assert out[1] != pytest.approx(1 + (2 - 1) * 2 / 36)


def test_ema_alpha():
    out = ema(np.array([1.0, 2.0]), 10)
    assert out[1] == pytest.approx(1 + 2 / 11)


def test_roc():
    assert roc(np.array([100.0, 110.0]))[1] == pytest.approx(10.0)


def test_atr_tradingview_rma():
    h = np.array([10, 11, 12, 13.0]); l = np.array([9, 9.5, 10, 12.0]); c = np.array([9.5, 10.5, 11, 12.5])
    tr = [1.0, max(1.5, 1.5, 0.0), max(2.0, 1.5, 0.5), max(1.0, 2.0, 1.0)]
    a = atr_rma(h, l, c, 3)
    assert np.isnan(a[1]) and a[2] == pytest.approx(np.mean(tr[:3]))
    assert a[3] == pytest.approx((a[2] * 2 + tr[3]) / 3)


@pytest.mark.parametrize("tf,tvfile", [("Min15", "tv_btc_15m.csv"), ("Min60", "tv_btc_60m.csv")])
def test_matches_tradingview_btc(tf, tvfile):
    """Needs: tests/data/<tvfile> (time,pmo,signal; last 20+ rows) and cached MEXC klines
    data/cache/BTC_USDT_<tf>.csv (>=1000 candles ending at the same candle). Skips otherwise."""
    tv, cache = DATA / tvfile, Path(__file__).parents[1] / "data/cache" / f"BTC_USDT_{tf}.csv"
    if not (tv.exists() and cache.exists()):
        pytest.skip("TradingView reference / cached BTC candles not provided yet")
    from tradebot.data.klines import _load_cache
    k = _load_cache(cache)
    p, s, _ = pmo(k["close"], 35, 10, 20, 10)
    ours = {int(t): (a, b) for t, a, b in zip(k["time"], p, s)}
    rows = list(csv.DictReader(open(tv)))[-20:]
    assert len(rows) == 20
    for r in rows:
        a, b = ours[int(r["time"])]
        assert abs(a - float(r["pmo"])) <= 0.01 and abs(b - float(r["signal"])) <= 0.01
