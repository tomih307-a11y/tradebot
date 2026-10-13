"""RSI / Stochastic RSI against literal pure-python references (TradingView definitions)."""
import numpy as np

from tradebot.indicators import rma, rsi, stoch_rsi


def ref_rsi(c, n):
    out = [None] * len(c)
    ups = [max(c[i] - c[i - 1], 0) for i in range(1, len(c))]
    dns = [max(c[i - 1] - c[i], 0) for i in range(1, len(c))]
    au, ad = sum(ups[:n]) / n, sum(dns[:n]) / n
    out[n] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    for i in range(n, len(ups)):
        au, ad = (au * (n - 1) + ups[i]) / n, (ad * (n - 1) + dns[i]) / n
        out[i + 1] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def test_rsi_matches_reference():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    a, b = rsi(c, 14), ref_rsi(list(c), 14)
    assert all(np.isnan(a[i]) for i in range(14))
    assert np.allclose(a[14:], b[14:], atol=1e-10)
    assert (a[14:] >= 0).all() and (a[14:] <= 100).all()


def test_rma_seed_is_sma():
    x = np.arange(1.0, 11.0)
    r = rma(x, 4)
    assert np.isnan(r[2]) and r[3] == np.mean(x[:4]) and r[4] == (r[3] * 3 + x[4]) / 4


def test_stoch_rsi_matches_reference():
    rng = np.random.default_rng(4)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    k, d = stoch_rsi(c, 14, 14, 3, 3)
    r = np.array([np.nan if v is None else v for v in ref_rsi(list(c), 14)])
    st = np.full(len(r), np.nan)
    for i in range(len(r)):
        w = r[i - 13:i + 1]
        if i >= 13 and not np.isnan(w).any() and w.max() > w.min():
            st[i] = 100 * (r[i] - w.min()) / (w.max() - w.min())
    kk = np.array([np.nan if i < 2 else np.mean(st[i - 2:i + 1]) for i in range(len(st))])
    dd = np.array([np.nan if i < 2 else np.mean(kk[i - 2:i + 1]) for i in range(len(kk))])
    ok = ~np.isnan(kk) & ~np.isnan(k)
    assert ok.sum() > 200 and np.allclose(k[ok], kk[ok], atol=1e-9)
    ok = ~np.isnan(dd) & ~np.isnan(d)
    assert np.allclose(d[ok], dd[ok], atol=1e-9)
    assert (k[~np.isnan(k)] >= -1e-9).all() and (k[~np.isnan(k)] <= 100 + 1e-9).all()
