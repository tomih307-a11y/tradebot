"""Read-only technical summary of one MEXC futures symbol. Usage: python tools/coin_ta.py SYMBOL"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def rsi(c, n=14):
    d = np.diff(c); g, l = np.maximum(d, 0), np.maximum(-d, 0)
    ag, al = g[:n].mean(), l[:n].mean()
    for i in range(n, len(d)):
        ag, al = (ag * (n - 1) + g[i]) / n, (al * (n - 1) + l[i]) / n
    return 100 - 100 / (1 + ag / al) if al else 100.0


def _ema(x, n):
    a, e = 2 / (n + 1), x[0]
    for v in x:
        e = a * v + (1 - a) * e
    return float(e)


def main():
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import TF_SECONDS, MexcClient

    sym = sys.argv[1]
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    f = lambda x: time.strftime("%m-%d %H:%M", time.gmtime(x))
    print(f"{sym} · {f(now)} UTC · السعر الآن {cl.last_price(sym)}")
    for tf, days in (("Day1", 120), ("Hour4", 60), ("Min60", 14), ("Min15", 4)):
        sec = TF_SECONDS[tf]
        k = drop_unclosed(fetch_range(cl, sym, tf, int(now // sec * sec) - days * 86400, now), tf, now)
        t, o, h, l, c, v = (np.asarray(k[x], float) for x in ("time", "open", "high", "low", "close", "vol"))
        if len(c) < 70:
            print(tf, "بيانات قليلة", len(c)); continue
        m = {n: c[-n:].mean() for n in (5, 10, 30, 60)}
        ema = lambda n: _ema(c, n)
        tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
        atr = tr[-14:].mean()
        w = 48 if tf in ("Min60", "Min15") else 30
        print(f"{tf}: إغلاق {c[-1]:.6g} · MA5 {m[5]:.6g} MA10 {m[10]:.6g} MA30 {m[30]:.6g} MA60 {m[60]:.6g} · EMA50 {ema(50):.6g} · RSI {rsi(c):.0f} · ATR {atr / c[-1] * 100:.1f}% · "
              f"أعلى {w} شمعة {h[-w:].max():.6g} · أدنى {l[-w:].min():.6g} · حجم آخر 5 ÷ وسيط 60: {v[-5:].mean() / max(np.median(v[-60:]), 1e-9):.1f}")
        if tf == "Day1":
            print("  آخر 10 أيام (إغلاق):", [round(float(x), 6) for x in c[-10:]], "· أعلى 30 يوماً", f"{h[-30:].max():.6g}", "· أدنى 30 يوماً", f"{l[-30:].min():.6g}")
        if tf == "Min60":
            print("  آخر 12 ساعة (إغلاق):", [round(float(x), 6) for x in c[-12:]])


if __name__ == "__main__":
    main()
