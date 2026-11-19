"""Re-simulate a long with an initial stop = entry - D that trails (highest high - D), using ordered 1-minute candles. Read-only.
Usage: python tools/resim_trail.py SYMBOL START_EPOCH ENTRY D [label]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def simulate(candles, entry, dist):
    """candles: iterable of (t, high, low) in time order. -> (status, stop, highest, exit_time or None, exit_px or None)."""
    stop, hh = entry - dist, entry
    for t, hi, lo in candles:
        if lo <= stop:
            return "ضُرب", stop, hh, t, stop
        hh = max(hh, hi)
        stop = max(stop, hh - dist)
    return "سارية", stop, hh, None, None


def main():
    from tradebot.config import load_config
    from tradebot.mexc.client import MexcClient

    sym, t0, entry, dist = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    cl = MexcClient(load_config("config.toml").mexc)
    now = int(time.time())
    k = cl.klines(sym, "Min1", t0, now)
    candles = [c for c in sorted(zip(k["time"], k["high"], k["low"])) if c[0] >= t0]
    st, stop, hh, te, px = simulate(candles, entry, dist)
    last = cl.last_price(sym)
    f = lambda x: time.strftime("%H:%M", time.gmtime(x))
    label = sys.argv[5] if len(sys.argv) > 5 else sym
    if st == "ضُرب":
        print(f"{label}: ضُرب الوقف {px:.6g} عند {f(te)} UTC · نتيجة {(px / entry - 1) * 100:+.1f}% · أعلى سعر قبله {(hh / entry - 1) * 100:+.1f}% · السعر الآن {last:.6g} ({(last / entry - 1) * 100:+.1f}%)")
    else:
        print(f"{label}: سارية · السعر الآن {last:.6g} ({(last / entry - 1) * 100:+.1f}%) · أعلى سعر {hh:.6g} ({(hh / entry - 1) * 100:+.1f}%) · الوقف المتحرك الآن {stop:.6g} ({(stop / entry - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
