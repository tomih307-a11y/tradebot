"""Read-only status of a recommendation from 1-minute MEXC candles since a start time.
Usage: python tools/level_status.py SYMBOL START_EPOCH ENTRY STOP TP1,TP2,... [long|short]"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from tradebot.config import load_config
    from tradebot.data.klines import fetch_range
    from tradebot.mexc.client import MexcClient

    sym, t0, entry, stop = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    tps = [float(x) for x in sys.argv[5].split(",")]
    long_ = (sys.argv[6] if len(sys.argv) > 6 else "long") == "long"
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    k = fetch_range(cl, sym, "Min1", t0 - 60, now)
    t, o, h, l, c = (np.asarray(k[x], float) for x in ("time", "open", "high", "low", "close"))
    m = t >= t0 - 60
    t, o, h, l, c = t[m], o[m], h[m], l[m], c[m]
    f = lambda x: time.strftime("%m-%d %H:%M", time.gmtime(x)) + " UTC"
    print(f"{sym}: {len(c)} شمعة دقيقة من {f(t[0])} إلى {f(t[-1])} · الدخول المرجعي {entry} · الآن {c[-1]:.6g} ({(c[-1] / entry - 1) * 100:+.1f}%)")
    print(f"أعلى سعر {h.max():.6g} ({(h.max() / entry - 1) * 100:+.1f}%) عند {f(t[h.argmax()])} · أدنى سعر {l.min():.6g} ({(l.min() / entry - 1) * 100:+.1f}%) عند {f(t[l.argmin()])}")
    for i, tp in enumerate(tps, 1):
        hit = np.where(h >= tp)[0] if long_ else np.where(l <= tp)[0]
        print(f"TP{i} {tp}: " + (f"بلغه {f(t[hit[0]])}" if len(hit) else "لم يبلغه"))
    s = np.where(l <= stop)[0] if long_ else np.where(h >= stop)[0]
    print(f"الوقف {stop}: " + (f"لُمس {f(t[s[0]])}" if len(s) else "لم يُلمس"))
    first = [("TP", i) for i in range(len(tps)) if (h >= tps[i]).any()]
    print("آخر 6 إغلاقات ساعة:", [round(float(x), 6) for x in c[-360::60]][-6:])


if __name__ == "__main__":
    main()
