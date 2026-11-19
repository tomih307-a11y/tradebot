"""Draw 15m charts (candles + TEMA50 + order blocks alive at the PMO zero-touch bar, MACD, PMO) for given coins and times.
Usage: python3 chart_multi.py SYMBOL:DD:HH:MM ...   (time = UTC+4 on the given day of November 2026)"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import luxalgo_ob as LX

TZ = 4


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))["data"]


def ema(x, n):
    a, e, out = 2 / (n + 1), x[0], []
    for v in x:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


def smooth(x, n):
    a, e, out = 2 / n, x[0], []
    for v in x:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


def tema(x, n):
    e1 = ema(x, n); e2 = ema(e1, n); e3 = ema(e2, n)
    return [3 * a - 3 * b + c for a, b, c in zip(e1, e2, e3)]


dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)
report = []
for spec in sys.argv[1:]:
    n, sym, md, hh, mm = spec.split(":")
    month, dd = (int(x) for x in md.split("-"))
    now = int(time.time())
    d = {k: [] for k in ("time", "open", "high", "low", "close")}
    a = now - 80 * 86400
    while a < now:                                                      # paged fetch (1000 candles per request)
        part = get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min15&start={a}&end={min(a + 900000, now)}")
        for k in d:
            d[k] += part[k]
        a += 900000
    seen = set(); idx = [i for i, tt in enumerate(d["time"]) if not (tt in seen or seen.add(tt))]
    d = {k: [d[k][i] for i in idx] for k in d}
    keep = [i for i, tt in enumerate(d["time"]) if tt + 900 <= now]
    t, o, h, l, c = ([d[k][i] for i in keep] for k in ("time", "open", "high", "low", "close"))
    roc = [0.0] + [(c[i] / c[i - 1] - 1) * 100 for i in range(1, len(c))]
    pmo = smooth([10 * v for v in smooth(roc, 35)], 20)
    sig = ema(pmo, 10)
    macd = [a - b for a, b in zip(ema(c, 12), ema(c, 26))]
    msig = ema(macd, 9)
    hist = [a - b for a, b in zip(macd, msig)]
    t50 = tema(c, 50)
    obs, snaps = LX.run(o, h, l, c)
    want = datetime(2026, month, int(dd), int(hh), int(mm), tzinfo=timezone.utc) - timedelta(hours=TZ)
    crosses = [i for i in range(1, len(c)) if pmo[i - 1] <= 0 < pmo[i]]
    cross = min(crosses, key=lambda i: abs(t[i] - want.timestamp()))
    i0 = max(cross - 80, 0)
    end = cross + 4                                                   # blind: only 3 candles after the touch
    t, o, h, l, c, pmo, sig, macd, msig, hist, t50 = (x[:end] for x in (t, o, h, l, c, pmo, sig, macd, msig, hist, t50))
    alive = snaps[cross]
    inside = [ob for ob in alive if ob["bias"] == 1 and ob["lo"] <= c[cross] <= ob["hi"]]
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    for k, i in enumerate(range(i0, len(c))):
        col = "#26a69a" if c[i] >= o[i] else "#ef5350"
        ax[0].plot([k, k], [l[i], h[i]], color=col, lw=0.8)
        ax[0].add_patch(Rectangle((k - 0.35, min(o[i], c[i])), 0.7, abs(c[i] - o[i]) or 1e-12, color=col))
    xs = range(len(c) - i0)
    ax[0].plot(xs, t50[i0:], color="#2962ff", lw=1.5, label="TEMA 50")
    for ob in alive:
        x0 = max(ob["t"] - i0, 0)
        ax[0].add_patch(Rectangle((x0, ob["lo"]), len(c) - i0 - x0, ob["hi"] - ob["lo"], color="#3179f5" if ob["bias"] == 1 else "#f77c80", alpha=0.3, lw=0))
    ax[1].plot(xs, macd[i0:], color="#2962ff", lw=1.1); ax[1].plot(xs, msig[i0:], color="#ff9800", lw=1.1)
    ax[1].bar(xs, hist[i0:], color=["#26a69a" if v >= 0 else "#ef5350" for v in hist[i0:]], width=0.8); ax[1].axhline(0, color="gray", ls="--", lw=0.8)
    ax[2].plot(xs, pmo[i0:], color="#2962ff", lw=1.3); ax[2].plot(xs, sig[i0:], color="#ff9800", lw=1.3); ax[2].axhline(0, color="gray", ls="--", lw=0.8)
    for a in ax:
        a.axvline(cross - i0, color="orange", lw=1.8); a.grid(alpha=0.2)
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title(f"case {n} | {sym} 15m | PMO(35,20,10) zero-touch up at {dt(t[cross]):%d %b %H:%M} (UTC+4) | blue = bullish OB alive then, red = bearish OB", fontsize=10)
    ticks = list(range(0, len(c) - i0, 12))
    ax[2].set_xticks(ticks); ax[2].set_xticklabels([f"{dt(t[i0 + k]):%d %H:%M}" for k in ticks], rotation=45, fontsize=8)
    plt.tight_layout(); plt.savefig(f"case_{n}.png", dpi=105); plt.close()
    report.append(f"{n} {sym}: لمس PMO للصفر {dt(t[cross]):%d %H:%M} (UTC+4) · إغلاق {c[cross]:.8g} · " +
                  ("داخل أوردر بلوك صاعد [" + f"{inside[0]['lo']:.8g}-{inside[0]['hi']:.8g}]" if inside else "ليس داخل أي أوردر بلوك صاعد") +
                  " · أوردر بلوك حي: " + "؛ ".join(f"{'صاعد' if ob['bias'] == 1 else 'هابط'} [{ob['lo']:.8g}-{ob['hi']:.8g}]" for ob in alive) +
                  f" · المكدي {'فوق' if macd[cross] > msig[cross] else 'تحت'} الإشارة · {'فوق' if c[cross] > t50[cross] else 'تحت'} TEMA50")
open("charts.txt", "w", encoding="utf-8").write("\n".join(report))
print("\n".join(report))
