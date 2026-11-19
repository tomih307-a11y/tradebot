"""Draw LUNC 15m with TEMA50, MACD(12,26,9) and PMO(35,20,10) from MEXC data, mark the PMO zero-touch (up)."""
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SYM, TZ = "LUNC_USDT", 4          # label times as UTC+4 (the user's chart)


def klines(sym, interval, days):
    now = int(time.time())
    d = json.load(urllib.request.urlopen(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={interval}&start={now - days * 86400}&end={now}", timeout=60))["data"]
    return d["time"], d["open"], d["high"], d["low"], d["close"]


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


t, o, h, l, c = klines(SYM, "Min15", 14)
roc = [0.0] + [(c[i] / c[i - 1] - 1) * 100 for i in range(1, len(c))]
pmo = smooth([10 * v for v in smooth(roc, 35)], 20)
sig = ema(pmo, 10)
macd = [a - b for a, b in zip(ema(c, 12), ema(c, 26))]
msig = ema(macd, 9)
hist = [a - b for a, b in zip(macd, msig)]
t50 = tema(c, 50)
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)

start = int(datetime(2026, 11, 18, 4, 0, tzinfo=timezone.utc).timestamp())      # window like the user's screenshot
idx = [i for i in range(len(t)) if t[i] >= start]
i0, n = idx[0], len(idx)
crossings = [i for i in range(i0 + 1, len(t)) if pmo[i - 1] <= 0 < pmo[i]]
mcross = [i for i in range(i0 + 1, len(t)) if macd[i - 1] <= 0 < macd[i]]
pk = max(range(i0, len(t)), key=lambda i: h[i])
cross = [i for i in crossings if i < pk][-1]

fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1.3]})
for k, i in enumerate(range(i0, len(t))):
    col = "#26a69a" if c[i] >= o[i] else "#ef5350"
    ax[0].plot([k, k], [l[i], h[i]], color=col, lw=0.8)
    ax[0].add_patch(Rectangle((k - 0.35, min(o[i], c[i])), 0.7, abs(c[i] - o[i]) or 1e-9, color=col))
xs = range(len(t) - i0)
ax[0].plot(xs, t50[i0:], color="#2962ff", lw=1.6, label="TEMA 50")
ax[1].plot(xs, macd[i0:], color="#2962ff", lw=1.2, label="MACD"); ax[1].plot(xs, msig[i0:], color="#ff9800", lw=1.2, label="Signal")
ax[1].bar(xs, hist[i0:], color=["#26a69a" if v >= 0 else "#ef5350" for v in hist[i0:]], width=0.8); ax[1].axhline(0, color="gray", ls="--", lw=0.8)
ax[2].plot(xs, pmo[i0:], color="#2962ff", lw=1.4, label="PMO"); ax[2].plot(xs, sig[i0:], color="#ff9800", lw=1.4, label="Signal"); ax[2].axhline(0, color="gray", ls="--", lw=0.8)
for a in ax:
    a.axvline(cross - i0, color="orange", lw=1.8)
    a.grid(alpha=0.2)
    a.legend(loc="upper left", fontsize=8)
ax[0].set_title(f"{SYM} 15m | PMO(35,20,10) zero-touch up at {dt(t[cross]):%d %b %H:%M} (UTC+4) | MACD(12,26,9) | TEMA50", fontsize=11)
ticks = list(range(0, len(t) - i0, 16))
ax[2].set_xticks(ticks); ax[2].set_xticklabels([f"{dt(t[i0 + k]):%d %H:%M}" for k in ticks], rotation=45, fontsize=8)
plt.tight_layout(); plt.savefig("lunc_chart.png", dpi=110)

seg_h = max(h[cross:]); seg_l = min(l[i0:cross + 1])
lines = [f"LUNC 15m · PMO(35,20,10) صعود عبر الصفر عند {dt(t[cross]):%d %b %H:%M} (UTC+4) · إغلاق الشمعة {c[cross]:.8f}",
         f"قيمة PMO عندها {pmo[cross]:.3f} (إشارة {sig[cross]:.3f}) · المكدي {macd[cross]:.3e} (إشارة {msig[cross]:.3e})",
         f"أعلى سعر بعدها {seg_h:.8f} ({(seg_h / c[cross] - 1) * 100:+.1f}% من إغلاق الشمعة) عند {dt(t[pk]):%d %b %H:%M}",
         "عبور المكدي للصفر صعوداً في النافذة: " + "، ".join(f"{dt(t[i]):%d %H:%M}" for i in mcross),
         f"فوق TEMA50 وقت عبور PMO: {'نعم' if c[cross] > t50[cross] else 'لا'} (إغلاق {c[cross]:.8f} مقابل {t50[cross]:.8f})"]
open("chart.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
