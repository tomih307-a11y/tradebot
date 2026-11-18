"""Apply the fixed channel MA5/10/30/60 rule (docs/CHANNEL_LOGIC.md) to S&P 500 data from Yahoo JSON files.
Usage: python tools/spx_ma.py DAILY.json HOURLY.json M15.json   (no tuning: same rule, ATR*1.5 stop, 2R target, 48-bar horizon; cost 0.02%)"""
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tradebot.strategies.mtf_entry import channel_logic as CL  # noqa: E402

CL.COST = 0.0002


def load(path):
    r = json.load(open(path))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    rows = [(t, o, h, l, c) for t, o, h, l, c in zip(r["timestamp"], q["open"], q["high"], q["low"], q["close"]) if None not in (o, h, l, c)]
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]


def setups_short(c):
    n = len(c)
    m5, m10, m30, m60 = (CL.sma(c, k) for k in (5, 10, 30, 60))

    def lag(x, k):
        y = np.full(n, np.nan); y[k:] = x[:-k]; return y

    with np.errstate(invalid="ignore"):
        stack = (m10 < m30) & (m30 < m60) & (m5 < m30)
        slope = (m30 < lag(m30, 3)) & (m60 <= lag(m60, 3))
        cross = np.zeros(n, bool)
        cross[1:] = (m10[1:] < m30[1:]) & (m10[:-1] >= m30[:-1])
        cs = np.cumsum(cross)
        fresh = (cs - np.concatenate((np.zeros(CL.FRESH + 1, int), cs[:-CL.FRESH - 1]))[:n]) >= 1
        b1 = stack & slope & fresh
        pull = (c > m5) & (c > m10) & (c < m30)
        s = b1 & pull
        s[1:] &= ~(b1[:-1] & pull[:-1])
    return s, b1


def outcomes_short(h, l, c, a, idx):
    n, H = len(c), CL.H
    idx = idx[(idx + H < n) & ~np.isnan(a[idx])]
    if len(idx) == 0:
        return idx, np.array([]), np.array([])
    fh, fl = swv(h[1:], H)[idx], swv(l[1:], H)[idx]
    e = c[idx]; risk = CL.ATR_MULT * a[idx]
    sl, tp = e + risk, e - CL.TP_R * risk
    hs, ht = fh >= sl[:, None], fl <= tp[:, None]
    fs = np.where(hs.any(1), hs.argmax(1), H + 1); ft = np.where(ht.any(1), ht.argmax(1), H + 1)
    win = ft < fs
    px = np.where(win, tp, np.where(fs <= ft, sl, c[idx + H]))
    return idx, (e - px) / risk - CL.COST * e / risk, win


def state(name, t, o, h, l, c):
    m = {k: CL.sma(c, k)[-1] for k in (5, 10, 30, 60)}
    sl, b1l = CL.setups(c); ss, b1s = setups_short(c)
    stack = "صاعد" if m[10] > m[30] > m[60] and m[5] > m[30] else ("هابط" if m[10] < m[30] < m[60] and m[5] < m[30] else "مختلط")
    return (f"{name}: إغلاق {c[-1]:.1f} · MA5 {m[5]:.0f} · MA10 {m[10]:.0f} · MA30 {m[30]:.0f} · MA60 {m[60]:.0f} · ترتيب المتوسطات: {stack} · "
            f"إشارة شراء (S) الآن: {'نعم' if sl[-1] else 'لا'} · إشارة بيع (مرآة) الآن: {'نعم' if ss[-1] else 'لا'} · "
            f"ترتيب هابط حديث (مرآة B1) في آخر 5 شموع: {'نعم' if b1s[-5:].any() else 'لا'} · ترتيب صاعد حديث: {'نعم' if b1l[-5:].any() else 'لا'}")


def test(name, t, o, h, l, c):
    a = CL.atr14(h, l, c)
    out = [f"== {name}: {len(c)} شمعة"]
    cut = t[0] + (t[-1] - t[0]) * 2 / 3
    for side, (s, b1) in (("شراء", CL.setups(c)), ("بيع", setups_short(c))):
        for lab, mask in (("S", s), ("B1", b1), ("B0", np.arange(len(c)) % 7 == 0)):
            idx = np.where(mask)[0]
            idx, r, w = (CL.outcomes(o, h, l, c, a, idx) if side == "شراء" else outcomes_short(h, l, c, a, idx))
            if not len(idx):
                continue
            for part, m in (("تصميم", t[idx] < cut), ("تحقق", t[idx] >= cut)):
                if m.sum() < 2:
                    continue
                lo, hi = CL.boot((t[idx][m] // 86400).astype(int), r[m])
                out.append(f"  {side} {lab:<3}{part} n={int(m.sum()):>5} بلوغ الهدف {w[m].mean() * 100:4.1f}% متوسط R {r[m].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return out


if __name__ == "__main__":
    d, hh, m15 = (load(p) for p in sys.argv[1:4])
    names = [("يومي", d), ("ساعة", hh), ("15 دقيقة", m15)]
    print("الحالة الآن:"); [print(" ", state(n, *x)) for n, x in names]
    print()
    for n, x in names:
        print("\n".join(test(n, *x)))
