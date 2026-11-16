"""Test of the channel-screenshot logic (rules fixed in docs/CHANNEL_LOGIC.md). Long only.
Usage: python -m tradebot.strategies.mtf_entry.channel_logic [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from tradebot.config import load_config
from tradebot.data.klines import drop_unclosed, fetch_range
from tradebot.mexc.client import TF_SECONDS, MexcClient

from .paper import NON_CRYPTO

FRAMES = [("Min5", 30), ("Min15", 90), ("Min60", 365), ("Hour4", 730)]
H, ATR_MULT, TP_R, COST, FRESH, SLOPE = 48, 1.5, 2.0, 0.002, 40, 3
NAMES = {"Min5": "5 دقائق", "Min15": "15 دقيقة", "Min60": "ساعة", "Hour4": "4 ساعات"}


def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def atr14(h, l, c, n=14):
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.full(len(c), np.nan)
    if len(c) > n:
        a = tr[1:n + 1].mean()
        out[n] = a
        for i in range(n + 1, len(c)):
            a = (a * (n - 1) + tr[i]) / n
            out[i] = a
    return out


def setups(c):
    """Boolean arrays (S, B1) per bar using only data up to the bar."""
    n = len(c)
    m5, m10, m30, m60 = (sma(c, k) for k in (5, 10, 30, 60))

    def lag(x, k):
        y = np.full(n, np.nan)
        y[k:] = x[:-k]
        return y

    with np.errstate(invalid="ignore"):
        stack = (m10 > m30) & (m30 > m60) & (m5 > m30)
        slope = (m30 > lag(m30, SLOPE)) & (m60 >= lag(m60, SLOPE))
        cross = np.zeros(n, bool)
        cross[1:] = (m10[1:] > m30[1:]) & (m10[:-1] <= m30[:-1])
        cs = np.cumsum(cross)
        fresh = (cs - np.concatenate((np.zeros(FRESH + 1, int), cs[:-FRESH - 1]))[:n]) >= 1
        b1 = stack & slope & fresh
        pull = (c < m5) & (c < m10) & (c > m30)
        s = b1 & pull
        s[1:] &= ~(b1[:-1] & pull[:-1])
    return s, b1


def outcomes(o, h, l, c, a, idx):
    """Net R for entries at bar indices idx (first-touch within H bars; stop wins ties)."""
    n = len(c)
    idx = idx[(idx + H < n) & ~np.isnan(a[idx])]
    if len(idx) == 0:
        return idx, np.array([]), np.array([])
    fh, fl = swv(h[1:], H)[idx], swv(l[1:], H)[idx]
    e = c[idx]
    risk = ATR_MULT * a[idx]
    sl, tp = e - risk, e + TP_R * risk
    hs, ht = fl <= sl[:, None], fh >= tp[:, None]
    fs = np.where(hs.any(1), hs.argmax(1), H + 1)
    ft = np.where(ht.any(1), ht.argmax(1), H + 1)
    win = ft < fs
    stop = fs <= ft
    exit_px = np.where(win, tp, np.where(stop, sl, c[idx + H]))
    r = (exit_px - e) / risk - COST * e / risk
    return idx, r, win


def boot(day, r, reps=1000, seed=7):
    if len(r) < 2:
        return float("nan"), float("nan")
    u, inv = np.unique(day, return_inverse=True)
    sums, cnts = np.bincount(inv, r), np.bincount(inv)
    rng = np.random.default_rng(seed)
    ms = np.empty(reps)
    for k in range(reps):
        pick = rng.integers(0, len(u), len(u))
        ms[k] = sums[pick].sum() / cnts[pick].sum()
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config("config.toml")
    cl, now = MexcClient(cfg.mexc), int(time.time())
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 300_000]
    lines = [f"اختبار منطق القناة (MA5/10/30/60 رجوع بعد كسر) · {len(tick)} عقداً · القواعد: docs/CHANNEL_LOGIC.md", ""]
    for tf, days in FRAMES:
        sec = TF_SECONDS[tf]
        rec = {"S": [], "B1": [], "B0": []}
        nsym = 0
        for t in tick:
            sym = t["symbol"]
            try:
                k = drop_unclosed(fetch_range(cl, sym, tf, int(now // sec * sec) - days * 86400 - 100 * sec, now), tf, now)
                o, h, l, c, tt = (np.asarray(k[f], float) for f in ("open", "high", "low", "close", "time"))
                if len(c) < 200:
                    continue
                a = atr14(h, l, c)
                s, b1 = setups(c)
                nsym += 1
                for name, mask in (("S", s), ("B1", b1), ("B0", np.arange(len(c)) % 7 == 0)):
                    idx, r, w = outcomes(o, h, l, c, a, np.where(mask)[0])
                    if len(idx):
                        rec[name].append((tt[idx], r, w))
            except Exception as e:
                print("skip", sym, e, file=sys.stderr)
        allt = np.concatenate([x[0] for v in rec.values() for x in v]) if any(rec.values()) else np.array([])
        if len(allt) == 0:
            lines.append(f"{NAMES[tf]}: لا بيانات")
            continue
        cut = allt.min() + (allt.max() - allt.min()) * 2 / 3
        lines.append(f"== {NAMES[tf]} · {nsym} عقداً · {days} يوماً · فاصل التحقق {time.strftime('%Y-%m-%d', time.gmtime(cut))}")
        lines.append(f"{'':4}{'الفترة':8}{'إشارات':>8}{'بلوغ الهدف':>12}{'متوسط R':>10}{'حد 95%':>20}")
        res = {}
        for name in ("S", "B1", "B0"):
            if not rec[name]:
                continue
            T = np.concatenate([x[0] for x in rec[name]]); R = np.concatenate([x[1] for x in rec[name]]); W = np.concatenate([x[2] for x in rec[name]])
            for part, m in (("تصميم", T < cut), ("تحقق", T >= cut)):
                if m.sum() == 0:
                    continue
                lo, hi = boot((T[m] // 86400).astype(int), R[m])
                res[(name, part)] = (int(m.sum()), R[m].mean(), lo)
                lines.append(f"{name:4}{part:8}{int(m.sum()):>8}{W[m].mean() * 100:>11.1f}%{R[m].mean():>+10.3f}   [{lo:+.3f}, {hi:+.3f}]")
        s, b1, b0 = res.get(("S", "تحقق")), res.get(("B1", "تحقق")), res.get(("B0", "تحقق"))
        ok = bool(s and s[0] >= 100 and s[1] > 0 and s[2] > 0 and b1 and s[1] > b1[1] and b0 and s[1] > b0[1])
        lines.append(f"الحكم المسجّل: {'نجح' if ok else 'فشل'}" + ("" if s else " (لا إشارات في التحقق)"))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w").write(text)


if __name__ == "__main__":
    main()
