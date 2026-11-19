"""SMC + MA + RSI + MACD confluence scanner for long setups entered from an order block (rules fixed in docs/QUALITY_SCAN.md).
Usage: python -m tradebot.strategies.mtf_entry.quality_scan [days] [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from tradebot.strategies.multi_agent.smc import order_block, structure

from .channel_logic import atr14
from .ob_entry import hourly
from .trail_exit import boot99

N_SWING, MAX_AGE, NEAR_ATR, BUF_ATR, HOLD, COST, COOL = 3, 48, 0.5, 0.25, 48, 0.002, 8
NAMES = ("اتجاه 4س", "كسر 1س", "أوردر بلوك", "متوسطات 1س", "متوسطات 4س", "RSI 40-65", "MACD")


def ema(x, n):
    a, e, out = 2 / (n + 1), x[0], np.empty(len(x))
    for i, v in enumerate(x):
        e = a * v + (1 - a) * e
        out[i] = e
    return out


def rsi(c, n=14):
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c)
    g, l = np.maximum(d, 0), np.maximum(-d, 0)
    ag, al = g[:n].mean(), l[:n].mean()
    out[n] = 100 - 100 / (1 + ag / al) if al else 100.0
    for i in range(n, len(d)):
        ag, al = (ag * (n - 1) + g[i]) / n, (al * (n - 1) + l[i]) / n
        out[i + 1] = 100 - 100 / (1 + ag / al) if al else 100.0
    return out


def four_hour(th, o, h, l, c):
    """4H candles from complete 1H candles (th = hour END time) -> (t_end, o, h, l, c)."""
    gid = ((th - 3600) // 14400).astype(np.int64)
    ids, first, cnt = np.unique(gid, return_index=True, return_counts=True)
    rows = [((ids[k] + 1) * 14400, o[first[k]], h[first[k]:first[k] + 4].max(), l[first[k]:first[k] + 4].min(), c[first[k] + 3])
            for k in range(len(ids)) if cnt[k] == 4]
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]


def components(th, o, h, l, c):
    """-> (comp [n,7] bool, ob_lo [n], atr [n]) per 1H bar, using only data up to each bar's close."""
    n = len(c)
    t4, o4, h4, l4, c4 = four_hour(th, o, h, l, c)
    _, ev4 = structure(c4, h4, l4, N_SWING)
    _, ev1 = structure(c, h, l, N_SWING)
    atr = atr14(h, l, c)
    e20, e50 = ema(c, 20), ema(c, 50)
    f20, f50 = ema(c4, 20), ema(c4, 50)
    r = rsi(c)
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    bull = [e for e in ev1 if e["dir"] == 1]
    obz = {id(e): order_block(e, o, h, l, c) for e in bull}
    bull_i = np.array([e["i"] for e in bull], int)
    comp = np.zeros((n, 7), bool)
    ob_lo = np.full(n, np.nan)
    ev4_i = np.array([e["i"] for e in ev4]) if ev4 else np.array([], int)
    ev4_d = np.array([e["dir"] for e in ev4]) if ev4 else np.array([], int)
    for i in range(60, n):
        j = int(np.searchsorted(t4, th[i], side="right")) - 1
        if j >= 60:
            k = int(np.searchsorted(ev4_i, j, side="right")) - 1
            comp[i, 0] = bool(k >= 0 and ev4_d[k] == 1)
            comp[i, 4] = bool(f20[j] > f50[j])
        kb = int(np.searchsorted(bull_i, i, side="right")) - 1
        if kb >= 0 and i - bull_i[kb] <= MAX_AGE:
            e = bull[kb]
            comp[i, 1] = True
            ob = obz[id(e)]
            if ob is not None and not np.isnan(atr[i]) and (c[e["i"] + 1:i + 1] >= ob[0]).all() and ob[0] <= c[i] <= ob[1] + NEAR_ATR * atr[i]:
                comp[i, 2] = True
                ob_lo[i] = ob[0]
        comp[i, 3] = bool(e20[i] > e50[i] and c[i] > e50[i])
        comp[i, 5] = bool(40 <= r[i] <= 65)
        comp[i, 6] = bool(macd[i] > sig[i])
    return comp, ob_lo, atr


def trade(o, h, l, c, i, dist):
    """Long entered at close c[i]; trailing stop = highest high - dist (starts at c[i]-dist); HOLD bars max. -> net R or None."""
    if i + HOLD >= len(c) or not np.isfinite(dist) or dist <= 0:
        return None
    e = c[i]
    hh, ll, oo = h[i + 1:i + 1 + HOLD], l[i + 1:i + 1 + HOLD], o[i + 1:i + 1 + HOLD]
    stop = np.maximum.accumulate(np.concatenate(([e], hh)))[:-1] - dist
    hit = np.where(ll <= stop)[0]
    px = min(stop[hit[0]], oo[hit[0]]) if len(hit) else c[i + HOLD]
    return float((px - e) / dist - COST * e / dist)


def setup_risk(c_i, ob_lo_i, atr_i):
    return max(c_i - (ob_lo_i - BUF_ATR * atr_i), 1.0 * atr_i)


def main():
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import MexcClient

    from .paper import NON_CRYPTO

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    out = sys.argv[2] if len(sys.argv) > 2 else None
    short = len(sys.argv) > 3 and sys.argv[3] == "short"               # price-inverse mirror (docs/QUALITY_SCAN.md addendum)
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 1_000_000]
    rows, base, nsym, live = [], [], 0, []     # rows: (tier, t, day, R, comp-vector); base: (t, day, R)
    for tk in tick:
        sym = tk["symbol"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - days * 96 * 900, now), "Min15", now)
            t, o15, h15, l15, c15 = (np.asarray(k[x], float) for x in ("time", "open", "high", "low", "close"))
            if len(c15) < 1500:
                continue
            th, o, h, l, c = hourly(t, o15, h15, l15, c15)
            if short:
                o, h, l, c = 1 / o, 1 / l, 1 / h, 1 / c
            comp, ob_lo, atr = components(th, o, h, l, c)
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
            continue
        nsym += 1
        score = comp.sum(axis=1)
        last = {"A": -10**9, "B": -10**9}
        for i in range(120, len(c)):
            if i % 7 == 0 and np.isfinite(atr[i]):
                r = trade(o, h, l, c, i, atr[i])
                if r is not None:
                    base.append((th[i], int(th[i] // 86400), r))
            if not comp[i, 2] or np.isnan(atr[i]):
                continue
            tier = "A" if score[i] >= 6 else ("B" if score[i] == 5 else None)
            if tier is None or i - last[tier] < COOL:
                continue
            last[tier] = i
            d = setup_risk(c[i], ob_lo[i], atr[i])
            r = trade(o, h, l, c, i, d)
            if r is not None:
                rows.append((tier, th[i], int(th[i] // 86400), r, comp[i].copy()))
            elif i >= len(c) - 3 and float(tk.get("amount24") or 0) >= 5e6:
                live.append((sym, tier, (len(c) - 1 - i), c[i], ob_lo[i], d, atr[i], comp[i].copy(), float(tk["amount24"])))
    cut = max(r[1] for r in rows) - 30 * 86400
    bh = [b for b in base if b[0] >= cut]
    bm = float(np.mean([b[2] for b in bh]))
    lines = [f"{'شورت (مرآة السعر) · ' if short else ''}نظام التقاء SMC/المتوسطات/RSI/MACD · {nsym} عقداً (سيولة ≥1M) · {days} يوماً · فاصل التحقق {time.strftime('%Y-%m-%d', time.gmtime(cut))} · القواعد: docs/QUALITY_SCAN.md", "",
             f"أساس عشوائي (تحقق): متوسط R {bm:+.3f} ({len(bh)} صفقة)",
             f"{'المستوى':<8}{'إشارات(تصميم/تحقق)':>22}{'/يوم':>7}{'متوسط R':>10}{'حد 99%':>20}{'رابحة':>8}  الحكم"]
    hold_days = max((max(r[1] for r in rows) - cut) / 86400, 1)
    for tier in ("A", "B"):
        d = [r for r in rows if r[0] == tier and r[1] < cut]
        h_ = [r for r in rows if r[0] == tier and r[1] >= cut]
        if len(h_) < 2:
            lines.append(f"{tier:<8}{len(d):>10}/{len(h_)}  غير حاسم (إشارات قليلة)"); continue
        R = np.array([r[3] for r in h_])
        lo, hi = boot99(np.array([r[2] for r in h_]), R)
        verdict = "غير حاسم (<100)" if len(h_) < 100 else ("نجح" if R.mean() > 0 and lo > 0 and R.mean() > bm else "رُفض")
        lines.append(f"{tier:<8}{len(d):>10}/{len(h_):<11}{len(h_) / hold_days:>7.2f}{R.mean():>+10.3f}   [{lo:+.3f}, {hi:+.3f}]{(R > 0).mean() * 100:>7.0f}%  {verdict}")
    allr = [r for r in rows if r[1] >= cut]
    if len(allr) >= 10:
        lines += ["", "وصفي: متوسط R في التحقق (A+B) حين يكون المكوّن غائباً مقابل حاضراً:"]
        for ci, nm in enumerate(NAMES):
            pres = [r[3] for r in allr if r[4][ci]]; ab = [r[3] for r in allr if not r[4][ci]]
            lines.append(f"  {nm:<12} حاضر {np.mean(pres):+.3f} ({len(pres)}) · غائب {np.mean(ab):+.3f} ({len(ab)})" if ab else f"  {nm:<12} حاضر دائماً ({len(pres)})")
    lines += ["", "إشارات حالية (آخر 3 ساعات، سيولة ≥ 5M):"]
    for sym, tier, age, px, lo_, d, a_, cp, tn in sorted(live, key=lambda x: (x[1], x[2])):
        stop = px - d
        if short:                                                     # back to real prices
            real_px, real_stop = 1 / px, 1 / stop
            lines.append(f"  {tier} شورت {sym:<14} منذ {age}س · سعر {real_px:.6g} · وقف {real_stop:.6g} (+{(real_stop / real_px - 1) * 100:.1f}%) وقف متحرك بالمسافة نفسها (تقريباً) · ATR1س {a_ / px * 100:.1f}% · "
                         f"سيولة {tn / 1e6:.1f}M · مكوّنات {int(cp.sum())}/7: " + "، ".join(nm for nm, v in zip(NAMES, cp) if v))
            continue
        lines.append(f"  {tier} {sym:<14} منذ {age}س · سعر {px:.6g} · أدنى OB {lo_:.6g} · وقف {stop:.6g} (−{d / px * 100:.1f}%) وقف متحرك بالمسافة نفسها · ATR1س {a_ / px * 100:.1f}% · "
                     f"سيولة {tn / 1e6:.1f}M · مكوّنات {int(cp.sum())}/7: " + "، ".join(nm for nm, v in zip(NAMES, cp) if v))
    if not live:
        lines.append("  لا شيء")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
