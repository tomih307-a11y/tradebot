"""Open-target trailing-stop exit on hunter alerts (rules fixed in docs/TRAIL_EXIT.md). Long only.
Usage: python -m tradebot.strategies.mtf_entry.trail_exit [days] [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from .channel_logic import atr14
from .early_signs import WARM, features
from .hunter_tiers import alerts, tier_of
from .immediate_move import alert_features

HMAX, COST, KS = 192, 0.002, (2, 3)


def trail_trade(o, h, l, c, i, risk):
    """Simulate a long entered at close c[i]; trailing stop = highest high so far - risk. -> (net R, pct of entry, bars, exit_px) or None."""
    n = len(c)
    if i + HMAX >= n or not np.isfinite(risk) or risk <= 0:
        return None
    e = c[i]
    hh, ll, oo = h[i + 1:i + 1 + HMAX], l[i + 1:i + 1 + HMAX], o[i + 1:i + 1 + HMAX]
    prior = np.maximum.accumulate(np.concatenate(([e], hh)))[:-1]       # highest high through the PREVIOUS bar
    stop = prior - risk
    hit = np.where(ll <= stop)[0]
    if len(hit):
        j = int(hit[0]); px = min(stop[j], oo[j]); bars = j + 1
    else:
        px, bars = c[i + HMAX], HMAX
    r = (px - e) / risk - COST * e / risk
    return float(r), float((px / e - 1) * 100), bars, float(px)


def boot99(day, r, reps=2000, seed=11):
    u, inv = np.unique(day, return_inverse=True)
    sums, cnts = np.bincount(inv, r), np.bincount(inv)
    rng = np.random.default_rng(seed)
    m = np.array([sums[p].sum() / cnts[p].sum() for p in (rng.integers(0, len(u), len(u)) for _ in range(reps))])
    return float(np.percentile(m, 0.5)), float(np.percentile(m, 99.5))


def main():
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import MexcClient

    from .paper import NON_CRYPTO

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    out = sys.argv[2] if len(sys.argv) > 2 else None
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 300_000]
    rows, base, nsym, live = [], [], 0, []     # rows: (k, group-tier, t, day, R, pct, bars, L); base: (k, t, day, R)
    for tk in tick:
        sym = tk["symbol"]
        try:
            kl = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - days * 96 * 900, now), "Min15", now)
            o, h, l, c, v, amt, t = (np.asarray(kl[x], float) for x in ("open", "high", "low", "close", "vol", "amount", "time"))
            n = len(c)
            if n < WARM + 200:
                continue
            f, _, valid = features(o, h, l, c, v, t, np.array([0.0]), np.array([0.0]), np.zeros(n))
            tier = tier_of({x: f[x] for x in ("F1", "F2", "F3")}); tier[:WARM] = -1; tier[~valid] = -1
            a15 = atr14(h, l, c)
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
            continue
        nsym += 1
        for k in KS:
            for i in range(300, n - HMAX - 1, 7):
                r = trail_trade(o, h, l, c, i, k * 2 * a15[i])
                if r:
                    base.append((k, t[i], int(t[i] // 86400), r[0]))
        for i, kk in alerts(tier):
            if kk > 2 or i < 100 or np.isnan(a15[i]):
                continue
            ft = alert_features(c, amt, a15, i)
            for k in KS:
                r = trail_trade(o, h, l, c, i, k * 2 * a15[i])
                if r:
                    rows.append((k, kk, t[i], int(t[i] // 86400), r[0], r[1], r[2], bool(ft["L"])))
            if i >= n - 8:
                live.append((sym, kk, (n - 1 - i) * 15, c[-1], 2 * a15[i] / c[-1] * 100, ft["turn"]))
    cut = max(r[2] for r in rows) - 30 * 86400
    lines = [f"هدف مفتوح ووقف متحرك على تنبيهات T1-T3 · {nsym} عقداً · {len(rows) // 2} تنبيهاً · فاصل التحقق {time.strftime('%Y-%m-%d', time.gmtime(cut))} · القواعد: docs/TRAIL_EXIT.md", ""]
    for k in KS:
        bh = [b for b in base if b[0] == k and b[1] >= cut]
        bm = float(np.mean([b[3] for b in bh]))
        lines.append(f"== k={k} (وقف = {k}×ATR الساعة) · أساس عشوائي في التحقق: متوسط R {bm:+.3f} ({len(bh)} صفقة)")
        lines.append(f"{'المجموعة':<10}{'صفقات(تصميم/تحقق)':>20}{'متوسط R':>10}{'حد 99%':>20}{'رابحة':>8}{'متوسط %':>9}{'وسيط مدة':>10}  الحكم")
        groups = (("T1-T3", lambda r: True), ("T1", lambda r: r[1] == 0), ("T2", lambda r: r[1] == 1), ("T3", lambda r: r[1] == 2), ("T1-T3+L", lambda r: r[7]))
        for g, test in groups:
            d = [r for r in rows if r[0] == k and r[2] < cut and test(r)]
            h_ = [r for r in rows if r[0] == k and r[2] >= cut and test(r)]
            if not h_:
                lines.append(f"{g:<10}{len(d):>9}/0"); continue
            R = np.array([r[4] for r in h_]); lo, hi = boot99(np.array([r[3] for r in h_]), R)
            ok = len(h_) >= 100 and R.mean() > 0 and lo > 0 and R.mean() > bm
            lines.append(f"{g:<10}{len(d):>9}/{len(h_):<10}{R.mean():>+10.3f}   [{lo:+.3f}, {hi:+.3f}]{(R > 0).mean() * 100:>7.0f}%{np.mean([r[5] for r in h_]):>+8.1f}%"
                         f"{np.median([r[6] for r in h_]) * 15 / 60:>8.1f}س  {'نجحت' if ok else 'رُفضت'}")
        lines.append("")
    lines.append("تنبيهات حالية (آخر ساعتين) ومسافة الوقف المتحرك (2×ATR الساعة تقريباً بالنسبة المئوية):")
    for sym, kk, age, px, atrp, tn in sorted(live, key=lambda x: (x[1], x[2]))[:14]:
        lines.append(f"  T{kk + 1} {sym:<14} منذ {age}د · سعر {px:.6g} · ATR ساعة {atrp:.1f}% · وقف 2× = −{2 * atrp:.1f}% · وقف 3× = −{3 * atrp:.1f}% · سيولة {tn / 1e6:.1f}M")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
