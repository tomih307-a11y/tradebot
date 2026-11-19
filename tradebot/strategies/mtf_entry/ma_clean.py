"""Moving-average-only entry study for 'clean winners' (rules fixed in docs/MA_CLEAN.md). Long only.
Usage: python -m tradebot.strategies.mtf_entry.ma_clean [days] [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from .boom_watch import day_boot
from .channel_logic import sma
from .early_signs import WARM, features
from .hunter_tiers import alerts, tier_of
from .immediate_move import clean_label

COOL = 8
GROUPS = ("M1", "M2", "M3", "M2+L", "M2+T", "M3+L")


def ma_flags(c):
    """Per-bar boolean arrays M1, M2, M3 (data up to each bar only)."""
    n = len(c)
    m5, m10, m30, m60 = (sma(c, k) for k in (5, 10, 30, 60))
    h20, h40, h120, h240 = (sma(c, k) for k in (20, 40, 120, 240))
    with np.errstate(invalid="ignore"):
        m1 = (m5 > m10) & (m10 > m30) & (m30 > m60) & (c > m5)
        m2 = m1 & (h20 > h40) & (h40 > h120) & (h120 > h240) & (c > h20)
        cross = np.zeros(n, bool)
        cross[1:] = (m10[1:] > m30[1:]) & (m10[:-1] <= m30[:-1])
        cs = np.cumsum(cross)
        recent = (cs - np.concatenate((np.zeros(9, int), cs[:-9]))[:n]) >= 1
        m3 = m2 & recent
    return m1, m2, m3


def episodes(flag):
    """First bar of each run, with COOL bars of silence between alerts."""
    out, last = [], -10**9
    for i in np.where(flag)[0]:
        if i - last >= COOL:
            out.append(int(i))
        last = int(i) if i - last >= COOL else last
    return out


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
    rows, base, nsym, live = [], [], 0, []     # rows: (group, t, day, clean, bars, mae); base: (isL, t, day, clean)
    for tk in tick:
        sym = tk["symbol"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - days * 96 * 900, now), "Min15", now)
            o, h, l, c, v, amt, t = (np.asarray(k[x], float) for x in ("open", "high", "low", "close", "vol", "amount", "time"))
            n = len(c)
            if n < WARM + 300:
                continue
            f, _, valid = features(o, h, l, c, v, t, np.array([0.0]), np.array([0.0]), np.zeros(n))
            tier = tier_of({x: f[x] for x in ("F1", "F2", "F3")}); tier[:WARM] = -1; tier[~valid] = -1
            al = [i for i, kk in alerts(tier) if kk <= 2]
            tflag = np.zeros(n, bool)
            for i in al:
                tflag[i:i + COOL] = True
            m1, m2, m3 = ma_flags(c)
            cs = np.concatenate(([0.0], np.cumsum(amt)))
            turn = np.full(n, np.nan); turn[95:] = cs[96:] - cs[:-96]
            lq = np.nan_to_num(turn) >= 5e6
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
            continue
        nsym += 1
        for i in range(300, n, 7):                                  # unconditional baselines
            lab = clean_label(h, l, c, i)
            if lab is not None:
                base.append((bool(lq[i]), t[i], int(t[i] // 86400), lab[0]))
        spec = {"M1": m1, "M2": m2, "M3": m3, "M2+L": m2 & lq, "M2+T": m2 & tflag, "M3+L": m3 & lq}
        for g, flag in spec.items():
            flag = flag.copy(); flag[:300] = False
            for i in episodes(flag):
                lab = clean_label(h, l, c, i)
                if lab is not None:
                    rows.append((g, t[i], int(t[i] // 86400), lab[0], lab[1], lab[2]))
                elif i >= n - 8 and g in ("M2+L", "M2+T", "M3+L"):
                    live.append((g, sym, (n - 1 - i) * 15, c[-1], turn[i]))
    cut = max(r[1] for r in rows) - 30 * 86400
    hold_days = max((max(r[1] for r in rows) - cut) / 86400, 1)
    bh = [b for b in base if b[1] >= cut]
    base_all, base_l = np.mean([b[3] for b in bh]), np.mean([b[3] for b in bh if b[0]])
    lines = [f"دراسة المتوسطات (فائز نظيف) · {nsym} عقداً · فاصل التحقق {time.strftime('%Y-%m-%d', time.gmtime(cut))} · القواعد: docs/MA_CLEAN.md", "",
             f"أساس عشوائي (تحقق): {base_all * 100:.1f}% ({len(bh)}) · أساس السيولة: {base_l * 100:.1f}%",
             f"{'المجموعة':<8}{'إشارات(تصميم/تحقق)':>22}{'/يوم':>7}{'فائز نظيف':>12}{'حد 95%':>18}{'مهلة +5% وسيط':>16}  الحكم"]
    for g in GROUPS:
        d = [r for r in rows if r[0] == g and r[1] < cut]
        h_ = [r for r in rows if r[0] == g and r[1] >= cut]
        if not h_:
            lines.append(f"{g:<8}{len(d):>10}/0"); continue
        win = np.array([r[3] for r in h_], float)
        lo, hi = day_boot([np.array([r[3] for r in h_ if r[2] == dd], float) for dd in sorted({r[2] for r in h_})])
        w = [r for r in h_ if r[3]]
        lead = np.median([r[4] * 15 for r in w]) if w else float("nan")
        bs = base_l if "L" in g else base_all
        ok = len(h_) >= 100 and win.mean() >= 1.5 * bs and lo > bs and lo > 0.25
        lines.append(f"{g:<8}{len(d):>10}/{len(h_):<11}{len(h_) / hold_days:>7.1f}{win.mean() * 100:>11.1f}%   [{lo * 100:.1f}%, {hi * 100:.1f}%]{lead:>13.0f}د  {'نجحت' if ok else 'رُفضت'}")
    lines += ["", "إشارات حالية (آخر ساعتين) من مجموعات السيولة/التنبيه:"]
    for g, sym, age, px, tn in sorted(live, key=lambda x: (x[0], x[2])):
        lines.append(f"  {g:<6}{sym:<14} منذ {age}د · سعر {px:.6g} · سيولة 24س {tn / 1e6:.1f}M")
    if not live:
        lines.append("  لا شيء")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
