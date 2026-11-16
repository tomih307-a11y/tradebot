"""Tiered alert test for the long-only explosion hunter (rules fixed in docs/HUNTER_TIERS.md).
Usage: python -m tradebot.strategies.mtf_entry.hunter_tiers [days] [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from .boom_watch import day_boot
from .early_signs import AHEAD, WARM, features

TIERS = ("T1", "T2", "T3", "T4")
NAMES = {"T1": "عليا: تكرار+قفزة حجم", "T2": "وسطى: تكرار+تراكم", "T3": "متابعة: تكرار", "T4": "بلا تكرار: قفزة+تراكم"}
COOL = 8
CRIT = {"T1": (50, 0.15, 0.08), "T2": (100, 0.08, 0.04), "T3": (100, 0.05, 0.03), "T4": (100, 0.02, 0.012)}


def tier_of(f):
    """Per-bar tier index (0..3) or -1 using the fixed flags F1,F2,F3."""
    f1, f2, f3 = f["F1"], f["F2"], f["F3"]
    t = np.full(len(f1), -1)
    t[f3 & f2 & ~f1] = 3
    t[f1] = 2
    t[f1 & f2] = 1
    t[f1 & f3] = 0
    return t


def alerts(tier):
    """Indices where a tier alert fires, with a per-(tier) cooldown of COOL bars."""
    out = []
    for k in range(4):
        last = -10**9
        for i in np.where(tier == k)[0]:
            if i - last >= COOL:
                out.append((int(i), k))
            last = i if i - last >= COOL else last
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
    A, nsym, expl = [], 0, []            # A rows: tier, t, day, fmax, mae, lead_bars
    for tk in tick:
        sym = tk["symbol"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - days * 96 * 900, now), "Min15", now)
            o, h, l, c, v, t = (np.asarray(k[x], float) for x in ("open", "high", "low", "close", "vol", "time"))
            n = len(c)
            if n < WARM + 200:
                continue
            f, fmax, valid = features(o, h, l, c, v, t, np.array([0.0]), np.array([0.0]), np.zeros(n))
            f = {x: f[x] for x in ("F1", "F2", "F3")}
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
            continue
        nsym += 1
        tier = tier_of(f)
        tier[:WARM] = -1
        tier[~valid] = -1
        al = alerts(tier)
        al_idx = np.array([i for i, _ in al], int)
        for i, kk in al:
            if i + AHEAD >= n:
                continue
            fh, fl = h[i + 1:i + 1 + AHEAD], l[i + 1:i + 1 + AHEAD]
            hit = np.where(fh / c[i] - 1 >= 0.10)[0]
            A.append((kk, t[i], int(t[i] // 86400), (fh.max() / c[i] - 1) * 100, (fl.min() / c[i] - 1) * 100, (hit[0] + 1) * 15 if len(hit) else np.nan))
        # explosion events (>=10% in next 4 bars from a decision point) and whether any alert came in the 8 bars before
        ok = np.where(~np.isnan(fmax[:n - AHEAD]) & (fmax[:n - AHEAD] >= 10))[0]
        ok = ok[ok >= WARM]
        last = -10**9
        for j in ok:
            if j - last >= 8:
                prior = al_idx[(al_idx <= j) & (al_idx > j - COOL)]
                expl.append((t[j], len(prior) > 0))
            last = j
    A = np.array(A, float)
    cut = A[:, 1].max() - 30 * 86400
    hold, dev = A[:, 1] >= cut, A[:, 1] < cut
    ndays = max((A[hold, 1].max() - cut) / 86400, 1)
    lines = [f"طبقات الصائد · {nsym} عقداً · {int(len(A))} تنبيهاً · فاصل التحقق {time.strftime('%Y-%m-%d', time.gmtime(cut))} · القواعد: docs/HUNTER_TIERS.md", ""]
    lines.append(f"{'':3}{'الطبقة':<26}{'تنبيهات(تصميم/تحقق)':>22}{'/يوم':>6}{'≥10%':>8}{'≥20%':>8}{'≥30%':>8}{'حد ثقة ≥10%':>18}{'مهلة وسيط(د)':>14}{'هبوط وسيط ناجح':>16}{'هبوط وسيط الكل':>16}  الحكم")
    for k in range(4):
        d, hh = (A[:, 0] == k) & dev, (A[:, 0] == k) & hold
        if hh.sum() == 0:
            lines.append(f"{TIERS[k]:<3}{NAMES[TIERS[k]]:<26}{int(d.sum()):>10}/{0:<9}  لا تنبيهات")
            continue
        pr = [float((A[hh, 3] >= x).mean()) for x in (10, 20, 30)]
        byday = [(A[hh & (A[:, 2] == dd), 3] >= 10).astype(float) for dd in np.unique(A[hh, 2])]
        lo, hi = day_boot(byday)
        win = hh & (A[:, 3] >= 10)
        lead = np.nanmedian(A[win, 5]) if win.any() else float("nan")
        mae_w = np.median(A[win, 4]) if win.any() else float("nan")
        mae_a = np.median(A[hh, 4])
        nmin, pmin, lmin = CRIT[TIERS[k]]
        ok = hh.sum() >= nmin and pr[0] >= pmin and lo >= lmin
        lines.append(f"{TIERS[k]:<3}{NAMES[TIERS[k]]:<26}{int(d.sum()):>10}/{int(hh.sum()):<11}{hh.sum() / ndays:>6.1f}{pr[0] * 100:>7.1f}%{pr[1] * 100:>7.1f}%{pr[2] * 100:>7.1f}%"
                     f"   [{lo * 100:.1f}%, {hi * 100:.1f}%]{lead:>14.0f}{mae_w:>15.1f}%{mae_a:>15.1f}%  {'نجح' if ok else 'فشل'}")
    E = np.array(expl, float)
    eh = E[:, 0] >= cut
    lines += ["", f"الاستدعاء على التحقق: من {int(eh.sum())} انفجاراً (≥10% خلال ساعة، متباعدة ≥ 8 شموع) سبقه تنبيه من أي طبقة خلال ساعتين قبله: {int(E[eh, 1].sum())} ({E[eh, 1].mean() * 100:.0f}%)"]
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
