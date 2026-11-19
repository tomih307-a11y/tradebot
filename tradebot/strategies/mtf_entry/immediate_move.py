"""'Clean winner' study + live screen (rules fixed in docs/IMMEDIATE_MOVE.md). Long only, tiers T1-T3.
Usage: python -m tradebot.strategies.mtf_entry.immediate_move [days] [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from .boom_watch import day_boot
from .early_signs import WARM, features
from .hunter_tiers import NAMES, TIERS, alerts, tier_of
from .channel_logic import atr14

AHEAD, TARGET, MAXADV = 16, 0.05, 0.015
GROUPS = {"بلا مرشح": (), "L": ("L",), "L+P": ("L", "P"), "L+P+A": ("L", "P", "A"), "L+P+A+U": ("L", "P", "A", "U")}


def alert_features(c, amt, a15, i):
    """Filter flags using data up to and including bar i."""
    turn24 = float(amt[max(i - 95, 0):i + 1].sum())
    ret24 = c[i] / c[i - 96] - 1 if i >= 96 else np.nan
    atr1h = a15[i] * 2 / c[i]
    return dict(L=turn24 >= 5e6, P=bool(ret24 <= 0.15), A=bool(atr1h <= 0.05), U=bool(c[i] > c[i - 59:i + 1].mean()),
                turn=turn24, ret24=ret24, atr1h=atr1h)


def clean_label(h, l, c, i):
    """(clean winner?, bars to +5%, max adverse %) over the next AHEAD bars; None if not enough data."""
    if i + AHEAD >= len(c):
        return None
    fh, fl = h[i + 1:i + 1 + AHEAD], l[i + 1:i + 1 + AHEAD]
    hit = np.where(fh / c[i] - 1 >= TARGET)[0]
    if len(hit) == 0:
        return False, np.nan, (fl.min() / c[i] - 1) * 100
    mae = fl[:hit[0] + 1].min() / c[i] - 1
    return bool(mae >= -MAXADV), int(hit[0] + 1), mae * 100


def passes(f, grp):
    return all(f[k] for k in GROUPS[grp])


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
    kb = drop_unclosed(fetch_range(cl, "BTC_USDT", "Min15", int(now // 900 * 900) - (days + 3) * 96 * 900, now), "Min15", now)
    tb, cb = np.asarray(kb["time"], float), np.asarray(kb["close"], float)
    rows, live, nsym, crows = [], [], 0, []
    for tk in tick:
        sym = tk["symbol"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - days * 96 * 900, now), "Min15", now)
            o, h, l, c, v, amt, t = (np.asarray(k[x], float) for x in ("open", "high", "low", "close", "vol", "amount", "time"))
            n = len(c)
            if n < WARM + 200:
                continue
            f, fmax, valid = features(o, h, l, c, v, t, np.array([0.0]), np.array([0.0]), np.zeros(n))
            tier = tier_of({x: f[x] for x in ("F1", "F2", "F3")})
            tier[:WARM] = -1
            tier[~valid] = -1
            a15 = atr14(h, l, c)
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
            continue
        nsym += 1
        for i, kk in alerts(tier):
            if kk > 2 or i < 100 or np.isnan(a15[i]):
                continue
            ft = alert_features(c, amt, a15, i)
            ib = int(np.searchsorted(tb, t[i], side="right")) - 1
            ft["M"] = bool(ib >= 96 and cb[ib] / cb[ib - 96] - 1 >= 0)
            lab = clean_label(h, l, c, i)
            j = i + 1
            if j < n and c[j] > c[i] and l[j] >= c[i] * (1 - 0.005):          # confirmation bar (docs/IMMEDIATE_MOVE.md addendum)
                lc = clean_label(h, l, c, j)
                if lc is not None:
                    crows.append((kk, t[j], int(t[j] // 86400), lc[0], lc[1], lc[2], ft))
            if lab is not None:
                rows.append((kk, t[i], int(t[i] // 86400), lab[0], lab[1], lab[2], ft))
            elif i >= n - 8:
                live.append((sym, kk, (n - 1 - i) * 15, c[-1], ft, float(tk.get("amount24") or 0)))
    cut = max(r[1] for r in rows) - 30 * 86400
    lines = [f"دراسة «الفائز النظيف» · {nsym} عقداً · {len(rows)} تنبيهاً (T1-T3) · فاصل التحقق {time.strftime('%Y-%m-%d', time.gmtime(cut))} · القواعد: docs/IMMEDIATE_MOVE.md", ""]
    hold_days = max((max(r[1] for r in rows) - cut) / 86400, 1)
    base_h = np.mean([r[3] for r in rows if r[1] >= cut])
    lines.append(f"نسبة الأساس في التحقق (بلا مرشح، T1-T3): {base_h * 100:.1f}%")
    lines.append(f"{'المجموعة':<10}{'تنبيهات(تصميم/تحقق)':>22}{'/يوم':>7}{'فائز نظيف(تحقق)':>17}{'حد 95%':>18}{'مهلة +5% وسيط':>16}{'هبوط وسيط للفائز':>18}  الحكم")
    best = None
    for g in GROUPS:
        d = [r for r in rows if r[1] < cut and passes(r[6], g)]
        h_ = [r for r in rows if r[1] >= cut and passes(r[6], g)]
        if not h_:
            lines.append(f"{g:<10}{len(d):>10}/0")
            continue
        win = np.array([r[3] for r in h_], float)
        byday = [np.array([r[3] for r in h_ if r[2] == dd], float) for dd in sorted({r[2] for r in h_})]
        lo, hi = day_boot(byday)
        w = [r for r in h_ if r[3]]
        lead = np.median([r[4] * 15 for r in w]) if w else float("nan")
        mae = np.median([r[5] for r in w]) if w else float("nan")
        ok = len(h_) >= 50 and win.mean() >= 0.10 and lo > base_h and win.mean() >= 2 * base_h
        if ok and (best is None or win.mean() > best[1]):
            best = (g, win.mean())
        lines.append(f"{g:<10}{len(d):>10}/{len(h_):<11}{len(h_) / hold_days:>7.1f}{win.mean() * 100:>16.1f}%   [{lo * 100:.1f}%, {hi * 100:.1f}%]{lead:>13.0f}د{mae:>17.1f}%  {'نجحت' if ok else 'رُفضت'}")
    lines += ["", "== ملحق: الدخول بعد تأكيد (شمعة تالية تغلق فوق إغلاق التنبيه، وأدنى سعر ≥ −0.5%) ==",
              f"{'المجموعة':<10}{'مؤكدة(تصميم/تحقق)':>22}{'/يوم':>7}{'فائز نظيف(تحقق)':>17}{'حد 95%':>18}{'مهلة +5% وسيط':>16}  الحكم"]
    cbest = None
    for g in ("C", "C+L", "C+L+P+A"):
        keys = {"C": (), "C+L": ("L",), "C+L+P+A": ("L", "P", "A")}[g]
        ok_f = lambda r: all(r[6][k] for k in keys)
        d = [r for r in crows if r[1] < cut and ok_f(r)]
        h_ = [r for r in crows if r[1] >= cut and ok_f(r)]
        if not h_:
            lines.append(f"{g:<10}{len(d):>10}/0"); continue
        win = np.array([r[3] for r in h_], float)
        lo, hi = day_boot([np.array([r[3] for r in h_ if r[2] == dd], float) for dd in sorted({r[2] for r in h_})])
        w = [r for r in h_ if r[3]]
        lead = np.median([r[4] * 15 for r in w]) if w else float("nan")
        ok = len(h_) >= 50 and win.mean() >= 1.5 * base_h and lo > base_h
        if ok and (cbest is None or win.mean() > cbest[1]):
            cbest = (g, win.mean())
        lines.append(f"{g:<10}{len(d):>10}/{len(h_):<11}{len(h_) / hold_days:>7.1f}{win.mean() * 100:>16.1f}%   [{lo * 100:.1f}%, {hi * 100:.1f}%]{lead:>13.0f}د  {'نجحت' if ok else 'رُفضت'}")
    lines.append(f"المجموعة الأفضل الناجحة (بعد تأكيد): {cbest[0] if cbest else 'لا شيء'}")
    lines += ["", "== ملحق 2: حالة السوق (M: BTC صاعد 24س) وطبقة T1 وحدها ==",
              f"{'المجموعة':<10}{'تنبيهات(تصميم/تحقق)':>22}{'/يوم':>7}{'فائز نظيف(تحقق)':>17}{'حد 95%':>18}{'مهلة +5% وسيط':>16}  الحكم"]
    for g, test in (("T1", lambda r: r[0] == 0), ("T1+L", lambda r: r[0] == 0 and r[6]["L"]), ("M", lambda r: r[6]["M"]), ("T1+M", lambda r: r[0] == 0 and r[6]["M"])):
        d = [r for r in rows if r[1] < cut and test(r)]
        h_ = [r for r in rows if r[1] >= cut and test(r)]
        if not h_:
            lines.append(f"{g:<10}{len(d):>10}/0"); continue
        win = np.array([r[3] for r in h_], float)
        lo, hi = day_boot([np.array([r[3] for r in h_ if r[2] == dd], float) for dd in sorted({r[2] for r in h_})])
        w = [r for r in h_ if r[3]]
        lead = np.median([r[4] * 15 for r in w]) if w else float("nan")
        ok = len(h_) >= 50 and win.mean() >= 1.5 * base_h and lo > base_h and lo > 0.25
        lines.append(f"{g:<10}{len(d):>10}/{len(h_):<11}{len(h_) / hold_days:>7.1f}{win.mean() * 100:>16.1f}%   [{lo * 100:.1f}%, {hi * 100:.1f}%]{lead:>13.0f}د  {'نجحت' if ok else 'رُفضت'}")
    lines.append("")
    lines.append(f"المجموعة الأفضل الناجحة: {best[0] if best else 'لا شيء'}")
    grp = best[0] if best else "L+P+A"
    lines += ["", f"التنبيهات الحالية (آخر ساعتين) التي تمر المرشِّح «{grp}»" + ("" if best else " — للعلم فقط، المجموعة لم تنجح") + ":"]
    cur = [x for x in live if passes(x[4], grp)]
    for sym, kk, age, px, ft, a24 in sorted(cur, key=lambda x: (x[1], x[2])):
        lines.append(f"  {TIERS[kk]} {sym:<14} منذ {age}د · سعر {px:.6g} · سيولة 24س {ft['turn'] / 1e6:.1f}M · عائد 24س {ft['ret24'] * 100:+.1f}% · تذبذب الساعة {ft['atr1h'] * 100:.1f}%")
    if not cur:
        lines.append("  لا شيء")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
