"""Recurrence study (rules fixed in docs/WATCHLIST.md): does a coin that exploded recently explode again soon, and how far ahead can we flag it?
Usage: python -m tradebot.strategies.mtf_entry.boom_watch study [days] [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from . import model as M

EXPLODE_PCT, MERGE_BARS, STEP, AHEAD = 10.0, 48, 16, 96          # 4h decision step, 24h label horizon (15m bars)
D7, D14, H48 = 7 * 96, 14 * 96, 48 * 4
SQ_WIN, SQ_RANK = 200, 0.30
RULES = ("R1", "R2", "R3", "R4")


def event_starts(o, c):
    """Indices (and directions) of merged explosion events: first bar of a run of same-direction >=10% candles within MERGE_BARS."""
    ret = (c / o - 1) * 100
    last, ev = {}, []
    for j in np.where(np.abs(ret) >= EXPLODE_PCT)[0]:
        d = 1 if ret[j] > 0 else -1
        if j - last.get(d, -10**9) < MERGE_BARS:
            last[d] = j
            continue
        last[d] = j
        ev.append((int(j), d))
    return ev


def squeeze_rank(bw, t):
    hist = bw[t - SQ_WIN:t]
    if t < SQ_WIN or np.isnan(hist).any() or np.isnan(bw[t]):
        return float("nan")
    return float((hist < bw[t]).mean())


def flags(ev_idx, t, bw):
    """Rule flags at decision bar t using only events that started at or before t."""
    n7 = int(((ev_idx > t - D7) & (ev_idx <= t)).sum())
    n14 = int(((ev_idx > t - D14) & (ev_idx <= t)).sum())
    n48 = int(((ev_idx > t - H48) & (ev_idx <= t)).sum())
    sq = squeeze_rank(bw, t)
    return {"R1": n7 >= 2, "R2": n14 >= 3, "R3": n48 >= 1, "R4": n7 >= 2 and sq == sq and sq <= SQ_RANK}


def label_and_lead(ev, t):
    """(label, bars_to_first_event, direction) for events starting in (t, t+AHEAD]."""
    nxt = [(j, d) for j, d in ev if t < j <= t + AHEAD]
    return (True, nxt[0][0] - t, nxt[0][1]) if nxt else (False, None, None)


def day_boot(vals_by_day, n=2000, seed=5):
    """95% CI of a mean via day-block bootstrap; vals_by_day: list of arrays (one per day)."""
    rng = np.random.default_rng(seed)
    days = [np.asarray(v, float) for v in vals_by_day if len(v)]
    if not days:
        return float("nan"), float("nan")
    means = []
    for _ in range(n):
        pick = [days[k] for k in rng.integers(0, len(days), len(days))]
        means.append(np.concatenate(pick).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def study(days=90, out=None):
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import MexcClient

    from .paper import NON_CRYPTO

    cfg = load_config("config.toml")
    cl, now = MexcClient(cfg.mexc), int(time.time())
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 300_000]
    rows, nsym = [], 0
    for tk in tick:
        sym = tk["symbol"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - days * 96 * 900, now), "Min15", now)
            o, h, l, c, tt = (np.asarray(k[f], float) for f in ("open", "high", "low", "close", "time"))
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
            continue
        if len(c) < D14 + AHEAD + 300:
            continue
        nsym += 1
        ev = event_starts(o, c)
        ev_idx = np.array([j for j, _ in ev], int)
        mid = M._sma(c, M.SQ_BB_LEN)
        from numpy.lib.stride_tricks import sliding_window_view as swv
        sd = np.full(len(c), np.nan); sd[M.SQ_BB_LEN - 1:] = swv(c, M.SQ_BB_LEN).std(axis=1)
        bw = 4 * sd / mid
        for t in range(D14, len(c) - AHEAD, STEP):
            lab, lead, d = label_and_lead(ev, t)
            prev = [x for x in ev if x[0] <= t]
            rows.append({"sym": sym, "t": tt[t], "day": int(tt[t] // 86400), **flags(ev_idx, t, bw), "label": lab,
                         "lead_h": lead * 15 / 60 if lead else None, "dir_same": (d == prev[-1][1]) if (lab and prev) else None})
    if not rows:
        print("no data"); return
    tmin, tmax = min(r["t"] for r in rows), max(r["t"] for r in rows)
    cut = tmax - 30 * 86400
    lines = [f"قائمة العملات المتكررة · {nsym} عقداً · {len(rows)} نقطة قرار (كل 4س) · القواعد: docs/WATCHLIST.md", ""]

    def stats(sel, rule):
        base = np.mean([r["label"] for r in sel])
        fl = [r for r in sel if r[rule]]
        if not fl:
            return base, 0, float("nan"), float("nan"), float("nan")
        by_day = {}
        for r in fl:
            by_day.setdefault(r["day"], []).append(float(r["label"]))
        lo, hi = day_boot(list(by_day.values()))
        return base, len(fl), float(np.mean([r["label"] for r in fl])), lo, hi

    dev = [r for r in rows if r["t"] < cut]; hold = [r for r in rows if r["t"] >= cut]
    lines.append(f"{'القاعدة':<7}{'نقاط التصميم':>13}{'نسبتها':>9}{'الأساس':>8}{'رفع':>6}   {'نقاط التحقق':>12}{'نسبتها':>9}{'الأساس':>8}{'رفع':>6}  فترة ثقة التحقق")
    best, best_lift = None, 0
    for rule in RULES:
        b1, n1, p1, _, _ = stats(dev, rule)
        b2, n2, p2, lo2, hi2 = stats(hold, rule)
        lift1 = p1 / b1 if n1 and b1 else float("nan")
        lift2 = p2 / b2 if n2 and b2 else float("nan")
        lines.append(f"{rule:<7}{n1:>13}{p1 * 100:>8.1f}%{b1 * 100:>7.1f}%{lift1:>6.1f}   {n2:>12}{p2 * 100:>8.1f}%{b2 * 100:>7.1f}%{lift2:>6.1f}  [{lo2 * 100:.1f}%, {hi2 * 100:.1f}%]")
        if n1 >= 200 and lift1 == lift1 and lift1 > best_lift:
            best, best_lift = rule, lift1
    lines.append("")
    if best:
        b2, n2, p2, lo2, hi2 = stats(hold, best)
        ok = n2 >= 200 and p2 >= 2 * b2 and lo2 > b2
        lines.append(f"القاعدة الأساسية (مختارة من التصميم وحده): {best} · على التحقق: {p2 * 100:.1f}% مقابل أساس {b2 * 100:.1f}% (رفع {p2 / b2:.1f}) · "
                     f"الحكم المسجّل: {'نجحت' if ok else 'لم تنجح'}")
        hits = [r for r in hold if r[best] and r["label"]]
        if hits:
            lh = np.array([r["lead_h"] for r in hits])
            lines.append(f"المهلة بين التعليم وأول انفجار (الناجحون، {len(hits)} حالة): وسيط {np.median(lh):.1f} ساعة · خلال 6س {np.mean(lh <= 6) * 100:.0f}% · "
                         f"12س {np.mean(lh <= 12) * 100:.0f}% · 24س 100%")
            sd_ = [r["dir_same"] for r in hits if r["dir_same"] is not None]
            lines.append(f"جهة الانفجار القادم = جهة آخر انفجار سابق: {np.mean(sd_) * 100:.0f}% (الصدفة 50%) على {len(sd_)} حالة")
        syms = {}
        for r in hold:
            if r[best]:
                syms[r["sym"]] = syms.get(r["sym"], 0) + 1
        lines.append("أكثر العقود تعليماً في التحقق: " + "، ".join(f"{s} ({n})" for s, n in sorted(syms.items(), key=lambda x: -x[1])[:10]))
    lines += ["", "تحيّز معروف: العقود المؤهلة هي السائلة اليوم، والانفجار يرفع السيولة، فقد يميل الاختيار لصالح القواعد."]
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "study":
        study(int(sys.argv[2]) if len(sys.argv) > 2 else 90, sys.argv[3] if len(sys.argv) > 3 else None)
