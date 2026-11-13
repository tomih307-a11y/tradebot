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
    pos_all = [r for r in hold if r["label"]]
    if pos_all:
        rec = np.mean([r["R3"] for r in pos_all])
        lines.append(f"الاستدعاء (ملحق أ، R3): من نقاط التحقق التي تبعها انفجار خلال 24س ({len(pos_all)})، سبقها تعليم R3 في {rec * 100:.0f}% منها، "
                     f"أي أن {100 - rec * 100:.0f}% من الانفجارات جاءت من عقود لم تنفجر خلال 48 ساعة قبلها.")
    lines += ["", "تحيّز معروف: العقود المؤهلة هي السائلة اليوم، والانفجار يرفع السيولة، فقد يميل الاختيار لصالح القواعد."]
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "study":
        study(int(sys.argv[2]) if len(sys.argv) > 2 else 90, sys.argv[3] if len(sys.argv) > 3 else None)


# ───────── live watchlist ─────────
def watch_snapshot(client, days: int = 14):
    """Liquid crypto contracts with their recent explosion history (15m candles, last `days`), newest explosion first."""
    from tradebot.data.klines import drop_unclosed, fetch_range

    from .paper import NON_CRYPTO

    now = int(time.time())
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 300_000]
    out = []
    for tk in tick:
        sym = tk["symbol"]
        try:
            k = drop_unclosed(fetch_range(client, sym, "Min15", int(now // 900 * 900) - days * 96 * 900, now), "Min15", now)
            o, h, l, c, v, tt = (np.asarray(k[f], float) for f in ("open", "high", "low", "close", "vol", "time"))
        except Exception:
            continue
        if len(c) < 400:
            continue
        ev = event_starts(o, c)
        if not ev:
            continue
        j, d = ev[-1]
        t = len(c) - 1
        pre = M.squeeze_precompute(o, h, l, c, v)
        hrs = (t - j) * 15 / 60
        n7 = sum(1 for e, _ in ev if e > t - D7)
        out.append({"sym": sym, "last_h": hrs, "last_dir": d, "last_pct": float((c[j] / o[j] - 1) * 100), "n7": n7, "n14": len(ev),
                    "since_pct": float((c[t] / o[j] - 1) * 100), "price": float(c[t]), "sq": squeeze_rank(np.asarray(pre["bw"]), t),
                    "vol_x": float(v[-4:].mean() / max(np.median(v[-96:]), 1e-12)), "watch": hrs <= 48})
    return sorted(out, key=lambda r: r["last_h"])


def watch_text(rows) -> str:
    w = [r for r in rows if r["watch"]]
    lines = [f"قائمة المراقبة (انفجر خلال آخر 48 ساعة) — {len(w)} عملة. في الدراسة المسجّلة: بعد التعليم انفجر 39.5% منها خلال 24س مقابل 2.8% للعقود عموماً (وسيط المهلة 10س).",
             "الجهة القادمة غير مضمونة: لم تطابق جهة آخر انفجار إلا في 41% (أي الأرجح العكس بنحو 59%).", ""]
    for r in w:
        lines.append(f"{r['sym']:<14} آخر انفجار قبل {r['last_h']:.1f}س ({'صعود' if r['last_dir'] == 1 else 'هبوط'} {r['last_pct']:+.1f}%) · منذها {r['since_pct']:+.1f}% · "
                     f"انفجارات 7أيام: {r['n7']} · 14يوماً: {r['n14']} · ضغط {r['sq'] * 100:.0f}% · حجم آخر ساعة ×{r['vol_x']:.1f} · سعر {r['price']:.6g}")
    rec = sorted((r for r in rows if r["n14"] >= 3 and not r["watch"]), key=lambda r: -r["n14"])
    lines += ["", f"متكررة لكنها هادئة الآن (≥3 انفجارات في 14 يوماً ولا انفجار خلال 48س) — {len(rec)}:"]
    for r in rec:
        lines.append(f"{r['sym']:<14} آخر انفجار قبل {r['last_h']:.0f}س · انفجارات 14يوماً: {r['n14']} · ضغط {r['sq'] * 100:.0f}% · سعر {r['price']:.6g}")
    return "\n".join(lines)


if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "watch":
    from tradebot.config import load_config
    from tradebot.mexc.client import MexcClient
    txt = watch_text(watch_snapshot(MexcClient(load_config("config.toml").mexc)))
    print(txt)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w", encoding="utf-8").write(txt)


def pick_text(client) -> str:
    """Watchlist coins crossed with the multi-frame direction, zone and extension, to rank a possible trade (paper/analysis only)."""
    from .paper import analyse, fetch

    rows = [r for r in watch_snapshot(client) if r["watch"]]
    now, out = int(time.time()), []
    for r in rows:
        try:
            d, score, zone, hit, _, states = analyse(client, r["sym"], now)
            k = fetch(client, r["sym"], M.ZONE_TF, now)
            hh, ll, cc = (np.asarray(k[f], float) for f in ("high", "low", "close"))
            atr, ef = M.last_atr(hh, ll, cc), float(M.ema(cc, M.EMA_FAST)[-1])
        except Exception:
            continue
        lean = d or (1 if score > 0 else -1 if score < 0 else 0)
        ext = (r["price"] - ef) / atr * lean if lean and atr == atr else float("nan")
        out.append({**r, "dir": d, "lean": lean, "score": score, "zone": zone, "ext": ext, "states": states, "atr_pct": atr / r["price"] * 100})
    lines = ["مرشحون من قائمة المراقبة مع اتجاه الفريمات (النتيجة ±1: اتفاق كل الفريمات)"]
    for r in sorted(out, key=lambda r: -abs(r["score"])):
        z = f"{r['zone'][0]:.6g}-{r['zone'][1]:.6g}" if r["zone"] else "-"
        lines.append(f"{r['sym']:<10} {'لونق' if r['lean'] == 1 else 'شورت' if r['lean'] == -1 else 'محايد'} {'مؤكد' if r['dir'] else 'ضعيف'} {r['score']:+.2f} · امتداد {r['ext']:+.2f} · "
                     f"آخر انفجار قبل {r['last_h']:.0f}س {'↑' if r['last_dir'] == 1 else '↓'} · منذها {r['since_pct']:+.1f}% · ATR الساعة {r['atr_pct']:.1f}% · سعر {r['price']:.6g} · منطقة {z} · "
                     + " ".join(f"{k[:4]}{'▲' if v == 1 else '▼' if v == -1 else '•'}" for k, v in r["states"].items()))
    return "\n".join(lines)


if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "pick":
    from tradebot.config import load_config
    from tradebot.mexc.client import MexcClient
    print(pick_text(MexcClient(load_config("config.toml").mexc)))


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "coin":
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import MexcClient

    cl, sym, now = MexcClient(load_config("config.toml").mexc), sys.argv[2], int(time.time())
    k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - 90 * 96 * 900, now), "Min15", now)
    o, h, l, c = (np.asarray(k[f], float) for f in ("open", "high", "low", "close"))
    ev = event_starts(o, c)
    ret = (c / o - 1) * 100
    up24 = np.array([(h[i + 1:i + 97].max() / c[i] - 1) * 100 for i in range(0, len(c) - 97, 4)])
    dn24 = np.array([(1 - l[i + 1:i + 97].min() / c[i]) * 100 for i in range(0, len(c) - 97, 4)])
    print(f"{sym}: {len(c)} شمعة 15د (~{len(c) / 96:.0f} يوماً)")
    print(f"شموع 15د ≥10%: {int((np.abs(ret) >= 10).sum())} · أحداث انفجار مدموجة: {len(ev)} · أكبر شمعة 15د: {ret[np.argmax(np.abs(ret))]:+.1f}%")
    print(f"أعلى صعود خلال 24س من أي نقطة: وسيط {np.median(up24):.1f}% · 90% من الحالات أقل من {np.percentile(up24, 90):.1f}% · الأقصى {up24.max():.1f}%")
    print(f"احتمال صعود ≥10% خلال 24س: {np.mean(up24 >= 10) * 100:.1f}% · ≥5%: {np.mean(up24 >= 5) * 100:.1f}% · هبوط ≥10%: {np.mean(dn24 >= 10) * 100:.1f}%")
