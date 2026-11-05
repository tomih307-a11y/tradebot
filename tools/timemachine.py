"""Time Machine for MEXC futures: what are the odds of a planned PMO trade, BEFORE you risk money?

The rule (config via flags): on the 15m chart PMO crosses ABOVE its signal line with both lines above zero (below zero for shorts), AND the
higher-timeframe PMO (1H or 4H, last closed candle) is on the same side of zero. PMO length is a flag (--roc-len 4 matches a chart showing
"PMO close 4 20 10"). Closed candles only. Entry = next 15m open (+slippage). Stop = recent swing extreme. No fixed target: the stop trails
by N x ATR(1H) once the trade is `act_r` x risk in profit; unprofitable trades are cut after `loser_hours`.

Mode 1  --symbol X_USDT : (a) every past signal of the rule on this coin, run to its exit; (b) the SAME entries at random times as a baseline
                          (permutation test: does the rule beat random timing on this coin?); (c) 10,000 Monte Carlo replays of the planned
                          trade built from the coin's own recent 15m bars (same time-of-day +-2 bars); (d) one honest verdict.
Mode 2  --universe N    : the rule on the N most liquid contracts, validation on the last 40% of the period, reverse-direction baseline,
                          day-block bootstrap CI, liquidity tiers.
Usage:  python tools/timemachine.py --symbol NIULAI_USDT --roc-len 4 --htf Min60 --side long --out reports/tm
        python tools/timemachine.py --universe 400 --roc-len 4 --htf Min60 --side both --out reports/tm_universe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rules_study import simulate_open                                  # noqa: E402
from smc_study import summarize                                        # noqa: E402
from smc_study2 import boot_ci                                         # noqa: E402
from tradebot.config import load_config                                # noqa: E402
from tradebot.data.klines import get_klines                            # noqa: E402
from tradebot.indicators import atr_rma, pmo                           # noqa: E402
from tradebot.mexc.client import MexcClient                            # noqa: E402

M15, H1, H4 = 900, 3600, 14400
REASONS = {0: "max_hold", 1: "stop", 2: "trail_stop", 3: "time_loser"}


# ───────────────────────────── signals ─────────────────────────────
def pmo_masked(close, ind, roc_len):
    p, sg, _ = pmo(close, roc_len, ind.roc_scale, ind.pmo_smooth_len, ind.signal_len)
    p, sg = p.copy(), sg.copy()
    p[: int(ind.warmup_bars)] = np.nan
    sg[: int(ind.warmup_bars)] = np.nan
    return p, sg


def htf_on_15m(k_htf, step, t15, ind, roc_len):
    """PMO of the higher timeframe as of each 15m bar's close: the last candle already CLOSED (no look-ahead)."""
    ph, _ = pmo_masked(k_htf["close"], ind, roc_len)
    ix = np.searchsorted(k_htf["time"] + step, t15 + M15, side="right") - 1
    return np.where(ix >= 0, ph[np.clip(ix, 0, None)], np.nan)


def find_signals(p, sg, phtf, s):
    """Bars where PMO crosses above its signal (below for s=-1) with both lines on the s-side of zero and the HTF PMO on the s-side too."""
    x, y, h = s * p, s * sg, s * phtf
    prev = np.roll(x - y, 1)
    ok = (x - y >= 0) & (prev < 0) & (x > 0) & (y > 0) & (h > 0)
    ok[0] = False
    return np.where(ok & ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(h))[0]


def swing_stop(s, low, high, close_j, j, atr_j, bars, buf_atr, min_risk_pct):
    """Stop beyond the swing extreme of the last `bars` candles; never closer than min_risk_pct of price."""
    if s == 1:
        stop = float(low[max(0, j - bars + 1): j + 1].min()) - buf_atr * atr_j
    else:
        stop = float(high[max(0, j - bars + 1): j + 1].max()) + buf_atr * atr_j
    floor = min_risk_pct / 100.0 * close_j
    if s * (close_j - stop) < floor:
        stop = close_j - s * floor
    return stop


# ───────────────────────────── vectorised Monte Carlo ─────────────────────────────
def slot_candidates(times, width=2):
    slot = (np.asarray(times) // M15) % 96
    return [np.where(np.minimum((slot - k) % 96, (k - slot) % 96) <= width)[0] for k in range(96)]


def mc_paths(s, entry_ref, stop0, ratios, cand, slot0, n, T, trail_atr, act_r, atr_pct, loser_bars, fee_pct, slip_pct, rng):
    """n replays of the planned trade. ratios = (Ro, Rh, Rl, Rc) of recent bars over their previous close; each step draws a bar from the
    same time of day (+-2 bars). Same exit logic as simulate_open: stop (adverse first) -> unprofitable cut -> trailing update."""
    Ro, Rh, Rl, Rc = ratios
    px_in = entry_ref * (1 + s * slip_pct / 100.0)
    risk = s * (px_in - stop0)
    if risk <= 0:
        raise ValueError("stop is not beyond the entry")
    stop = np.full(n, float(stop0))
    pc = np.full(n, entry_ref, dtype=float)
    alive = np.ones(n, bool)
    exit_px = np.full(n, np.nan)
    reason = np.zeros(n, int)
    for k in range(T):
        c = cand[(slot0 + k) % 96]
        pick = c[rng.integers(0, len(c), size=n)]
        o, h, l, cl = pc * Ro[pick], pc * Rh[pick], pc * Rl[pick], pc * Rc[pick]
        if k == 0:                                         # the entry fills at the first candle's open
            f = px_in / o
            h, l, cl, o = h * f, l * f, cl * f, np.full(n, px_in)
        adverse = l if s == 1 else h
        hit = alive & (s * (adverse - stop) <= 0)
        gap = hit & (k > 0) & (s * (o - stop) <= 0)
        px = np.where(gap, o, stop) * (1 - s * slip_pct / 100.0)
        exit_px[hit] = px[hit]
        reason[hit] = np.where(s * (stop - stop0) > 0, 2, 1)[hit]
        alive &= ~hit
        upnl = s * (cl - px_in)
        lose = alive & (k >= loser_bars) & (upnl <= 0)
        exit_px[lose] = (cl * (1 - s * slip_pct / 100.0))[lose]
        reason[lose] = 3
        alive &= ~lose
        act = alive & (upnl >= act_r * risk)
        trail = cl - s * trail_atr * atr_pct * cl
        stop = np.where(act & (s * (trail - stop) > 0), trail, stop)
        pc = cl
    last = alive
    exit_px[last] = (pc * (1 - s * slip_pct / 100.0))[last]
    gross = s * (exit_px - px_in) / risk
    fees = fee_pct / 100.0 * (px_in + exit_px) / risk
    return gross - fees, gross, reason


# ───────────────────────────── per-coin evaluation ─────────────────────────────
def load_coin(client, dd, sym, now, days, htf):
    start = now - days * 86400
    k15 = get_klines(client, dd, sym, "Min15", start - 500 * M15, now)
    k1 = get_klines(client, dd, sym, "Min60", start - 45 * 86400, now)
    k4 = get_klines(client, dd, sym, "Hour4", start - 135 * 86400, now) if htf == "Hour4" else None
    return k15, k1, k4


def prepare(k15, k1, k4, ind, roc_len, htf, atr_len):
    p, sg = pmo_masked(k15["close"], ind, roc_len)
    hk, step = (k1, H1) if htf == "Min60" else (k4, H4)
    phtf = htf_on_15m(hk, step, k15["time"], ind, roc_len)
    a1 = atr_rma(k1["high"], k1["low"], k1["close"], atr_len)
    ix = np.searchsorted(k1["time"] + H1, k15["time"] + M15, side="right") - 1
    atr15 = np.where(ix >= 0, a1[np.clip(ix, 0, None)], np.nan)
    return p, sg, phtf, atr15


def run_trade(s, j, k15, atr15, a):
    """Entry at the open of bar j+1 after a signal/entry decision at bar j. -> result dict or None."""
    o, h, l, c = k15["open"], k15["high"], k15["low"], k15["close"]
    if j + 2 >= len(c) or not np.isfinite(atr15[j]):
        return None
    stop0 = swing_stop(s, l, h, float(c[j]), j, float(atr15[j]), a.swing_bars, a.stop_buf_atr, a.min_risk_pct)
    r = simulate_open(s, stop0, o, h, l, c, atr15, j + 1, a.trail_atr, a.act_r, a.max_hold_bars, a.loser_bars, a.fee, a.slip)
    if not r["filled"]:
        return None
    r.update(stop0=stop0, T=int(k15["time"][j]) + M15)
    return r


def sequential(sig, s, k15, atr15, a, reverse=False):
    out, busy = [], -1
    for j in sig:
        if j + 1 <= busy:
            continue
        r = run_trade(s, int(j), k15, atr15, a) if not reverse else _reverse_trade(s, int(j), k15, atr15, a)
        if r is None:
            continue
        busy = r["exit_j"]
        out.append({**r, "side": "long" if (s == 1) != reverse else "short"})
    return out


def _reverse_trade(s, j, k15, atr15, a):
    o, h, l, c = k15["open"], k15["high"], k15["low"], k15["close"]
    if j + 2 >= len(c) or not np.isfinite(atr15[j]):
        return None
    stop0 = swing_stop(s, l, h, float(c[j]), j, float(atr15[j]), a.swing_bars, a.stop_buf_atr, a.min_risk_pct)
    dist = s * (float(o[j + 1]) - stop0)
    rs, stop_r = -s, float(o[j + 1]) + s * dist
    r = simulate_open(rs, stop_r, o, h, l, c, atr15, j + 1, a.trail_atr, a.act_r, a.max_hold_bars, a.loser_bars, a.fee, a.slip)
    if not r["filled"]:
        return None
    r.update(stop0=stop_r, T=int(k15["time"][j]) + M15)
    return r


def verdict(n, mean, lo, hi, p_perm):
    """One honest sentence. Thresholds are fixed here, not tuned."""
    if n < 30:
        return "عدد الإشارات على هذه العملة قليل (أقل من 30): لا يمكن الحكم."
    if lo > 0 and p_perm < 0.05:
        return "القاعدة تتفوق على الدخول العشوائي على هذه العملة في هذه الفترة (متوسط R موجب وفترة الثقة فوق الصفر). هذا وصف للماضي، لا وعد بالمستقبل."
    if hi < 0:
        return "القاعدة خسرت بدلالة إحصائية على هذه العملة في هذه الفترة (فترة الثقة كلها تحت الصفر)."
    if p_perm >= 0.05:
        return "لا فرق قابل للقياس عن الدخول العشوائي على هذه العملة: أقرب إلى رمية عملة."
    return "النتيجة غير حاسمة: فترة الثقة تشمل الصفر."


def single(a, cfg, client):
    ind = cfg.pmo_zero_cross.indicator
    now = int(time.time())
    k15, k1, k4 = load_coin(client, cfg.pmo_zero_cross.backtest.data_dir, a.symbol, now, a.days, a.htf)
    if len(k15["close"]) < int(ind.warmup_bars) + 400:
        raise SystemExit("بيانات 15 دقيقة غير كافية لهذه العملة")
    p, sg, phtf, atr15 = prepare(k15, k1, k4, ind, a.roc_len, a.htf, int(cfg.agents.technical.atr_len))
    rng = np.random.default_rng(a.seed)
    sides = (1, -1) if a.side == "both" else ((1,) if a.side == "long" else (-1,))
    res = {"symbol": a.symbol, "roc_len": a.roc_len, "htf": a.htf, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "bars_15m": int(len(k15["close"])), "last_price": float(k15["close"][-1]), "sides": {}}
    valid = np.arange(int(ind.warmup_bars) + 1, len(k15["close"]) - 3)
    for s in sides:
        sig = find_signals(p, sg, phtf, s)
        sig = sig[sig > int(ind.warmup_bars)]
        seq = sequential(sig, s, k15, atr15, a)
        # R of EVERY possible entry bar with the same stop/exit rule: the random-timing baseline
        allR = np.full(len(k15["close"]), np.nan)
        for j in valid:
            r = run_trade(s, int(j), k15, atr15, a)
            if r is not None:
                allR[j] = r["R_net"]
        rs_sig = allR[sig][~np.isnan(allR[sig])]
        pool = allR[~np.isnan(allR)]
        p_perm, base_mean = None, float(pool.mean()) if len(pool) else None
        if len(rs_sig) >= 5 and len(pool) > len(rs_sig):
            obs = rs_sig.mean()
            draws = np.array([rng.choice(pool, size=len(rs_sig), replace=False).mean() for _ in range(2000)])
            p_perm = float((draws >= obs).mean())
        lo, hi = boot_ci(seq) if len(seq) >= 5 else (float("nan"), float("nan"))
        sm = summarize([t["R_net"] for t in seq]) if seq else None
        reasons = {}
        for t in seq:
            reasons.setdefault(t["reason"], []).append(t["R_net"])
        # planned trade for Monte Carlo: the latest bar's entry (or the user's plan)
        j_last = len(k15["close"]) - 2
        entry = a.entry if a.entry else float(k15["close"][-1])
        if a.plan_stop:
            stop, stop_src = a.plan_stop, "من المستخدم"
        elif seq:                                           # typical stop distance of this rule on this coin, not the (arbitrary) latest bar
            med = float(np.median([t["risk_pct"] for t in seq]))
            stop, stop_src = entry * (1 - s * med / 100.0), f"وسيط مسافة وقف القاعدة في تاريخ العملة ({med:.2f}%)"
        else:
            stop, stop_src = swing_stop(s, k15["low"], k15["high"], entry, j_last, float(atr15[j_last]), a.swing_bars, a.stop_buf_atr, a.min_risk_pct), "قاع آخر شموع"
        mc = None
        if s * (entry - stop) > 0:
            tm = k15["time"]
            m = tm >= tm[-1] - a.mc_days * 86400
            pcv = np.roll(k15["close"], 1)
            ratios = tuple(k15[f][m][1:] / pcv[m][1:] for f in ("open", "high", "low", "close"))
            cand = slot_candidates(tm[m][1:])
            slot0 = int((tm[-1] + M15) // M15) % 96
            atr_pct = float(atr15[j_last] / k15["close"][j_last])
            R, G, rsn = mc_paths(s, entry, stop, ratios, cand, slot0, a.replays, a.max_hold_bars + 1, a.trail_atr, a.act_r, atr_pct,
                                 a.loser_bars, a.fee, a.slip, rng)
            mc = {"entry": entry, "stop": stop, "stop_source": stop_src, "risk_pct": float(s * (entry - stop) / entry * 100), "replays": a.replays,
                  "mean_R_net": float(R.mean()), "median_R_net": float(np.median(R)), "p_win": float((R > 0).mean()),
                  "p10": float(np.percentile(R, 10)), "p90": float(np.percentile(R, 90)),
                  "reason": {REASONS[k]: float((rsn == k).mean()) for k in REASONS}}
        res["sides"]["long" if s == 1 else "short"] = {
            "signals": int(len(sig)), "trades": len(seq), "mean_R_net": sm["mean"] if sm else None, "win_pct": sm["win"] if sm else None,
            "mean_R_gross": float(np.mean([t["R_gross"] for t in seq])) if seq else None, "ci": [lo, hi],
            "random_entry_mean_R": base_mean, "p_vs_random": p_perm,
            "mean_stop_pct": float(np.mean([t["risk_pct"] for t in seq])) if seq else None,
            "mean_hold_h": float(np.mean([t["bars"] for t in seq]) * 0.25) if seq else None,
            "reasons": {k: {"n": len(v), "mean_R": float(np.mean(v))} for k, v in reasons.items()},
            "monte_carlo": mc,
            "verdict": verdict(len(seq), sm["mean"] if sm else 0.0, lo, hi, p_perm if p_perm is not None else 1.0),
        }
    return res


def single_md(res, a):
    L = [f"# آلة الزمن: {res['symbol']} (PMO {res['roc_len']:g}، الفريم الأكبر {res['htf']})", "",
         f"- {res['generated']} · شموع 15د: {res['bars_15m']} · آخر سعر: {res['last_price']:.6g}",
         f"- القاعدة: PMO على 15د يقطع الإشارة صاعداً فوق الصفر (هابطاً تحت الصفر للبيع) وPMO الفريم الأكبر في نفس الجهة. دخول عند فتح الشمعة التالية، "
         f"وقف عند قاع/قمة آخر {a.swing_bars} شمعة، وقف متحرك {a.trail_atr}×ATR(ساعة) بعد ربح {a.act_r}R، قطع الخاسرة بعد {a.loser_bars // 4} ساعات.", ""]
    for sd, v in res["sides"].items():
        L.append(f"## {'لونق' if sd == 'long' else 'شورت'}")
        L.append(f"**الحكم:** {v['verdict']}\n")
        f = lambda x, d=3: "–" if x is None else f"{x:+.{d}f}"
        L.append("### 1) تاريخ القاعدة على هذه العملة")
        L.append("| إشارات | صفقات (بلا تداخل) | صافي R | إجمالي R | فترة الثقة 95% | ربح % | الدخول العشوائي (صافي R) | p مقابل العشوائي | وقف متوسط | احتفاظ (ساعة) |\n|---|---|---|---|---|---|---|---|---|---|")
        L.append(f"| {v['signals']} | {v['trades']} | {f(v['mean_R_net'])} | {f(v['mean_R_gross'])} | [{v['ci'][0]:+.3f} ، {v['ci'][1]:+.3f}] | "
                 f"{'–' if v['win_pct'] is None else format(v['win_pct'], '.0f') + '%'} | {f(v['random_entry_mean_R'])} | "
                 f"{'–' if v['p_vs_random'] is None else format(v['p_vs_random'], '.3f')} | "
                 f"{'–' if v['mean_stop_pct'] is None else format(v['mean_stop_pct'], '.2f') + '%'} | {'–' if v['mean_hold_h'] is None else format(v['mean_hold_h'], '.1f')} |")
        if v["reasons"]:
            L.append("\n| سبب الخروج | عدد | متوسط R |\n|---|---|---|")
            for k, r in sorted(v["reasons"].items()):
                L.append(f"| {k} | {r['n']} | {r['mean_R']:+.3f} |")
        mc = v["monte_carlo"]
        if mc:
            L.append(f"\n### 2) {mc['replays']:,} محاكاة للصفقة المخططة (دخول {mc['entry']:.6g}، وقف {mc['stop']:.6g}، مسافة {mc['risk_pct']:.2f}%)")
            L.append(f"الوقف المخطط: {mc['stop_source']}. أعطِ `--entry` و`--plan-stop` لتجربة صفقتك أنت.")
            L.append("| صافي R متوسط | وسيط | احتمال الربح | أسوأ 10% | أفضل 10% |\n|---|---|---|---|---|")
            L.append(f"| {mc['mean_R_net']:+.3f} | {mc['median_R_net']:+.3f} | {mc['p_win'] * 100:.0f}% | {mc['p10']:+.2f} | {mc['p90']:+.2f} |")
            L.append("\n| " + " | ".join(mc["reason"].keys()) + " |\n|" + "---|" * len(mc["reason"]) + "\n| " + " | ".join(f"{x * 100:.0f}%" for x in mc["reason"].values()) + " |")
        L.append("")
    L.append("**كيف تُقرأ:** R = المخاطرة الأولية (المسافة للوقف). «الدخول العشوائي» = نفس الوقف والخروج لكن بدخول في أوقات عشوائية على العملة نفسها: "
             "إن لم تتفوق القاعدة عليه فلا قيمة لتوقيتها. المحاكاة تُبنى من شموع العملة نفسها (آخر %d يوماً) فهي تقيس الضجيج، ولا تتنبأ. "
             "كل ما هنا وصف للماضي، وفيه رسوم Taker وانزلاق." % a.mc_days)
    return "\n".join(L)


# ───────────────────────────── universe mode ─────────────────────────────
def universe(a, cfg, client):
    ind = cfg.pmo_zero_cross.indicator
    now = int(time.time())
    eval_start = now - a.days * 86400
    split = eval_start + (now - eval_start) * (100 - a.val_pct) // 100
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    symbols = [t["symbol"] for t in tick[: a.universe]]
    rank = {s_: i + 1 for i, s_ in enumerate(symbols)}
    sides = (1, -1) if a.side == "both" else ((1,) if a.side == "long" else (-1,))
    trades, rev, used = [], [], 0
    dd = cfg.pmo_zero_cross.backtest.data_dir
    for n_, sym in enumerate(symbols, 1):
        try:
            k15, k1, k4 = load_coin(client, dd, sym, now, a.days, a.htf)
            if len(k15["close"]) < int(ind.warmup_bars) + 600 or len(k1["close"]) < 400 or (a.htf == "Hour4" and (k4 is None or len(k4["close"]) < 400)):
                continue
            p, sg, phtf, atr15 = prepare(k15, k1, k4, ind, a.roc_len, a.htf, int(cfg.agents.technical.atr_len))
        except Exception as e:
            print(f"  {sym}: {e}", flush=True)
            continue
        used += 1
        t15 = k15["time"]
        for s in sides:
            sig = find_signals(p, sg, phtf, s)
            sig = sig[(t15[sig] + M15 >= eval_start) & (sig > int(ind.warmup_bars))]
            for t in sequential(sig, s, k15, atr15, a):
                trades.append({**t, "symbol": sym})
            for t in sequential(sig, s, k15, atr15, a, reverse=True):
                rev.append({**t, "symbol": sym})
        if n_ % 50 == 0:
            print(f"  [{n_}/{len(symbols)}] {sym} trades={len(trades)}", flush=True)
    trades.sort(key=lambda t: t["T"])
    rev.sort(key=lambda t: t["T"])
    val = [t for t in trades if t["T"] >= split]
    sel = [t for t in trades if t["T"] < split]
    rval = [t for t in rev if t["T"] >= split]

    def st(tr):
        sm = summarize([t["R_net"] for t in tr])
        if not sm:
            return None
        sm["gross"] = float(np.mean([t["R_gross"] for t in tr]))
        return sm

    def row(label, sm):
        if not sm:
            return f"| {label} | 0 | - | - | - | - | - |"
        return f"| {label} | {sm['n']} | {sm['mean']:+.3f} | {sm['gross']:+.3f} | {sm['win']:.0f}% | {sm['pf']:.2f} | {sm['t']:+.2f} |"

    lo, hi = boot_ci(val) if val else (float("nan"), float("nan"))
    z, r_ = st(val), st(rval)
    L = [f"# اختبار قاعدتك على {used} عقداً: PMO {a.roc_len:g} + الفريم الأكبر {a.htf} (الاتجاه: {a.side})", "",
         f"- آخر {a.days} يوماً · الحكم على آخر {a.val_pct}% · عقود موجودة اليوم فقط (انحياز بقاء) · رسوم Taker {a.fee}% للجهتين + انزلاق {a.slip}%.",
         f"- الدخول: عند فتح الشمعة بعد التقاطع. الوقف: قاع/قمة آخر {a.swing_bars} شمعة 15د. وقف متحرك {a.trail_atr}×ATR(ساعة) بعد ربح {a.act_r}R. لا حد صفقتين، ولا أخبار.", "",
         "| | الصفقات | صافي R | إجمالي R | ربح % | PF | t |\n|---|---|---|---|---|---|---|",
         row("قاعدتك (التحقق)", z), row("عكس الاتجاه نفس المسافة (التحقق)", r_), row("قاعدتك (التصميم)", st(sel)),
         f"\n- فترة ثقة 95% لصافي R في التحقق: [{lo:+.3f} ، {hi:+.3f}]"]
    if val:
        risk = float(np.mean([t["risk_pct"] for t in val]))
        L.append(f"- مسافة الوقف المتوسطة {risk:.2f}% ⇒ بهامش 10% ورافعة 5: ≈ {z['mean'] * risk / 100 * 5 * 10:+.3f}% من الرصيد للصفقة")
        L.append("\n| شريحة السيولة | عقود | صفقات | صافي R | إجمالي R |\n|---|---|---|---|---|")
        for nm, a_, b_ in (("1–40", 1, 40), ("41–150", 41, 150), ("151–400", 151, 400), ("401+", 401, 10 ** 6)):
            xs = [t for t in val if a_ <= rank[t["symbol"]] <= b_]
            L.append(f"| {nm} | {len({t['symbol'] for t in xs})} | {len(xs)} | {np.mean([t['R_net'] for t in xs]):+.3f} | {np.mean([t['R_gross'] for t in xs]):+.3f} |"
                     if xs else f"| {nm} | 0 | 0 | - | - |")
        L.append("\n| سبب الخروج | عدد | متوسط R |\n|---|---|---|")
        for k in sorted({t["reason"] for t in val}):
            xs = [t["R_net"] for t in val if t["reason"] == k]
            L.append(f"| {k} | {len(xs)} | {np.mean(xs):+.3f} |")
        L.append("\n| الاتجاه | عدد | صافي R |\n|---|---|---|")
        for sd in ("long", "short"):
            xs = [t["R_net"] for t in val if t["side"] == sd]
            L.append(f"| {sd} | {len(xs)} | {np.mean(xs):+.3f} |" if xs else f"| {sd} | 0 | - |")
    ok = bool(z and z["n"] >= 100 and z["mean"] > 0 and lo > 0 and (not r_ or z["mean"] > r_["mean"]))
    L.append("\n## الحكم الآلي\n\n**" + ("اجتاز" if ok else "لم يجتز") + "**  (المعيار: صافي R > 0 في التحقق، والحد الأدنى لفترة الثقة > 0، ويتفوق على عكس الاتجاه، وعدد الصفقات ≥ 100)")
    return "\n".join(L), {"used": used, "validation": z, "reverse": r_, "ci": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--universe", type=int, default=0)
    ap.add_argument("--roc-len", type=float, default=4)
    ap.add_argument("--htf", default="Min60", choices=["Min60", "Hour4"])
    ap.add_argument("--side", default="long", choices=["long", "short", "both"])
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--val-pct", type=int, default=40)
    ap.add_argument("--swing-bars", type=int, default=16)
    ap.add_argument("--stop-buf-atr", type=float, default=0.1)
    ap.add_argument("--min-risk-pct", type=float, default=0.3)
    ap.add_argument("--trail-atr", type=float, default=2.0)
    ap.add_argument("--act-r", type=float, default=1.0)
    ap.add_argument("--loser-hours", type=int, default=24)
    ap.add_argument("--max-hold-days", type=int, default=10)
    ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--replays", type=int, default=10000)
    ap.add_argument("--mc-days", type=int, default=60)
    ap.add_argument("--entry", type=float, default=None, help="planned entry price (default: last close)")
    ap.add_argument("--plan-stop", type=float, default=None, help="planned stop (default: swing stop)")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--out", default="reports/tm")
    a = ap.parse_args()
    cfg = load_config(a.config)
    a.fee = cfg.trading.taker_fee_pct if a.fee is None else a.fee
    a.loser_bars, a.max_hold_bars = a.loser_hours * 4, a.max_hold_days * 96
    client = MexcClient(cfg.mexc)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.symbol:
        res = single(a, cfg, client)
        md = single_md(res, a)
        (Path(a.out) / f"{a.symbol}_{a.htf}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        (Path(a.out) / f"{a.symbol}_{a.htf}.md").write_text(md, encoding="utf-8")
    elif a.universe:
        md, raw = universe(a, cfg, client)
        (Path(a.out) / f"universe_{a.htf}.md").write_text(md, encoding="utf-8")
        (Path(a.out) / f"universe_{a.htf}.json").write_text(json.dumps(raw, ensure_ascii=False, default=float))
    else:
        raise SystemExit("حدّد --symbol أو --universe")
    print(md)


if __name__ == "__main__":
    main()
