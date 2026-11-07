"""Cross-sectional long-short study on DAILY MEXC futures data (a different hypothesis class from the 15m timing rules).

Each rebalance day t (close): rank the liquid contracts by a signal, go LONG the top quintile and SHORT the bottom quintile with equal weights
(dollar-neutral, so the market's direction cancels), enter at the NEXT day's open and exit H days later at the open. Costs: taker fee + slippage
on entry and exit of every leg. Returns are per unit of GROSS exposure (long + short)/2, so one period is comparable to one trade.
Signals (fixed in advance, sign fixed in advance): mom7, mom30 (long winners), rev1, rev3 (long losers), lowvol30 (long low volatility).
Holding periods H in {1, 3, 7} days, non-overlapping rebalances. Control: random scores, 30 seeds -> how large a t-stat appears by chance.
Windows by date: DESIGN first 50%, SELECTION next 25%, HOLDOUT last 25%.
Candidate: expectancy > 0 in DESIGN and SELECTION, and t >= 2.9 in SELECTION (Bonferroni over the configs). Found: candidate also positive with t >= 1.96 on HOLDOUT.
Per window it reports win rate, payoff ratio (avg win / avg loss) and the win rate that payoff ratio REQUIRES for break-even: 1 / (1 + payoff).
Usage: python tools/xsec.py --top 250 --days 1000 --out reports/xsec
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.config import load_config                    # noqa: E402
from tradebot.data.klines import get_klines                 # noqa: E402
from tradebot.mexc.client import MexcClient                # noqa: E402

DAY = 86400
SIGNALS = ["mom7", "mom30", "rev1", "rev3", "lowvol30"]
HOLDS = (1, 3, 7)


# ───────────────────────────── features ─────────────────────────────
def build_matrices(series, t0, t1):
    """series: sym -> dict(time, open, close, amount). -> dates, O, C, A of shape (T, N); NaN where a contract has no candle."""
    dates = np.arange(t0 // DAY * DAY, t1 // DAY * DAY + DAY, DAY)
    T, N = len(dates), len(series)
    O, C, A = (np.full((T, N), np.nan) for _ in range(3))
    for i, (sym, k) in enumerate(series.items()):
        ix = ((np.asarray(k["time"], dtype=float) - dates[0]) // DAY).astype(int)       # cached candle times can be floats
        m = (ix >= 0) & (ix < T)
        O[ix[m], i], C[ix[m], i], A[ix[m], i] = k["open"][m], k["close"][m], k["amount"][m]
    return dates, O, C, A


def lag(x, n):
    out = np.full_like(x, np.nan)
    out[n:] = x[:-n]
    return out


def rolling(x, n, fn):
    out = np.full_like(x, np.nan)
    for t in range(n, len(x)):
        w = x[t - n + 1: t + 1]
        ok = np.isfinite(w).sum(0) >= int(n * 0.8)
        v = np.full(x.shape[1], np.nan)
        if ok.any():
            v[ok] = fn(w[:, ok])
        out[t] = v
    return out


def features(C, A):
    ret1 = C / lag(C, 1) - 1
    f = {"mom7": C / lag(C, 7) - 1, "mom30": C / lag(C, 30) - 1, "rev1": -ret1, "rev3": -(C / lag(C, 3) - 1),
         "lowvol30": -rolling(ret1, 30, lambda w: np.nanstd(w, axis=0))}
    turn30 = rolling(A, 30, lambda w: np.nanmean(w, axis=0))
    return f, turn30


# ───────────────────────────── portfolio ─────────────────────────────
def period_returns(score, turn30, O, dates, H, top_m, cost_leg, start_i, end_i, min_names=25, q=0.2, step=None, skip_m=0):
    """Rebalances at rows start_i, start_i+step, ... < end_i (step defaults to H = non-overlapping; step=1 = every start day, overlapping).
    -> (dates, net return per unit of gross exposure)."""
    out_d, out_r = [], []
    t = start_i
    while t + 1 + H < min(end_i, len(dates)):
        sc, tv = score[t], turn30[t]
        elig = np.where(np.isfinite(sc) & np.isfinite(tv))[0]
        if len(elig) >= min_names:
            elig = elig[np.argsort(-tv[elig])][skip_m: skip_m + top_m]          # skip_m > 0: liquidity ranks skip_m+1 .. skip_m+top_m (disjoint assets)
            r = O[t + 1 + H, elig] / O[t + 1, elig] - 1
            ok = np.isfinite(r)
            elig, r, s_ = elig[ok], r[ok], sc[elig][ok]
            if len(elig) >= min_names:
                kq = max(1, int(np.ceil(q * len(elig))))
                order = np.argsort(s_)
                short, long_ = r[order[:kq]].mean(), r[order[-kq:]].mean()
                out_d.append(dates[t])
                out_r.append((long_ - short) / 2.0 - cost_leg)
        t += (step or H)
    return np.array(out_d), np.array(out_r)


def block(r):
    n = len(r)
    if n < 10:
        return None
    win = r > 0
    aw = float(r[win].mean()) if win.any() else 0.0
    al = float(-r[~win].mean()) if (~win).any() else 0.0
    payoff = aw / al if al > 0 else float("nan")
    sd = r.std(ddof=1)
    return {"n": int(n), "win": float(win.mean()), "payoff": float(payoff), "required_win": float(1 / (1 + payoff)) if payoff == payoff else float("nan"),
            "exp_pct": float(r.mean() * 100), "t": float(r.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0,
            "sharpe": float(r.mean() / sd * np.sqrt(365.0 / 1)) if sd > 0 else 0.0}


def hac_t(r, L):
    """Newey-West t-statistic of the mean (Bartlett kernel, lag L): overlapping H-day holdings make neighbouring returns correlated."""
    n = len(r)
    if n < 30:
        return 0.0
    u = r - r.mean()
    s_ = (u * u).sum() / n
    for l in range(1, min(L, n - 1) + 1):
        s_ += 2 * (1 - l / (L + 1)) * (u[l:] * u[:-l]).sum() / n
    se = np.sqrt(max(s_, 1e-18) / n)
    return float(r.mean() / se)


def block_boot_ci(r, blk, n_boot=1000, seed=0):
    n = len(r)
    if n < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / blk))
    starts = rng.integers(0, n - blk + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(blk)[None, None, :]).reshape(n_boot, -1)[:, :n]
    m = r[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def block_ov(r, H):
    b = block(r)
    if not b:
        return None
    L = max(H, 1)
    b["t_hac"] = hac_t(r, 2 * L)
    b["ci_lo"], b["ci_hi"] = (v * 100 for v in block_boot_ci(r, max(2 * L, 5)))
    return b


def load_series(a, cfg, client, now):
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    symbols = [t["symbol"] for t in tick[: a.top]]
    dd = cfg.pmo_zero_cross.backtest.data_dir
    series = {}
    for n_, sym in enumerate(symbols, 1):
        try:
            k = get_klines(client, dd, sym, "Day1", now - a.days * DAY, now)
        except Exception as e:
            print(f"  {sym}: {e}", flush=True)
            continue
        if len(k["time"]) >= 120:
            series[sym] = k
        if n_ % 50 == 0:
            print(f"  [{n_}/{len(symbols)}] loaded={len(series)}", flush=True)
    return series


def replicate_from_matrices(dates, O, C, A, cutoff_ts, cost_leg, top_m, seed=19, control_seeds=30, H=7, skip_m=0):
    """Pre-registered replication (addendum A of docs/PREREG_FORWARD.md): H1 = mom30, H2 = mom7 at H=7, rows strictly before cutoff_ts."""
    f, turn30 = features(C, A)
    first = 40
    cut_i = int(np.searchsorted(dates, cutoff_ts))
    out = {"window": {"from": datetime.fromtimestamp(int(dates[first]), timezone.utc).strftime("%Y-%m-%d") if cut_i > first else None,
                      "to": datetime.fromtimestamp(int(dates[max(cut_i - 1, 0)]), timezone.utc).strftime("%Y-%m-%d"), "rows": max(cut_i - first, 0)},
           "configs": {}, "control": []}
    if cut_i - first < 60:
        out["note"] = "تاريخ غير كافٍ قبل التاريخ المحدد"
        return out
    elig = [max(int((np.isfinite(f["mom30"][t]) & np.isfinite(turn30[t])).sum()) - skip_m, 0) for t in range(first, cut_i)]      # names available in the requested rank band
    out["eligible"] = {"min": int(min(elig)), "median": int(np.median(elig)), "max": int(max(elig)), "rows_with_25plus": int(sum(e >= 25 for e in elig))}
    rng = np.random.default_rng(seed)
    ctrl_means, ctrl_t = [], []
    for sd in range(control_seeds):
        sc = rng.random(C.shape)
        _, r = period_returns(sc, turn30, O, dates, H, top_m, cost_leg, first, cut_i, step=1, skip_m=skip_m)
        b = block_ov(r, H)
        if b:
            ctrl_means.append(b["exp_pct"])
            ctrl_t.append(b["t_hac"])
    out["control"] = {"mean_pct": float(np.mean(ctrl_means)) if ctrl_means else None, "t_mean": float(np.mean(ctrl_t)) if ctrl_t else None,
                      "t_max": float(np.max(ctrl_t)) if ctrl_t else None}
    for name, thr in (("mom30", 1.96), ("mom7", 2.24)):
        _, r_ov = period_returns(f[name], turn30, O, dates, H, top_m, cost_leg, first, cut_i, step=1, skip_m=skip_m)
        _, r_no = period_returns(f[name], turn30, O, dates, H, top_m, cost_leg, first, cut_i, skip_m=skip_m)
        b_ov, b_no = block_ov(r_ov, H), block(r_no)
        verdict_ = "غير حاسمة: أقل من 90 فترة غير متداخلة" if (not b_no or b_no["n"] < 90) else "لم تنجح"
        if b_ov and b_no and b_no["n"] >= 90 and b_ov["exp_pct"] > 0 and b_ov["t_hac"] >= thr and b_ov["ci_lo"] > 0 \
                and out["control"]["mean_pct"] is not None and b_ov["exp_pct"] > out["control"]["mean_pct"]:
            verdict_ = "نجحت (حسب المعيار المسجّل)"
        out["configs"][name] = {"threshold_t": thr, "overlapping": b_ov, "non_overlapping": b_no, "verdict": verdict_}
    return out


def replicate(a, cfg, client):
    now = int(time.time())
    series = load_series(a, cfg, client, now)
    dates, O, C, A = build_matrices(series, now - a.days * DAY, now - DAY)
    cutoff = int(datetime.strptime(a.replication_before, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    cost_leg = 2 * (a.fee + a.slip) / 100.0
    res = replicate_from_matrices(dates, O, C, A, cutoff, cost_leg, a.top_m, a.seed, a.control_seeds, skip_m=a.skip_m)
    res["meta"] = {"skip_m": a.skip_m, "top_m": a.top_m, "slip": a.slip, "symbols": len(series), "history_from": datetime.fromtimestamp(int(dates[0]), timezone.utc).strftime("%Y-%m-%d"),
                   "cutoff": a.replication_before, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    return res


def replication_md(res):
    m = res["meta"]
    band = f"ترتيب السيولة {m['skip_m'] + 1}–{m['skip_m'] + m['top_m']}" if m.get("skip_m") else f"أعلى {m.get('top_m', 150)}"
    L = [f"# تكرار للزخم المقطعي ({band}) حتى {m['cutoff']} ({m['generated']})", "",
         f"- {m['symbols']} عقداً محمّلاً · أقدم يوم متاح: {m['history_from']} · نافذة التقييم: {res['window']['from']} → {res['window']['to']} ({res['window']['rows']} يوماً).",
         "- التسجيل المسبق: الإضافة أ في `docs/PREREG_FORWARD.md` (مثبّتة قبل تحميل هذه البيانات). H1 = mom30، H2 = mom7، فترة الاحتفاظ 7 أيام فقط."]
    if res.get("note"):
        return "\n".join(L + ["", f"**{res['note']}**"])
    e = res["eligible"]
    L.append(f"- العقود المؤهلة يومياً: أدنى {e['min']}، وسيط {e['median']}، أعلى {e['max']}؛ أيام فيها ≥25 عقداً: {e['rows_with_25plus']} من {res['window']['rows']}.")
    c = res["control"]
    if c and c.get("mean_pct") is not None:
        L.append(f"- الشاهد العشوائي (30 بذرة): متوسط {c['mean_pct']:+.3f}% لكل شريحة، t متوسط {c['t_mean']:+.2f}، أعلى t {c['t_max']:+.2f}.")
    L += ["", "| الفرضية | فترات غير متداخلة | فوز % (لازم) / payoff | متوسط صافٍ % (محافظ متداخلة) [فترة 95%] | t (Newey-West) | الشرط | الحالة |", "|---|---|---|---|---|---|---|"]
    for name, r in res["configs"].items():
        o, n = r["overlapping"], r["non_overlapping"]
        if o and n:
            L.append(f"| {'H1' if name == 'mom30' else 'H2'} {name} | {n['n']} | {n['win'] * 100:.0f}% ({n['required_win'] * 100:.0f}%) / {n['payoff']:.2f} | "
                     f"{o['exp_pct']:+.3f} [{o['ci_lo']:+.2f}, {o['ci_hi']:+.2f}] | {o['t_hac']:+.2f} | ≥ {r['threshold_t']} | {r['verdict']} |")
        else:
            L.append(f"| {name} | – | – | – | – | ≥ {r['threshold_t']} | {r['verdict']} |")
    return "\n".join(L)


def evaluate(a, cfg, client):
    now = int(time.time())
    series = load_series(a, cfg, client, now)
    dates, O, C, A = build_matrices(series, now - a.days * DAY, now - DAY)
    f, turn30 = features(C, A)
    first = 40                                                      # features need up to 30 days of history
    n_dates = len(dates)
    i1, i2 = first + (n_dates - first) // 2, first + (n_dates - first) * 3 // 4
    cost_leg = 2 * (a.fee + a.slip) / 100.0                         # entry + exit, fee + slippage, per leg
    res = {"meta": {"symbols": len(series), "days": n_dates, "from": datetime.fromtimestamp(int(dates[first]), timezone.utc).strftime("%Y-%m-%d"),
                    "to": datetime.fromtimestamp(int(dates[-1]), timezone.utc).strftime("%Y-%m-%d"),
                    "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "cost_leg_pct": cost_leg * 100}, "configs": [], "control": []}
    windows = {"design": (first, i1), "selection": (i1, i2), "holdout": (i2, n_dates)}
    for name, H in product(SIGNALS, HOLDS):
        row = {"signal": name, "H": H}
        for w, (s_, e_) in windows.items():
            _, r = period_returns(f[name], turn30, O, dates, H, a.top_m, cost_leg, s_, e_)
            row[w] = block(r)
            row[w + "_gross"] = float((r.mean() + cost_leg) * 100) if len(r) else None
        for w, (s_, e_) in windows.items():
            _, r = period_returns(f[name], turn30, O, dates, H, a.top_m, cost_leg, s_, e_, step=1)
            row["ov_" + w] = block_ov(r, H)
        res["configs"].append(row)
    rng = np.random.default_rng(a.seed)
    for seed in range(a.control_seeds):
        score = rng.random(C.shape)
        for H in HOLDS:
            row = {"seed": seed, "H": H}
            for w, (s_, e_) in windows.items():
                _, r = period_returns(score, turn30, O, dates, H, a.top_m, cost_leg, s_, e_)
                row[w] = block(r)
                _, r1 = period_returns(score, turn30, O, dates, H, a.top_m, cost_leg, s_, e_, step=1)
                row["ov_" + w] = block_ov(r1, H)
            res["control"].append(row)
    return res


def report(res, a):
    n_cfg = len(res["configs"])
    L = [f"# دراسة المقطع العرضي اليومي ({res['meta']['generated']})", "",
         f"- {res['meta']['symbols']} عقداً · {res['meta']['from']} → {res['meta']['to']} ({res['meta']['days']} يوماً) · أكثر {a.top_m} عقداً سيولة (حجم تداول 30 يوماً، محسوب بلا نظر للمستقبل) · عقود موجودة اليوم فقط (انحياز بقاء).",
         f"- المحفظة: لونق أعلى 20% وشورت أدنى 20% بأوزان متساوية (محايدة للدولار)، دخول عند فتح اليوم التالي، خروج بعد H أيام. التكلفة لكل ساق (دخول+خروج) = {res['meta']['cost_leg_pct']:.2f}%. العائد لكل وحدة تعرّض إجمالي.",
         f"- {n_cfg} تركيباً ثابتة سلفاً (5 إشارات × 3 فترات احتفاظ) + شاهد عشوائي ({a.control_seeds} بذرة × 3). نوافذ زمنية: تصميم 50%، اختيار 25%، **تحقق نهائي** 25%.", ""]
    cand = lambda r: bool(r["design"] and r["selection"] and r["design"]["exp_pct"] > 0 and r["selection"]["exp_pct"] > 0 and r["selection"]["t"] >= 2.9)
    ctrl_t = [c["selection"]["t"] for c in res["control"] if c["selection"]]
    L.append("## الشاهد العشوائي (ما يظهر بالصدفة)")
    if ctrl_t:
        L.append(f"- t في نافذة الاختيار لإشارات عشوائية: متوسط {np.mean(ctrl_t):+.2f}، أعلى قيمة {max(ctrl_t):+.2f}، أدنى {min(ctrl_t):+.2f}؛ نسبة ما تجاوز 2.9: {np.mean([t >= 2.9 for t in ctrl_t]) * 100:.1f}%.")

    def fmt(b):
        return "–" if not b else f"{b['n']} / فوز {b['win'] * 100:.0f}% (لازم {b['required_win'] * 100:.0f}%) / payoff {b['payoff']:.2f} / {b['exp_pct']:+.3f}% / t {b['t']:+.1f}"
    L.append("\n## كل التراكيب (صافي بعد التكاليف)\n")
    L.append("| الإشارة | H | تصميم: n / فوز (لازم) / payoff / توقع لكل فترة / t | اختيار | تحقق نهائي |\n|---|---|---|---|---|")
    for r in res["configs"]:
        L.append(f"| {r['signal']} | {r['H']} | {fmt(r['design'])} | {fmt(r['selection'])} | {fmt(r['holdout'])} |")
    cands = [r for r in res["configs"] if cand(r)]
    found = [r for r in cands if r["holdout"] and r["holdout"]["exp_pct"] > 0 and r["holdout"]["t"] >= 1.96]
    L.append("\n## قبل الرسوم والانزلاق (إجمالي لكل فترة، نافذة الاختيار) للمعرفة فقط\n")
    L.append("| الإشارة | H | إجمالي % | صافي % |\n|---|---|---|---|")
    for r in res["configs"]:
        if r["selection"]:
            L.append(f"| {r['signal']} | {r['H']} | {r['selection_gross']:+.3f} | {r['selection']['exp_pct']:+.3f} |")
    # same 15 configs, same thresholds, more efficient estimator: every start day (overlapping holdings), Newey-West t
    ov_ctrl = [c["ov_selection"]["t_hac"] for c in res["control"] if c.get("ov_selection")]
    L.append("\n## نفس التراكيب بمحافظ متداخلة (كل يوم بدء) وخطأ معياري Newey-West")
    if ov_ctrl:
        L.append(f"- الشاهد العشوائي بنفس الطريقة (نافذة الاختيار): متوسط t {np.mean(ov_ctrl):+.2f}، أعلى {max(ov_ctrl):+.2f}، نسبة ما تجاوز 2.9: {np.mean([t >= 2.9 for t in ov_ctrl]) * 100:.1f}%.")

    def fo(b):
        return "–" if not b else f"{b['n']} / فوز {b['win'] * 100:.0f}% (لازم {b['required_win'] * 100:.0f}%) / {b['exp_pct']:+.3f}% [{b['ci_lo']:+.2f},{b['ci_hi']:+.2f}] / t {b['t_hac']:+.1f}"
    L.append("\n| الإشارة | H | تصميم: n / فوز (لازم) / توقع [فترة 95%] / t | اختيار | تحقق نهائي |\n|---|---|---|---|---|")
    for r in res["configs"]:
        L.append(f"| {r['signal']} | {r['H']} | {fo(r.get('ov_design'))} | {fo(r.get('ov_selection'))} | {fo(r.get('ov_holdout'))} |")
    ov_cand = [r for r in res["configs"] if r.get("ov_design") and r.get("ov_selection") and r["ov_design"]["exp_pct"] > 0 and r["ov_selection"]["exp_pct"] > 0 and r["ov_selection"]["t_hac"] >= 2.9]
    ov_found = [r for r in ov_cand if r["ov_holdout"] and r["ov_holdout"]["exp_pct"] > 0 and r["ov_holdout"]["t_hac"] >= 1.96 and r["ov_holdout"]["ci_lo"] > 0]
    L.append("\n## الحكم الآلي\n")
    L.append(f"- (محافظ متداخلة) مرشّحون: {len(ov_cand)} · وُجد (موجب على التحقق النهائي بـ t ≥ 1.96 وحد ثقة أدنى موجب): {len(ov_found)}")
    for r in ov_found:
        L.append(f"  - **{r['signal']} / H={r['H']}**: {fo(r['ov_holdout'])}")
    L.append(f"- (محافظ غير متداخلة) مرشّحون (موجب في التصميم والاختيار، وt ≥ 2.9 في الاختيار): {len(cands)}")
    if found:
        L.append(f"**وُجد {len(found)} تركيب يبقى موجباً على التحقق النهائي (t ≥ 1.96):**")
        for r in found:
            L.append(f"- {r['signal']} / H={r['H']}: {fmt(r['holdout'])}")
    else:
        L.append("**لم يوجد أي تركيب يحقق الشروط كلها.**")
    return "\n".join(L), cands, ov_found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=250, help="contracts downloaded (by today's turnover)")
    ap.add_argument("--top-m", type=int, default=120, help="universe size per day (trailing 30d turnover rank)")
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--control-seeds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--out", default="reports/xsec")
    ap.add_argument("--skip-m", type=int, default=0, help="liquidity ranks skip_m+1 .. skip_m+top_m instead of 1 .. top_m")
    ap.add_argument("--replication-before", default=None, help="YYYY-MM-DD: run ONLY the pre-registered replication on rows before this date")
    a = ap.parse_args()
    cfg = load_config(a.config)
    a.fee = cfg.trading.taker_fee_pct if a.fee is None else a.fee
    client = MexcClient(cfg.mexc)
    if a.replication_before:
        res = replicate(a, cfg, client)
        md = replication_md(res)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "xsec_replication.md").write_text(md, encoding="utf-8")
        (Path(a.out) / "xsec_replication.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
        print(md)
        return
    res = evaluate(a, cfg, client)
    md, cands, found = report(res, a)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "xsec.md").write_text(md, encoding="utf-8")
    (Path(a.out) / "xsec.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
    print(md)


if __name__ == "__main__":
    main()
