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
        ix = (k["time"] - dates[0]) // DAY
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
def period_returns(score, turn30, O, dates, H, top_m, cost_leg, start_i, end_i, min_names=25, q=0.2):
    """Non-overlapping rebalances at rows start_i, start_i+H, ... < end_i. -> (dates, net return per unit of gross exposure)."""
    out_d, out_r = [], []
    t = start_i
    while t + 1 + H < min(end_i, len(dates)):
        sc, tv = score[t], turn30[t]
        elig = np.where(np.isfinite(sc) & np.isfinite(tv))[0]
        if len(elig) >= min_names:
            elig = elig[np.argsort(-tv[elig])][:top_m]
            r = O[t + 1 + H, elig] / O[t + 1, elig] - 1
            ok = np.isfinite(r)
            elig, r, s_ = elig[ok], r[ok], sc[elig][ok]
            if len(elig) >= min_names:
                kq = max(1, int(np.ceil(q * len(elig))))
                order = np.argsort(s_)
                short, long_ = r[order[:kq]].mean(), r[order[-kq:]].mean()
                out_d.append(dates[t])
                out_r.append((long_ - short) / 2.0 - cost_leg)
        t += H
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


def evaluate(a, cfg, client):
    now = int(time.time())
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
        res["configs"].append(row)
    rng = np.random.default_rng(a.seed)
    for seed in range(a.control_seeds):
        score = rng.random(C.shape)
        for H in HOLDS:
            row = {"seed": seed, "H": H}
            for w, (s_, e_) in windows.items():
                _, r = period_returns(score, turn30, O, dates, H, a.top_m, cost_leg, s_, e_)
                row[w] = block(r)
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
    L.append("\n## الحكم الآلي\n")
    L.append(f"- مرشّحون (موجب في التصميم والاختيار، وt ≥ 2.9 في الاختيار): {len(cands)}")
    if found:
        L.append(f"**وُجد {len(found)} تركيب يبقى موجباً على التحقق النهائي (t ≥ 1.96):**")
        for r in found:
            L.append(f"- {r['signal']} / H={r['H']}: {fmt(r['holdout'])}")
    else:
        L.append("**لم يوجد أي تركيب يحقق الشروط كلها.**")
    return "\n".join(L), cands, found


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
    a = ap.parse_args()
    cfg = load_config(a.config)
    a.fee = cfg.trading.taker_fee_pct if a.fee is None else a.fee
    client = MexcClient(cfg.mexc)
    res = evaluate(a, cfg, client)
    md, cands, found = report(res, a)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "xsec.md").write_text(md, encoding="utf-8")
    (Path(a.out) / "xsec.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
    print(md)


if __name__ == "__main__":
    main()
