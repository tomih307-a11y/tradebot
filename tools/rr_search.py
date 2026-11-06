"""Risk/reward x win-rate search with honest out-of-sample validation.

Question: is there ANY (entry signal, stop distance, reward:risk target) whose win rate beats the break-even win rate, net of costs?
For a bracket trade with stop = k x ATR(1H) and target = RR x that distance, expectancy (in R) = w*RR - (1-w) - c, where c = costs in R.
Break-even win rate  w* = (1 + c) / (1 + RR).  A "positive agreement" = observed w clearly above w*.

Protocol (fixed before looking at results):
  * Signals (long and short, closed candles, entry at the next 15m open + slippage, one signal per 24h per symbol and side):
      pmo4_1h   PMO(4,20,10) 15m crosses above its signal with both > 0 and 1H PMO > 0   (the user's rule; mirrored for shorts)
      pmo4_4h   same with the 4H PMO
      pmo4_touch  first bar where 15m PMO(4) reaches the zero line from below (above for shorts)
      pmo35_1h  same as pmo4_1h with PMO(35)
      break24   close beyond the previous 24h high (low for shorts)
      rsi_rev   RSI(14) crosses back above 25 (below 75 for shorts)
      random    random bars (CONTROL: shows how many 'positive' configs appear by chance)
  * Exits: stop k x ATR(1H), k in {1, 2}; target RR x stop distance, RR in {0.5, 1, 1.5, 2, 3}; time exit after 24h; adverse-first inside a candle.
  * Windows by entry time: DESIGN first 50%, SELECTION next 25%, HOLDOUT last 25% of the period.
  * A config is a CANDIDATE only if expectancy > 0 in DESIGN and in SELECTION and the SELECTION day-block CI lower bound > 0 at a Bonferroni level.
    Candidates are then read on the HOLDOUT once. 'Found' means a candidate also has HOLDOUT expectancy > 0 with CI lower bound > 0.
Usage: python tools/rr_search.py --universe 5000 --days 150 --out reports/rr
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
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from timemachine import find_signals, load_coin, prepare                 # noqa: E402
from tradebot.config import load_config                                  # noqa: E402
from tradebot.indicators import rsi                                       # noqa: E402
from tradebot.mexc.client import MexcClient                               # noqa: E402

M15 = 900
BARS = 96                         # 24h time limit and cooldown
FAMILIES = ["pmo4_1h", "pmo4_4h", "pmo4_touch", "pmo35_1h", "break24", "rsi_rev", "random"]
KS = (1.0, 2.0)
RRS = (0.5, 1.0, 1.5, 2.0, 3.0)


# ───────────────────────────── signals ─────────────────────────────
def cooldown(idx, gap=BARS):
    out, last = [], -10 ** 9
    for j in idx:
        if j - last >= gap:
            out.append(int(j))
            last = j
    return np.array(out, dtype=int)


def touch_signals(p, s):
    x = s * p
    prev = np.roll(x, 1)
    ok = (prev < 0) & (x >= 0)
    ok[0] = False
    return np.where(ok & ~np.isnan(x) & ~np.isnan(prev))[0]


def breakout_signals(close, high, low, s, n=BARS):
    idx = []
    if s == 1:
        roll = sliding_window_view(high, n)[:-1].max(axis=1)           # max of the previous n highs, aligned to bar n..
        ok = close[n:] > roll
    else:
        roll = sliding_window_view(low, n)[:-1].min(axis=1)
        ok = close[n:] < roll
    return np.where(ok)[0] + n


def rsi_signals(r, s):
    prev = np.roll(r, 1)
    ok = (prev < 25) & (r >= 25) if s == 1 else (prev > 75) & (r <= 75)
    ok[0] = False
    return np.where(ok & ~np.isnan(r) & ~np.isnan(prev))[0]


# ───────────────────────────── vectorised bracket simulation ─────────────────────────────
def bracket(s, j, o, h, l, c, atr_sig, k, rr, fee_pct, slip_pct, min_stop_pct=0.3):
    """Entries decided at bar j (array), filled at the open of j+1. -> (R_net, outcome) arrays.
    outcome: 0 time exit, 1 stop, 2 target. Entry/exit slippage applied to market-type fills; the target is a limit fill at its price."""
    j0 = j + 1
    W = BARS
    O = sliding_window_view(o, W)[j0]
    H = sliding_window_view(h, W)[j0]
    L = sliding_window_view(l, W)[j0]
    C = sliding_window_view(c, W)[j0]
    px = O[:, 0] * (1 + s * slip_pct / 100.0)
    d = np.maximum(k * atr_sig, min_stop_pct / 100.0 * px)        # a stop inside the noise floor makes R meaningless (stale/illiquid contracts)
    stop = px - s * d
    tgt = px + s * rr * d
    adverse = L if s == 1 else H
    favour = H if s == 1 else L
    hit_s = s * (adverse - stop[:, None]) <= 0
    hit_t = s * (favour - tgt[:, None]) >= 0
    big = W
    fs = np.where(hit_s.any(1), hit_s.argmax(1), big)
    ft = np.where(hit_t.any(1), hit_t.argmax(1), big)
    stopped = (fs <= ft) & (fs < big)                                 # a candle touching both counts as the stop
    targeted = (ft < fs) & (ft < big)
    rows = np.arange(len(j))
    gap_open = O[rows, np.minimum(fs, W - 1)]
    stop_fill = np.where((fs > 0) & (s * (gap_open - stop) <= 0), gap_open, stop) * (1 - s * slip_pct / 100.0)
    time_fill = C[:, -1] * (1 - s * slip_pct / 100.0)
    exit_px = np.where(stopped, stop_fill, np.where(targeted, tgt, time_fill))
    gross = s * (exit_px - px) / d
    fees = fee_pct / 100.0 * (px + exit_px) / d
    out = np.where(stopped, 1, np.where(targeted, 2, 0))
    return gross - fees, out, fees


# ───────────────────────────── statistics ─────────────────────────────
def day_boot(day, R, n_boot=1000, seed=0):
    """95% CI of mean R, resampling whole days. day = integer day index per trade."""
    if len(R) < 5:
        return (float("nan"), float("nan"))
    u, inv = np.unique(day, return_inverse=True)
    sums, cnt = np.bincount(inv, weights=R), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def stat_block(T, R, out, alpha=0.05):
    n = len(R)
    if n == 0:
        return None
    win = R > 0
    hit = out == 2
    lo, hi = day_boot(T // 86400, R)
    rt = float(R[hit].mean()) if hit.any() else 0.0                  # mean net R when the target is hit
    rn = float(R[~hit].mean()) if (~hit).any() else 0.0              # mean net R of every other outcome (stops and time exits)
    req = -rn / (rt - rn) if rt > rn else float("nan")               # target-hit rate that makes expectancy exactly zero for THIS mix of outcomes
    return {"n": int(n), "win": float(win.mean()), "exp": float(R.mean()), "ci_lo": lo, "ci_hi": hi,
            "avg_win": float(R[win].mean()) if win.any() else 0.0, "avg_loss": float(R[~win].mean()) if (~win).any() else 0.0,
            "p_target": float(hit.mean()), "p_stop": float((out == 1).mean()), "p_time": float((out == 0).mean()),
            "r_target": rt, "r_other": rn, "required_target": float(req)}


def run(a, cfg, client):
    ind = cfg.pmo_zero_cross.indicator
    now = int(time.time())
    start = now - a.days * 86400
    t1, t2 = start + (now - start) * 50 // 100, start + (now - start) * 75 // 100
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    symbols = [t["symbol"] for t in tick[: a.universe]]
    rng = np.random.default_rng(a.seed)
    acc = {}                                      # (family, side, k, rr) -> lists
    cost_c = {}
    dd = cfg.pmo_zero_cross.backtest.data_dir
    used = 0
    for n_, sym in enumerate(symbols, 1):
        try:
            k15, k1, k4 = load_coin(client, dd, sym, now, a.days, "Hour4")
            if len(k15["close"]) < int(ind.warmup_bars) + 800 or len(k1["close"]) < 400 or k4 is None or len(k4["close"]) < 400:
                continue
            p4, s4, ph1, atr15 = prepare(k15, k1, k4, ind, 4, "Min60", int(cfg.agents.technical.atr_len))
            _, _, ph4, _ = prepare(k15, k1, k4, ind, 4, "Hour4", int(cfg.agents.technical.atr_len))
            p35, s35, ph35, _ = prepare(k15, k1, k4, ind, 35, "Min60", int(cfg.agents.technical.atr_len))
        except Exception as e:
            print(f"  {sym}: {e}", flush=True)
            continue
        used += 1
        o, h, l, c, tm = k15["open"], k15["high"], k15["low"], k15["close"], k15["time"]
        n = len(c)
        rs = rsi(c, 14)
        lo_ok = int(ind.warmup_bars) + 1
        for s in (1, -1):
            fam = {
                "pmo4_1h": find_signals(p4, s4, ph1, s),
                "pmo4_4h": find_signals(p4, s4, ph4, s),
                "pmo4_touch": touch_signals(p4, s),
                "pmo35_1h": find_signals(p35, s35, ph35, s),
                "break24": breakout_signals(c, h, l, s),
                "rsi_rev": rsi_signals(rs, s),
                "random": np.where(rng.random(n) < 1 / 60.0)[0],
            }
            for name, idx in fam.items():
                idx = idx[(idx > lo_ok) & (idx + 1 + BARS < n) & (tm[np.minimum(idx, n - 1)] + M15 >= start)]
                idx = cooldown(idx[np.isfinite(atr15[idx])])
                if len(idx) == 0:
                    continue
                for k, rr in product(KS, RRS):
                    R, out, fees = bracket(s, idx, o, h, l, c, atr15[idx], k, rr, a.fee, a.slip)
                    key = (name, "long" if s == 1 else "short", k, rr)
                    d = acc.setdefault(key, {"T": [], "R": [], "out": [], "fee": []})
                    d["T"].append(tm[idx] + M15); d["R"].append(R); d["out"].append(out); d["fee"].append(fees)
        if n_ % 100 == 0:
            print(f"  [{n_}/{len(symbols)}] {sym}", flush=True)
    res = []
    for key, d in acc.items():
        T, R, out, fee = (np.concatenate(d[x]) for x in ("T", "R", "out", "fee"))
        row = {"family": key[0], "side": key[1], "k": key[2], "rr": key[3], "cost_R": float(fee.mean())}
        for nm, m in (("design", T < t1), ("selection", (T >= t1) & (T < t2)), ("holdout", T >= t2)):
            row[nm] = stat_block(T[m], R[m], out[m])
        res.append(row)
    return res, used, (start, t1, t2, now)


def report(res, used, bounds, a):
    n_cfg = len(res)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - 0.05 / n_cfg / 2)           # Bonferroni two-sided
    L = [f"# بحث المخاطرة/العائد ونسبة الفوز ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})", "",
         f"- {used} عقداً · {a.days} يوماً · نوافذ بحسب وقت الدخول: تصميم أول 50%، اختيار الـ25% التالية، **تحقق نهائي** آخر 25% · رسوم Taker {a.fee}% للجهتين + انزلاق {a.slip}% · إشارة واحدة كل 24 ساعة لكل عقد واتجاه.",
         f"- التجارب: {n_cfg} تركيباً (إشارة × اتجاه × وقف × نسبة هدف). تصحيح Bonferroni: z = {z:.2f} بدل 1.96.",
         "- **المرشّح** = توقع صافٍ موجب في التصميم وفي الاختيار معاً، وحد ثقة أدنى (95%) موجب في الاختيار. **الوجود** = مرشّح يبقى موجباً بحد ثقة أدنى موجب في التحقق النهائي.", ""]

    def cand(r):
        d, s = r["design"], r["selection"]
        return bool(d and s and d["exp"] > 0 and s["exp"] > 0 and s["ci_lo"] > 0)

    cands = [r for r in res if cand(r)]
    ctrl = [r for r in res if r["family"] == "random"]
    real = [r for r in res if r["family"] != "random"]
    pos = lambda rows: sum(1 for r in rows if r["selection"] and r["selection"]["exp"] > 0 and r["design"] and r["design"]["exp"] > 0)
    L.append("## ما يظهر بالصدفة (الإشارات العشوائية كشاهد)")
    L.append(f"- تراكيب عشوائية موجبة في التصميم والاختيار معاً: {pos(ctrl)} من {len(ctrl)} ({pos(ctrl) / max(1, len(ctrl)) * 100:.0f}%).")
    L.append(f"- تراكيب الإشارات الحقيقية الموجبة في النافذتين: {pos(real)} من {len(real)} ({pos(real) / max(1, len(real)) * 100:.0f}%).")
    L.append(f"- مرشّحون بحد ثقة موجب في الاختيار: {len(cands)} (عشوائي: {sum(1 for r in cands if r['family'] == 'random')}).\n")

    def line(r):
        f = lambda b: "–" if not b else f"{b['n']} / هدف {b['p_target'] * 100:.0f}% (لازم {b['required_target'] * 100:.0f}%) / {b['exp']:+.3f} [{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}]"
        return (f"| {r['family']} | {r['side']} | {r['k']:g}×ATR | 1:{r['rr']:g} | {f(r['design'])} | {f(r['selection'])} | {f(r['holdout'])} |")

    head = "| الإشارة | الاتجاه | الوقف | العائد | تصميم: n / إصابة الهدف (اللازمة للتعادل) / صافي R | اختيار | تحقق نهائي |\n|---|---|---|---|---|---|---|"
    L.append("## المرشّحون وقراءتهم على التحقق النهائي")
    if cands:
        L.append(head)
        for r in sorted(cands, key=lambda r: -r["selection"]["exp"])[:40]:
            L.append(line(r))
    else:
        L.append("لا يوجد مرشّح بهذه الشروط.")
    found = [r for r in cands if r["holdout"] and r["holdout"]["exp"] > 0 and r["holdout"]["ci_lo"] > 0]
    L.append("\n## الحكم الآلي\n")
    if found:
        L.append(f"**وُجد {len(found)} تركيب يبقى موجباً بحد ثقة موجب في التحقق النهائي** (راجعه بحذر: راجع الشاهد العشوائي أعلاه).")
        L.append(head)
        for r in sorted(found, key=lambda r: -r["holdout"]["exp"]):
            L.append(line(r))
    else:
        L.append("**لم يوجد أي تركيب** يحقق الشروط كلها: موجب في التصميم والاختيار بحد ثقة موجب، ثم موجب بحد ثقة موجب في التحقق النهائي.")
    L.append("\n## أفضل 15 تركيباً في نافذة الاختيار (للاطلاع فقط)\n")
    L.append(head)
    for r in sorted([r for r in res if r["selection"]], key=lambda r: -r["selection"]["exp"])[:15]:
        L.append(line(r))
    L.append("\n## نسبة إصابة الهدف مقابل اللازمة للتعادل (كل الإشارات الحقيقية، نافذتا التصميم والاختيار معاً)\n")
    L.append("| العائد | إصابة الهدف الفعلية | اللازمة للتعادل | الفرق (نقطة مئوية) | صافي R متوسط |\n|---|---|---|---|---|")
    for rr in RRS:
        rows = [r for r in real if r["rr"] == rr and r["selection"] and r["design"]]
        if rows:
            w = lambda key: np.mean([(r["design"][key] * r["design"]["n"] + r["selection"][key] * r["selection"]["n"]) / (r["design"]["n"] + r["selection"]["n"]) for r in rows])
            e, pt, rq = w("exp"), w("p_target"), w("required_target")
            L.append(f"| 1:{rr:g} | {pt * 100:.1f}% | {rq * 100:.1f}% | {(pt - rq) * 100:+.1f} | {e:+.3f} |")
    return "\n".join(L), found, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=int, default=5000)
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--out", default="reports/rr")
    a = ap.parse_args()
    cfg = load_config(a.config)
    a.fee = cfg.trading.taker_fee_pct if a.fee is None else a.fee
    client = MexcClient(cfg.mexc)
    res, used, bounds = run(a, cfg, client)
    md, found, cands = report(res, used, bounds, a)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "rr.md").write_text(md, encoding="utf-8")
    (Path(a.out) / "rr.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
    print(md)


if __name__ == "__main__":
    main()
