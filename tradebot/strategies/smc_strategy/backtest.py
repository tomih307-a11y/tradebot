"""Universe walk-forward of the SMC strategy with the validation protocol of docs/SMC_STRATEGY.md."""
from __future__ import annotations

import csv
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tradebot.data.klines import get_klines
from tradebot.indicators import atr_rma

from . import model as M

M15 = 900


def day_boot(day, R, n_boot=1000, seed=0):
    """(2.5th, 97.5th) percentile of the mean R, resampling whole days."""
    if len(R) < 5:
        return (float("nan"), float("nan"))
    u, inv = np.unique(day, return_inverse=True)
    sums, cnt = np.bincount(inv, weights=R), np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n_boot, len(u)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def block(trades):
    """Window statistics. 'required_hit' = target-hit rate that makes the expectancy zero for THIS mix of outcomes."""
    if not trades:
        return None
    R = np.array([t["R_net"] for t in trades])
    out = np.array([t["outcome"] for t in trades])
    T = np.array([t["T"] for t in trades])
    hit = out == 2
    rt = float(R[hit].mean()) if hit.any() else 0.0
    rn = float(R[~hit].mean()) if (~hit).any() else 0.0
    lo, hi = day_boot(T // 86400, R)
    return {"n": len(R), "mean": float(R.mean()), "ci_lo": lo, "ci_hi": hi, "gross": float(np.mean([t["R_gross"] for t in trades])),
            "hit": float(hit.mean()), "stop": float((out == 1).mean()), "time": float((out == 0).mean()),
            "required_hit": float(-rn / (rt - rn)) if rt > rn else float("nan"), "r_target": rt, "r_other": rn,
            "risk_pct": float(np.mean([t["risk_pct"] for t in trades])), "rr_plan": float(np.mean([t["rr_plan"] for t in trades]))}


def random_baseline(trades, o, h, l, c, side, fee, slip, rng, lo_i, per_trade=5):
    """For each real trade: random entry bars on the same contract with the same relative stop distance and planned reward/risk."""
    s = 1 if side == "long" else -1
    n = len(c)
    out = []
    for t in trades:
        d = t["risk_pct"] / 100.0
        for j in rng.integers(lo_i, n - M.MAX_BARS - 3, size=per_trade):
            px = float(o[j + 1])
            stop, tgt = px * (1 - s * d), px * (1 + s * t["rr_plan"] * d)
            r = M.sim_fixed(s, int(j + 1), stop, tgt, o, h, l, c, fee, slip)
            if r:
                out.append({**r, "T": int(0), "rr_plan": t["rr_plan"]})
    return out


def run_universe(cfg, client, days=150, universe=5000, out_dir="reports/smc_model", fee=None, slip=0.05, seed=19, log=print):
    fee = cfg.trading.taker_fee_pct if fee is None else fee
    now = int(time.time())
    start = now - days * 86400
    t1, t2 = start + (now - start) // 2, start + (now - start) * 3 // 4
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    symbols = [t["symbol"] for t in tick[:universe]]
    rank = {s: i + 1 for i, s in enumerate(symbols)}
    dd = cfg.pmo_zero_cross.backtest.data_dir
    rng = np.random.default_rng(seed)
    real, base, used = {"long": [], "short": []}, {"long": [], "short": []}, 0
    for n_, sym in enumerate(symbols, 1):
        try:
            k15 = get_klines(client, dd, sym, "Min15", start - 10 * 86400, now)
            k4 = get_klines(client, dd, sym, "Hour4", start - 60 * 86400, now)
        except Exception as e:
            log(f"  {sym}: {e}")
            continue
        if len(k15["close"]) < 1500 or len(k4["close"]) < 120:
            continue
        used += 1
        o, h, l, c, t15 = (np.asarray(k15[f], float) for f in ("open", "high", "low", "close", "time"))
        atr = atr_rma(h, l, c, 14)
        lo_i = int(np.searchsorted(t15, start))
        for side in ("long", "short"):
            tr = M.find_trades(o, h, l, c, atr, t15.astype(int), k4, side, fee, slip, start_ts=start)
            for t in tr:
                t.update(symbol=sym, rank=rank[sym])
            real[side] += tr
            for b in random_baseline(tr, o, h, l, c, side, fee, slip, rng, max(lo_i, 50)):
                b["T"] = int(t15[min(b["exit_i"], len(t15) - 1)])          # baseline trades are dated by their exit (used only for the window split)
                base[side].append(b)
        if n_ % 50 == 0:
            log(f"  [{n_}/{len(symbols)}] {sym} trades long={len(real['long'])} short={len(real['short'])}")
    res = {"meta": {"symbols": used, "days": days, "fee": fee, "slip": slip, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}, "sides": {}}
    for side in ("long", "short"):
        win = {"design": lambda t: t["T"] < t1, "selection": lambda t: t1 <= t["T"] < t2, "holdout": lambda t: t["T"] >= t2}
        d = {w: block([t for t in real[side] if f(t)]) for w, f in win.items()}
        b = {w: block([t for t in base[side] if f(t)]) for w, f in win.items()}
        tiers = {}
        for nm, a, z in (("1-40", 1, 40), ("41-150", 41, 150), ("151-400", 151, 400), ("401+", 401, 10 ** 6)):
            xs = [t for t in real[side] if t["T"] >= t2 and a <= t["rank"] <= z]
            tiers[nm] = block(xs)
        h_ = d["holdout"]
        ok = bool(h_ and h_["n"] >= 200 and h_["mean"] > 0 and h_["ci_lo"] > 0 and b["holdout"] and h_["mean"] > b["holdout"]["mean"]
                  and d["design"] and d["design"]["mean"] > 0 and d["selection"] and d["selection"]["mean"] > 0)
        res["sides"][side] = {"real": d, "baseline": b, "tiers": tiers, "passed": ok, "insufficient": bool(not h_ or h_["n"] < 200)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "smc_model.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
    (Path(out_dir) / "smc_model.md").write_text(report(res), encoding="utf-8")
    with gzip.open(Path(out_dir) / "trades.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "rank", "side", "T", "R_net", "R_gross", "outcome", "risk_pct", "rr_plan"])
        for side in ("long", "short"):
            for t in real[side]:
                w.writerow([t["symbol"], t["rank"], side, t["T"], f"{t['R_net']:.4f}", f"{t['R_gross']:.4f}", t["outcome"], f"{t['risk_pct']:.3f}", f"{t['rr_plan']:.2f}"])
    return res


def fmt(b):
    if not b:
        return "–"
    return f"{b['n']} / هدف {b['hit'] * 100:.0f}% (لازم {b['required_hit'] * 100:.0f}%) / {b['mean']:+.3f} [{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}]"


def report(res):
    m = res["meta"]
    L = ["# استراتيجية SMC: اختبار على كل عقود MEXC", "",
         f"- {m['symbols']} عقداً · {m['days']} يوماً · {m['generated']} · رسوم Taker {m['fee']}% + انزلاق {m['slip']}% · القواعد: `docs/SMC_STRATEGY.md` (مثبّتة قبل الاختبار).",
         "- لكل نافذة: عدد الصفقات / نسبة إصابة الهدف (اللازمة للتعادل من نفس الصفقات) / متوسط صافي R [حد ثقة 95% بإعادة عيّنة بالأيام].", ""]
    for side, name in (("long", "شراء"), ("short", "بيع")):
        s = res["sides"][side]
        L.append(f"## {name}")
        L.append("| | تصميم | اختيار | تحقق نهائي |\n|---|---|---|---|")
        L.append(f"| SMC | {fmt(s['real']['design'])} | {fmt(s['real']['selection'])} | {fmt(s['real']['holdout'])} |")
        L.append(f"| شاهد عشوائي (نفس الوقف ونسبة العائد) | {fmt(s['baseline']['design'])} | {fmt(s['baseline']['selection'])} | {fmt(s['baseline']['holdout'])} |")
        h = s["real"]["holdout"]
        if h:
            L.append(f"\n- على التحقق النهائي: وقف متوسط {h['risk_pct']:.2f}%، نسبة العائد/المخاطرة المخططة {h['rr_plan']:.2f}، إجمالي R قبل التكاليف {h['gross']:+.3f}، وقف {h['stop'] * 100:.0f}%، خروج زمني {h['time'] * 100:.0f}%.")
        L.append("\n| شريحة السيولة (تحقق نهائي) | صفقات | صافي R | هدف % |\n|---|---|---|---|")
        for nm, b in s["tiers"].items():
            L.append(f"| {nm} | {b['n']} | {b['mean']:+.3f} | {b['hit'] * 100:.0f} |" if b else f"| {nm} | 0 | – | – |")
        verdict = "اجتازت" if s["passed"] else ("غير حاسم: أقل من 200 صفقة في التحقق النهائي" if s["insufficient"] else "لم تجتز")
        L.append(f"\n**الحكم المسجّل:** {verdict} (المعيار: تحقق نهائي n ≥ 200، متوسط > 0، حد ثقة أدنى > 0، أعلى من الشاهد، وموجبة في التصميم والاختيار)\n")
    return "\n".join(L)
