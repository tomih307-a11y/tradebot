"""SMC study v2 — corrected score tiers and a strict selection/validation split.

Pre-registered grid (fixed BEFORE looking at any v2 result): swing_len {3,5} x score tier {>=3,>=4,>=5 of 6 checks} x targets
{1R/2R/3R equal thirds, single 2R, single 3R} = 18 variants, no breakeven. The best variant by mean net R on the SELECTION window
(first 60% of the period, min 150 trades) is the ONLY one carried to the held-out VALIDATION window (last 40%, after a 72h embargo).
Reported with a day-block bootstrap CI, next to the same-distance market/reverse baselines. Gross (pre-cost) R is shown too.
Usage: python tools/smc_study2.py --out reports/smc2 --days 150 --top 40
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smc_study import generate, run_variant, summarize        # noqa: E402
from tradebot.config import load_config                       # noqa: E402
from tradebot.mexc.client import MexcClient                    # noqa: E402

TIERS = {3: 3 / 6, 4: 4 / 6, 5: 5 / 6}
TPS = {"3 أهداف 1R/2R/3R": [1.0, 2.0, 3.0], "هدف واحد 2R": [2.0], "هدف واحد 3R": [3.0]}
SWINGS = (3, 5)
MIN_SEL_TRADES = 150


def tier_min(k: int) -> float:
    return TIERS[k] - 1e-9                       # 4/6 = 0.6667 must pass a ">= 4 checks" tier (float-safe)


def boot_ci(trades, n_boot=2000, seed=0):
    """95% CI of mean net R, resampling whole calendar days (trades on one day are correlated)."""
    if not trades:
        return (float("nan"), float("nan"))
    days = {}
    for t in trades:
        days.setdefault(t["T"] // 86400, []).append(t["R_net"])
    sums = np.array([sum(v) for v in days.values()])
    cnt = np.array([len(v) for v in days.values()], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def stats(trades):
    sm = summarize([t["R_net"] for t in trades])
    if not sm:
        return None
    sm["gross"] = float(np.mean([t["R_gross"] for t in trades]))
    return sm


def row(label, sm, extra=""):
    if not sm:
        return f"| {label} | 0 | - | - | - | - | - |{extra}"
    return (f"| {label} | {sm['n']} | {sm['mean']:+.3f} | {sm['gross']:+.3f} | {sm['win']:.0f}% | {sm['pf']:.2f} | {sm['t']:+.2f} |{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/smc2")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--expire", type=int, default=8)
    ap.add_argument("--hold", type=int, default=72)
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--sel-pct", type=int, default=60)
    a = ap.parse_args()
    cfg = load_config(a.config)
    client = MexcClient(cfg.mexc)
    now = int(time.time())
    eval_start = now - a.days * 86400
    split = eval_start + (now - eval_start) * a.sel_pct // 100
    val_lo = split + a.hold * 3600                                  # embargo: no validation trade can overlap selection trades
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    symbols = [t["symbol"] for t in tick[: a.top]]
    print(f"symbols={len(symbols)} days={a.days} split={split}", flush=True)

    datasets = {}
    for sw in SWINGS:
        print(f"-- generating setups swing_len={sw}", flush=True)
        datasets[sw], diag = generate(cfg, client, symbols, eval_start, now, a, overrides={"swing_len": sw})
        print(diag, flush=True)

    grid = [(sw, k, tn, tp) for sw in SWINGS for k in TIERS for tn, tp in TPS.items()]
    sel = {}
    for sw, k, tn, tp in grid:
        tr = run_variant(datasets[sw], cfg, tier_min(k), False, "zone", a, tp_r=tp, t_hi=split)
        sel[(sw, k, tn)] = (stats(tr), tp)
    ok = {key: v for key, v in sel.items() if v[0] and v[0]["n"] >= MIN_SEL_TRADES}
    chosen = max(ok, key=lambda kk: ok[kk][0]["mean"]) if ok else None

    L = ["# اختبار SMC v2 — تصحيح الدرجات + فصل بيانات الاختيار عن التحقق", ""]
    L.append(f"- العقود: أعلى {a.top} سيولة (انحياز بقاء ممكن) · آخر {a.days} يوماً · **اختيار:** أول {a.sel_pct}% · **تحقق:** آخر {100 - a.sel_pct}% بعد فجوة {a.hold} ساعة")
    L.append(f"- الشبكة المسجَّلة مسبقاً: {len(grid)} متغيراً (swing_len {list(SWINGS)} × درجة ≥3/4/5 من 6 × أهداف {list(TPS)})؛ بدون BE. الاختيار بمتوسط R الصافي على نافذة الاختيار فقط (حد أدنى {MIN_SEL_TRADES} صفقة).")
    L.append(f"- دخول بأمر حد عند أول حافة للمنطقة (صلاحية {a.expire}س)، احتفاظ أقصى {a.hold}س، رسوم Taker {cfg.trading.taker_fee_pct}% للجهتين + انزلاق {a.slip}% على الخروج السوقي. «إجمالي» = قبل الرسوم والانزلاق.")
    L.append("- **غير مُحاكى:** تأكيد الدقيقة، الأخبار، تصويت الوكلاء الآخرين، التمويل.\n")
    L.append("## 1) نافذة الاختيار (كل المتغيرات)\n")
    L.append("| المتغير | الصفقات | صافي R | إجمالي R | ربح % | PF | t | |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, (sm, tp) in sel.items():
        sw, k, tn = key
        L.append(row(f"swing {sw} · درجة≥{k} · {tn}", sm, " ✅ المختار |" if key == chosen else " |"))
    best_vals = [v[0]["mean"] for v in ok.values()]
    L.append(f"\n- متوسط صافي R عبر كل المتغيرات المؤهلة (نافذة الاختيار): {np.mean(best_vals):+.3f}؛ الأفضل: {max(best_vals):+.3f}" if best_vals else "\n- لا متغير بلغ الحد الأدنى للصفقات.")
    verdict = "لا متغير مؤهل"
    if chosen:
        sw, k, tn = chosen
        tp = TPS[tn]
        ds = datasets[sw]
        val = {m: run_variant(ds, cfg, tier_min(k), False, m, a, tp_r=tp, t_lo=val_lo) for m in ("zone", "market", "reverse")}
        L.append("\n## 2) التحقق على بيانات لم تُستخدم في الاختيار (المتغير المختار فقط)\n")
        L.append(f"المختار: **swing {sw} · درجة ≥{k} · {tn}**\n")
        L.append("| النمط | الصفقات | صافي R | إجمالي R | ربح % | PF | t | |\n|---|---|---|---|---|---|---|---|")
        names = {"zone": "SMC (حد عند المنطقة)", "market": "أساس: سوق بنفس الوقف", "reverse": "أساس: عكس الاتجاه"}
        sms = {}
        for m in ("zone", "market", "reverse"):
            sms[m] = stats(val[m])
            L.append(row(names[m], sms[m]))
        lo, hi = boot_ci(val["zone"])
        z = sms["zone"]
        L.append(f"\n- فترة ثقة 95% (إعادة عيّنة بالأيام) لمتوسط صافي R لـSMC في التحقق: [{lo:+.3f} ، {hi:+.3f}]")
        if z:
            risk = float(np.mean([t["risk_pct"] for t in val["zone"]]))
            L.append(f"- مسافة الوقف المتوسطة {risk:.2f}% ⇒ بهامش 10% ورافعة 5: ≈ {z['mean'] * risk / 100 * 5 * 10:+.3f}% من الرصيد للصفقة")
        L.append("\n## 3) كل المتغيرات على نافذة التحقق (للشفافية فقط — لا يُبنى عليها اختيار)\n")
        L.append("| المتغير | الصفقات | صافي R | إجمالي R | ربح % | PF | t | |\n|---|---|---|---|---|---|---|---|")
        allv = []
        for sw2, k2, tn2, tp2 in grid:
            sm2 = stats(run_variant(datasets[sw2], cfg, tier_min(k2), False, "zone", a, tp_r=tp2, t_lo=val_lo))
            L.append(row(f"swing {sw2} · درجة≥{k2} · {tn2}", sm2))
            if sm2:
                allv.append(sm2["mean"])
        L.append(f"\n- متوسط صافي R عبر الـ{len(allv)} متغيراً في التحقق: {np.mean(allv):+.3f} ؛ عدد الموجب منها: {sum(v > 0 for v in allv)}")
        passed = bool(z and z["mean"] > 0 and lo > 0 and sms["reverse"] and z["mean"] > sms["reverse"]["mean"])
        verdict = "اجتاز (متوسط موجب، حد ثقة أدنى موجب، يتفوق على العكس)" if passed else "لم يجتز: " + (
            "متوسط صافي R غير موجب" if not (z and z["mean"] > 0) else "حد الثقة الأدنى ≤ 0 أو لا يتفوق على العكس")
        if val["zone"]:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            with gzip.open(Path(a.out) / "validation_trades.csv.gz", "wt", newline="") as f:
                w = csv.writer(f)
                w.writerow(["symbol", "T", "side", "score", "R_net", "R_gross", "reason", "tp_hit", "risk_pct"])
                for t in val["zone"]:
                    w.writerow([t["symbol"], t["T"], t["side"], f"{t['score']:.2f}", f"{t['R_net']:.4f}", f"{t['R_gross']:.4f}", t["reason"], t["tp_hit"], f"{t['risk_pct']:.3f}"])
    L.append(f"\n## الحكم الآلي\n\n**{verdict}**  (المعيار: متوسط صافي R > 0، والحد الأدنى لفترة الثقة > 0، ويتفوق على أساس العكس)")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
