"""Does adding Stochastic RSI at the PMO_15 zero-cross moment change the result?
Each PMO_15 cross is a candidate (signal direction s0). A condition on StochRSI(K, D) at the cross bar decides
whether the trade is taken; the trade is s0 (original) or -s0 (reverse). Exits: fixed horizon, no stop.
Costs/slippage/funding exactly as in reverse_study. Selection/validation split by time.
Usage: python tools/stochrsi_study.py --out reports/stochrsi
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit import feats_for                                   # noqa: E402
from reverse_study import simulate, summarize                 # noqa: E402
from tradebot.config import load_config, parse_time           # noqa: E402
from tradebot.indicators import stoch_rsi                     # noqa: E402
from tradebot.mexc.client import MexcClient                   # noqa: E402

HORIZONS = {"1h": 4, "4h": 16, "24h": 96}


def conditions(k, d, s0, os_, ob):
    """Boolean masks on the cross bars; s0 = signal direction (+1 up-cross, -1 down-cross)."""
    ok = ~np.isnan(k) & ~np.isnan(d)
    with_d = s0 * (k - d) > 0                                   # K on the signal's side of D
    stretched_against = np.where(s0 == 1, k < os_, k > ob)      # long & oversold / short & overbought
    stretched_with = np.where(s0 == 1, k > ob, k < os_)         # long & overbought / short & oversold
    mid = (k >= os_) & (k <= ob)
    return {
        "بدون شرط": np.ones(len(k), bool),
        "K مع اتجاه الإشارة (فوق D للونق)": ok & with_d,
        "K ضد اتجاه الإشارة": ok & ~with_d,
        "تشبع عكسي (لونق وK<20 / شورت وK>80)": ok & stretched_against,
        "تشبع مع الاتجاه (لونق وK>80 / شورت وK<20)": ok & stretched_with,
        "منطقة وسطى (20-80)": ok & mid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/stochrsi")
    ap.add_argument("--config", default="config.toml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    bt, sr = cfg.pmo_zero_cross.backtest, cfg.research.stochrsi
    start, end = parse_time(bt.start), parse_time(bt.end)
    split_ts = start + int((end - start) * bt.split_selection_pct / 100.0)
    slip, fee, fund = bt.slippage_pct / 100.0, cfg.trading.taker_fee_pct, bt.funding_fixed_pct
    client = MexcClient(cfg.mexc)
    symbols = client.all_usdt_contracts()
    names = None
    res, kept, total = {}, {}, 0
    used = 0
    for n_, sym in enumerate(symbols, 1):
        if n_ % 100 == 0:
            print(f"  {n_}/{len(symbols)}", flush=True)
        f = feats_for(cfg, client, sym, start, end)
        if f is None:
            continue
        used += 1
        k, d = stoch_rsi(f.close, sr.rsi_len, sr.stoch_len, sr.smooth_k, sr.smooth_d)
        ok = (f.ts >= start) & (f.ts <= end)
        p0, p1 = f.pmo15[:-1], f.pmo15[1:]
        up = np.where((p0 < 0) & (p1 >= 0))[0] + 1
        dn = np.where((p0 > 0) & (p1 <= 0))[0] + 1
        idx = np.concatenate([up, dn])
        s0 = np.concatenate([np.ones(len(up), int), -np.ones(len(dn), int)])
        valid = ok[idx] & ~np.isnan(f.pmo1h[idx]) & ~np.isnan(f.pmo4h[idx])
        idx, s0 = idx[valid], s0[valid]
        order = np.argsort(idx)
        idx, s0 = idx[order], s0[order]
        total += len(idx)
        conds = conditions(k[idx], d[idx], s0, sr.oversold, sr.overbought)
        names = list(conds)
        for cname, mask in conds.items():
            kept[cname] = kept.get(cname, 0) + int(mask.sum())
            for dname, sgn in (("original", 1), ("reverse", -1)):
                for hname, H in HORIZONS.items():
                    ts, x = simulate(f, idx[mask], (sgn * s0[mask]), H, None, slip, fee, fund)
                    res.setdefault((cname, dname, hname), ([], []))
                    res[(cname, dname, hname)][0].append(ts)
                    res[(cname, dname, hname)][1].append(x)

    L = ["# هل يغيّر Stochastic RSI النتيجة لحظة عبور PMO للصفر؟", ""]
    L.append(f"- العقود: {used} · الفترة: {bt.start} → {bt.end} · إعدادات StochRSI: RSI {sr.rsi_len}، ستوك {sr.stoch_len}، K {sr.smooth_k}، D {sr.smooth_d}، تشبع {sr.oversold}/{sr.overbought}")
    L.append(f"- التكلفة: رسوم {fee}% لكل جهة + انزلاق {bt.slippage_pct}% لكل جهة + تمويل ثابت {fund}% لكل 8 ساعات؛ خروج بمدة ثابتة بدون وقف")
    L.append(f"- إجمالي عبورات PMO_15 المرشحة: {total}\n")
    L.append("## نسبة العبورات التي يُبقيها كل شرط\n")
    L.append("| الشرط | العبورات | النسبة |\n|---|---|---|")
    for c in names:
        L.append(f"| {c} | {kept[c]} | {kept[c] / total * 100:.1f}% |")
    L.append("\n## النتائج (متوسط العائد الصافي % بعد كل التكاليف)\n")
    L.append("| الشرط | الاتجاه | الخروج | الصفقات | متوسط الصافي % | ربح % | t | اختيار: صفقات | اختيار % | تحقق: صفقات | تحقق % |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    n_pos_val, n_all = 0, 0
    best = None
    for c in names:
        for dname in ("original", "reverse"):
            for hname in HORIZONS:
                tss, xs = res[(c, dname, hname)]
                ts, x = np.concatenate(tss), np.concatenate(xs)
                if len(x) == 0:
                    continue
                al, se, va = summarize(ts, x, split_ts)
                L.append(f"| {c} | {'عكسي' if dname == 'reverse' else 'أصلي'} | {hname} | {al['n']} | {al['mean']:.4f} | {al['win']:.1f}% | {al['t']:.1f} | "
                         f"{se['n']} | {se['mean']:.4f} | {va['n']} | {va['mean']:.4f} |")
                n_all += 1
                n_pos_val += va["mean"] > 0
                if best is None or se["mean"] > best[0]:
                    best = (se["mean"], c, dname, hname, va["mean"], va["n"])
    L.append(f"\n- عدد النسخ المختبرة: {n_all}؛ منها موجبة في قسم التحقق: **{n_pos_val}**")
    if best:
        L.append(f"- أفضل نسخة حسب قسم الاختيار فقط: «{best[1]}» / {'عكسي' if best[2] == 'reverse' else 'أصلي'} / {best[3]} → اختيار {best[0]:.4f}% ثم تحقق **{best[4]:.4f}%** ({best[5]} صفقة)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stochrsi.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
