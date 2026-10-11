"""Reverse-signal study: enter OPPOSITE to every PMO_15 zero-cross, exit after a fixed horizon (optionally with a
protective stop). Original direction is evaluated the same way for an apples-to-apples comparison.
Costs: taker fee both sides + slippage both sides (applied to fills) + fixed funding. One open trade per symbol
(a new cross is ignored while a trade is open and until the exit bar has passed). Unlimited concurrent symbols.
Usage: python tools/reverse_study.py --out reports/reverse
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit import feats_for                                  # noqa: E402
from tradebot.config import load_config, parse_time          # noqa: E402
from tradebot.mexc.client import MexcClient                  # noqa: E402

HORIZONS = {"1h": 4, "4h": 16, "24h": 96}
STOPS = {"بدون وقف": None, "وقف 3%": 3.0, "وقف 1.5%": 1.5}


def simulate(f, idxs, sides, H, stop_pct, slip, fee_pct, fund_pct_per_8h):
    """Sequential per-symbol simulation. Returns arrays (entry_ts, net_pct)."""
    c, o, hi, lo = f.close, f.open, f.high, f.low
    n = len(c)
    out_ts, out_net = [], []
    free_from = -1
    for i, s in zip(idxs, sides):
        if i <= free_from or i + 1 >= n:
            continue
        e = c[i] * (1 + s * slip)
        stop = None if stop_pct is None else e * (1 - s * stop_pct / 100.0)
        j_end = min(i + H, n - 1)
        exit_px, j_exit = None, j_end
        if stop is not None:
            w_lo, w_hi, w_op = lo[i + 1:j_end + 1], hi[i + 1:j_end + 1], o[i + 1:j_end + 1]
            hit = np.where(w_lo <= stop)[0] if s == 1 else np.where(w_hi >= stop)[0]
            if len(hit):
                k = hit[0]
                j_exit = i + 1 + k
                gap = (w_op[k] <= stop) if s == 1 else (w_op[k] >= stop)
                exit_px = w_op[k] if gap else stop
        if exit_px is None:
            exit_px = c[j_end] * (1 - s * slip)
        gross = s * (exit_px - e) / e * 100.0
        hold_h = (j_exit - i) * 0.25
        funding = s * fund_pct_per_8h * (hold_h / 8.0)          # long pays positive funding, short receives
        net = gross - 2 * fee_pct - funding
        out_ts.append(f.ts[i])
        out_net.append(net)
        free_from = j_exit
    return np.array(out_ts), np.array(out_net)


def summarize(ts, x, split_ts):
    def st(v):
        n = len(v)
        if n < 2:
            return {"n": n, "mean": float("nan"), "win": float("nan"), "t": float("nan")}
        sd = v.std(ddof=1)
        return {"n": n, "mean": v.mean(), "win": (v > 0).mean() * 100, "t": v.mean() / (sd / math.sqrt(n)) if sd > 0 else float("nan")}
    return st(x), st(x[ts < split_ts]), st(x[ts >= split_ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/reverse")
    ap.add_argument("--config", default="config.toml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    bt = cfg.pmo_zero_cross.backtest
    start, end = parse_time(bt.start), parse_time(bt.end)
    split_ts = start + int((end - start) * bt.split_selection_pct / 100.0)
    slip, fee, fund = bt.slippage_pct / 100.0, cfg.trading.taker_fee_pct, bt.funding_fixed_pct
    client = MexcClient(cfg.mexc)
    symbols = client.all_usdt_contracts()
    res = {(d, h, sname): ([], []) for d in ("original", "reverse") for h in HORIZONS for sname in STOPS}
    used = 0
    for n_, sym in enumerate(symbols, 1):
        if n_ % 100 == 0:
            print(f"  {n_}/{len(symbols)}", flush=True)
        f = feats_for(cfg, client, sym, start, end)
        if f is None:
            continue
        used += 1
        ok = (f.ts >= start) & (f.ts <= end)
        p0, p1 = f.pmo15[:-1], f.pmo15[1:]
        up = np.where((p0 < 0) & (p1 >= 0))[0] + 1
        dn = np.where((p0 > 0) & (p1 <= 0))[0] + 1
        idx = np.concatenate([up, dn])
        side_orig = np.concatenate([np.ones(len(up), int), -np.ones(len(dn), int)])
        valid = ok[idx] & ~np.isnan(f.pmo1h[idx]) & ~np.isnan(f.pmo4h[idx])
        idx, side_orig = idx[valid], side_orig[valid]
        order = np.argsort(idx)
        idx, side_orig = idx[order], side_orig[order]
        for dname, sides in (("original", side_orig), ("reverse", -side_orig)):
            for hname, H in HORIZONS.items():
                for sname, sp in STOPS.items():
                    ts, x = simulate(f, idx, sides, H, sp, slip, fee, fund)
                    res[(dname, hname, sname)][0].append(ts)
                    res[(dname, hname, sname)][1].append(x)

    L = ["# دراسة عكس الإشارة", ""]
    L.append(f"- العقود: {used} · الفترة: {bt.start} → {bt.end} · حد التقسيم: {split_ts}")
    L.append(f"- التكلفة: رسوم {fee}% لكل جهة + انزلاق {bt.slippage_pct}% لكل جهة + تمويل ثابت {fund}% لكل 8 ساعات")
    L.append("- صفقة واحدة مفتوحة لكل عملة؛ العائد الصافي % من القيمة الاسمية؛ بقسمي الاختيار والتحقق\n")
    L.append("| الاتجاه | الخروج | الوقف | الصفقات | متوسط الصافي % | ربح % | t | اختيار: صفقات | اختيار: متوسط % | تحقق: صفقات | تحقق: متوسط % |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    best = None
    for (d, h, sname), (tss, xs) in res.items():
        ts = np.concatenate(tss) if tss else np.array([])
        x = np.concatenate(xs) if xs else np.array([])
        if len(x) == 0:
            continue
        al, se, va = summarize(ts, x, split_ts)
        L.append(f"| {'عكسي' if d == 'reverse' else 'أصلي'} | {h} | {sname} | {al['n']} | {al['mean']:.4f} | {al['win']:.1f}% | "
                 f"{al['t']:.1f} | {se['n']} | {se['mean']:.4f} | {va['n']} | {va['mean']:.4f} |")
        if d == "reverse" and (best is None or se["mean"] > best[3]):      # pick on the SELECTION split only
            best = (va["mean"], h, sname, se["mean"], va["n"])
    if best:
        L.append(f"\nأفضل نسخة عكسية حسب قسم الاختيار فقط: خروج {best[1]} / {best[2]} → اختيار {best[3]:.4f}% ثم تحقق {best[0]:.4f}% ({best[4]} صفقة)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reverse.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
