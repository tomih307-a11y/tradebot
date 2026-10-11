"""Independent audit of the PMO zero-cross backtest (does not use the engine for part 1).

Part 1  SIGNAL ONLY: enter at the close of every PMO_15 zero-cross (long on up-cross, short on down-cross),
        exit after a fixed number of bars. No stops, no stages. Direction-adjusted forward return, before costs.
Part 2  ACCOUNTING: run the engine on a sample of symbols and recompute every fill / return from raw candles.
Usage:  python tools/audit.py --out reports/audit [--sample 40]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.config import load_config, parse_time                      # noqa: E402
from tradebot.data.klines import get_klines                              # noqa: E402
from tradebot.mexc.client import TF_SECONDS, MexcClient                  # noqa: E402
from tradebot.strategies.pmo_zero_cross import model as M                # noqa: E402
from tradebot.strategies.pmo_zero_cross.backtest import run_backtest     # noqa: E402
from tradebot.strategies.pmo_zero_cross.features import build_features   # noqa: E402

HORIZONS = {"15m": 1, "1h": 4, "4h": 16, "24h": 96}


def feats_for(cfg, client, sym, start, end):
    s = cfg.pmo_zero_cross
    pad = s.indicator.warmup_bars + 50
    k = {}
    for key, tf in (("15", s.tf_entry), ("1h", s.tf_1h), ("4h", s.tf_4h)):
        k[key] = get_klines(client, s.backtest.data_dir, sym, tf, start - pad * TF_SECONDS[tf], end)
    if min(len(k[x]["time"]) for x in k) == 0:
        return None
    return build_features(sym, k["15"], k["1h"], k["4h"], s, liquidity_window=s.backtest.liquidity_window_bars)


class Acc:
    def __init__(self):
        self.v = []

    def add(self, arr):
        self.v.append(arr)

    def stats(self):
        if not self.v:
            return None
        x = np.concatenate(self.v)
        n = len(x)
        sd = x.std(ddof=1) if n > 1 else float("nan")
        return {"n": n, "mean": x.mean(), "median": float(np.median(x)), "pos": float((x > 0).mean()),
                "t": x.mean() / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/audit")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--config", default="config.toml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    bt = cfg.pmo_zero_cross.backtest
    start, end = parse_time(bt.start), parse_time(bt.end)
    client = MexcClient(cfg.mexc)
    symbols = client.all_usdt_contracts()
    print(f"{len(symbols)} symbols", flush=True)

    sig = {h: {"dir": Acc(), "long": Acc(), "short": Acc(), "drift": Acc(), "rev": Acc()} for h in HORIZONS}
    by_slopes = {h: {} for h in HORIZONS}
    sample_feats, used = {}, 0
    for n, sym in enumerate(symbols, 1):
        if n % 100 == 0:
            print(f"  {n}/{len(symbols)}", flush=True)
        f = feats_for(cfg, client, sym, start, end)
        if f is None:
            continue
        used += 1
        ok = (f.ts >= start) & (f.ts <= end)
        p0, p1 = f.pmo15[:-1], f.pmo15[1:]
        up = np.where((p0 < 0) & (p1 >= 0))[0] + 1
        dn = np.where((p0 > 0) & (p1 <= 0))[0] + 1
        up, dn = up[ok[up]], dn[ok[dn]]
        c = f.close
        for hname, h in HORIZONS.items():
            def fwd(idx):
                idx = idx[(idx + h < len(c)) & ~np.isnan(f.pmo1h[idx]) & ~np.isnan(f.pmo4h[idx])]
                return idx, (c[idx + h] / c[idx] - 1.0) * 100.0
            iu, ru = fwd(up)
            idn, rd = fwd(dn)
            sig[hname]["long"].add(ru)
            sig[hname]["short"].add(-rd)                   # directional (short wins when price falls)
            sig[hname]["dir"].add(np.concatenate([ru, -rd]))
            sig[hname]["rev"].add(np.concatenate([-ru, rd]))   # the opposite trade
            # market drift control: every bar, long
            allidx = np.arange(0, len(c) - h)
            allidx = allidx[ok[allidx]]
            sig[hname]["drift"].add((c[allidx + h] / c[allidx] - 1.0) * 100.0)
            # by slope class at entry (1H slope, 4H slope vs trade direction)
            for idx, s_, r in ((iu, 1, ru), (idn, -1, -rd)):
                w1 = (s_ * f.slope1h[idx] >= 0)
                w4 = (s_ * f.slope4h[idx] >= 0)
                for key, mask in (("with|with", w1 & w4), ("with|against", w1 & ~w4),
                                  ("against|with", ~w1 & w4), ("against|against", ~w1 & ~w4)):
                    by_slopes[hname].setdefault(key, Acc()).add(r[mask])
        if len(sample_feats) < a.sample and (len(up) + len(dn)) > 0 and used % max(1, len(symbols) // (a.sample * 3)) == 0:
            sample_feats[sym] = f

    L = ["# تدقيق مستقل لاختبار PMO", ""]
    L.append(f"- العقود المستخدمة: {used} · الفترة: {bt.start} → {bt.end}")
    cost = 2 * cfg.trading.taker_fee_pct + 2 * bt.slippage_pct
    L.append(f"- التكلفة ذهاباً وإياباً (رسوم + انزلاق): {cost:.2f}%\n")
    L.append("## 1) الإشارة وحدها: العائد بعد العبور (قبل التكاليف، بدون وقف)\n")
    L.append("| الأفق | الإشارات | متوسط العائد % (اتجاه الصفقة) | وسيط | نسبة الصعود | t | لونق فقط % | شورت فقط % | الصفقة العكسية % | انجراف السوق (شراء كل شمعة) % |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for h in HORIZONS:
        d, lo, sh, rv, dr = (sig[h][k].stats() for k in ("dir", "long", "short", "rev", "drift"))
        L.append(f"| {h} | {d['n']} | {d['mean']:.4f} | {d['median']:.4f} | {d['pos'] * 100:.1f}% | {d['t']:.1f} | "
                 f"{lo['mean']:.4f} | {sh['mean']:.4f} | {rv['mean']:.4f} | {dr['mean']:.4f} |")
    L.append("\n## 1ب) نفس الشيء حسب ميل الساعة × ميل الـ4س لحظة العبور (أفق 4 ساعات و24 ساعة)\n")
    L.append("| الأفق | الحالة | الإشارات | متوسط % قبل التكاليف | t |\n|---|---|---|---|---|")
    for h in ("4h", "24h"):
        for k in ("with|with", "with|against", "against|with", "against|against"):
            st = by_slopes[h].get(k)
            st = st.stats() if st else None
            if st:
                L.append(f"| {h} | {k.replace('|', ' / ')} | {st['n']} | {st['mean']:.4f} | {st['t']:.1f} |")

    # ---------- part 2: accounting ----------
    L.append("\n## 2) تدقيق حسابات المحرك على عينة من العقود (أسعار الدخول/الخروج والعائد من الشموع الخام)\n")
    feats = dict(sample_feats)
    btc = feats_for(cfg, client, cfg.trading.btc_symbol, start, end)
    from tradebot.strategies.pmo_zero_cross.report import _override
    c2 = _override(cfg, **{"trading.max_open_trades": 100000})
    eng = run_backtest(c2, feats, btc, start, end)
    slip = bt.slippage_pct / 100.0
    fee2 = 2 * cfg.trading.taker_fee_pct
    bad = {"entry": 0, "exit_fill": 0, "gross": 0, "net": 0, "stop_not_touched": 0}
    n_checked = 0
    for t in eng.closed:
        if t.exit_reason == M.END_OF_DATA:
            continue
        f = feats[t.symbol]
        i_in, i_out = f.index_of(t.entry_ts), f.index_of(t.exit_ts)
        s = t.s
        n_checked += 1
        if not math.isclose(t.entry_price, f.close[i_in] * (1 + s * slip), rel_tol=1e-9):
            bad["entry"] += 1
        if t.exit_reason.startswith("STOP"):
            o = f.open[i_out]
            gap = (o <= t.stop) if s == 1 else (o >= t.stop)
            exp = o if gap else t.stop
            touched = (f.low[i_out] <= t.stop) if s == 1 else (f.high[i_out] >= t.stop)
            if not (gap or touched):
                bad["stop_not_touched"] += 1
        else:
            exp = f.close[i_out] * (1 - s * slip)
        if not math.isclose(t.exit_price, exp, rel_tol=1e-9):
            bad["exit_fill"] += 1
        g = s * (t.exit_price - t.entry_price) / t.entry_price * 100
        if not math.isclose(g, t.gross_ret_pct, rel_tol=1e-9, abs_tol=1e-9):
            bad["gross"] += 1
        if not math.isclose(t.net_ret_pct, g - fee2 - t.funding_pct, rel_tol=1e-9, abs_tol=1e-9):
            bad["net"] += 1
    gross = [t.gross_ret_pct for t in eng.closed if t.exit_reason != M.END_OF_DATA]
    net = [t.net_ret_pct for t in eng.closed if t.exit_reason != M.END_OF_DATA]
    L.append(f"- عقود العينة: {len(feats)} · صفقات مفحوصة: {n_checked}")
    L.append(f"- أخطاء سعر الدخول: {bad['entry']} · أخطاء سعر الخروج: {bad['exit_fill']} · "
             f"وقف بدون لمس: {bad['stop_not_touched']} · أخطاء العائد الإجمالي: {bad['gross']} · أخطاء الصافي: {bad['net']}")
    L.append(f"- متوسط العائد الإجمالي (قبل الرسوم والتمويل، بعد الانزلاق): {np.mean(gross):.4f}% · الصافي: {np.mean(net):.4f}%")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
