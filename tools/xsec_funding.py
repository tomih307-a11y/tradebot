"""DESCRIPTIVE check (not part of the pre-registered success criterion): how much does perpetual FUNDING eat of the cross-sectional momentum edge?

For every start day t (H=7 days, overlapping tranches) the long and short legs are chosen exactly as in tools/xsec.py. Funding paid by a leg =
sum of the contract's funding rates settled in (entry, exit]. Longs PAY the rate, shorts RECEIVE it, so per unit of gross exposure
funding PnL = (mean(short legs' funding) - mean(long legs' funding)) / 2.
Reported for two windows (before 2024-03-21 and from 2024-03-21), for mom30, mom7 and the random control, next to the price-only net return.
Usage: python tools/xsec_funding.py --top 300 --days 3000 --out reports/xsec_funding
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

from xsec import DAY, build_matrices, features, hac_t, load_series, period_returns   # noqa: E402
from tradebot.config import load_config                                              # noqa: E402
from tradebot.data.klines import get_funding                                         # noqa: E402
from tradebot.mexc.client import MexcClient                                          # noqa: E402


def legs(score, turn30, t, top_m, min_names=25, q=0.2):
    """Same selection as xsec.period_returns (needs the price matrix only for the finite filter at exit)."""
    sc, tv = score[t], turn30[t]
    elig = np.where(np.isfinite(sc) & np.isfinite(tv))[0]
    if len(elig) < min_names:
        return None
    elig = elig[np.argsort(-tv[elig])][:top_m]
    return elig, sc[elig]


def funding_sums(dates_row_ts, rates_by_col, cols, t0, t1):
    """Sum of funding rates settled in (t0, t1] for each column; NaN when the contract has no funding history covering the window."""
    out = np.full(len(cols), np.nan)
    for j, c in enumerate(cols):
        arr = rates_by_col.get(c)
        if arr is None or len(arr[0]) == 0 or arr[0][0] > t0 + DAY or arr[0][-1] < t1 - DAY:
            continue
        ts, r = arr
        m = (ts > t0) & (ts <= t1)
        out[j] = r[m].sum()
    return out


def funding_drag(score, turn30, O, dates, rates_by_col, H, top_m, lo_i, hi_i, q=0.2, min_names=25):
    """-> (mean funding PnL per tranche in %, number of tranches, mean price-only gross in %) over start rows in [lo_i, hi_i)."""
    f_list, g_list = [], []
    for t in range(lo_i, min(hi_i, len(dates) - H - 2)):
        sel = legs(score, turn30, t, top_m, min_names, q)
        if sel is None:
            continue
        elig, sc = sel
        r = O[t + 1 + H, elig] / O[t + 1, elig] - 1
        ok = np.isfinite(r)
        elig, r, sc = elig[ok], r[ok], sc[ok]
        if len(elig) < min_names:
            continue
        k = max(1, int(np.ceil(q * len(elig))))
        order = np.argsort(sc)
        shorts, longs = elig[order[:k]], elig[order[-k:]]
        fs = funding_sums(None, rates_by_col, list(shorts), dates[t + 1], dates[t + 1 + H])
        fl = funding_sums(None, rates_by_col, list(longs), dates[t + 1], dates[t + 1 + H])
        if np.isfinite(fs).sum() < 0.8 * k or np.isfinite(fl).sum() < 0.8 * k:
            continue
        f_list.append((np.nanmean(fs) - np.nanmean(fl)) / 2.0)
        g_list.append((r[order[-k:]].mean() - r[order[:k]].mean()) / 2.0)
    if not f_list:
        return None
    f = np.array(f_list)
    return {"n": len(f), "funding_pct": float(f.mean() * 100), "gross_pct": float(np.mean(g_list) * 100), "funding_t": hac_t(f, 14)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--top-m", type=int, default=150)
    ap.add_argument("--days", type=int, default=3000)
    ap.add_argument("--cut", default="2024-03-21")
    ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--out", default="reports/xsec_funding")
    a = ap.parse_args()
    cfg = load_config("config.toml")
    a.fee = cfg.trading.taker_fee_pct if a.fee is None else a.fee
    client = MexcClient(cfg.mexc)
    now = int(time.time())
    series = load_series(a, cfg, client, now)
    dates, O, C, A = build_matrices(series, now - a.days * DAY, now - DAY)
    f, turn30 = features(C, A)
    dd = cfg.pmo_zero_cross.backtest.data_dir
    rates = {}
    for i, sym in enumerate(series):
        rows = get_funding(client, dd, sym)
        if rows:
            ts = np.array([r[0] for r in rows], dtype=float)
            rv = np.array([r[1] for r in rows], dtype=float)
            rates[i] = (ts, rv)
        if (i + 1) % 50 == 0:
            print(f"  funding [{i + 1}/{len(series)}] with data: {len(rates)}", flush=True)
    cut_ts = int(datetime.strptime(a.cut, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    cut_i = int(np.searchsorted(dates, cut_ts))
    first = 40
    cost_leg = 2 * (a.fee + a.slip) / 100.0
    rng = np.random.default_rng(19)
    out = {"meta": {"symbols": len(series), "with_funding": len(rates), "cut": a.cut, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}, "rows": []}
    scores = {"mom30": f["mom30"], "mom7": f["mom7"], "random": rng.random(C.shape)}
    for name, sc in scores.items():
        for wname, (lo, hi) in (("قبل " + a.cut, (first, cut_i)), ("من " + a.cut, (cut_i, len(dates)))):
            d = funding_drag(sc, turn30, O, dates, rates, 7, a.top_m, lo, hi)
            _, r = period_returns(sc, turn30, O, dates, 7, a.top_m, cost_leg, lo, hi, step=1)
            net = float(r.mean() * 100) if len(r) else None
            out["rows"].append({"signal": name, "window": wname, "net_price_only_pct": net, "funding": d})
    L = [f"# أثر التمويل على الزخم المقطعي (وصفي) ({out['meta']['generated']})", "",
         f"- {out['meta']['symbols']} عقداً ({out['meta']['with_funding']} منها بتاريخ تمويل). اللونق يدفع التمويل والشورت يقبضه. لكل 7 أيام، لكل وحدة تعرّض إجمالي.",
         "- «صافي السعر» = العائد بعد رسوم/انزلاق 0.18% لكل ساق بلا تمويل (كما في الاختبار المسجّل). «التمويل» = أثر التمويل على نفس الفترات (سالب = يأكل الربح).", "",
         "| الإشارة | النافذة | فترات (مع تمويل) | صافي السعر % | أثر التمويل % | t (تمويل) | صافي بعد التمويل % |", "|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        d = r["funding"]
        if d and r["net_price_only_pct"] is not None:
            L.append(f"| {r['signal']} | {r['window']} | {d['n']} | {r['net_price_only_pct']:+.3f} | {d['funding_pct']:+.3f} | {d['funding_t']:+.1f} | {r['net_price_only_pct'] + d['funding_pct']:+.3f} |")
        else:
            L.append(f"| {r['signal']} | {r['window']} | – | – | – | – | – |")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "funding.md").write_text("\n".join(L), encoding="utf-8")
    (Path(a.out) / "funding.json").write_text(json.dumps(out, ensure_ascii=False, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
