"""Order-block entry filter on hunter alerts (rules fixed in docs/OB_ENTRY.md). Long only.
Usage: python -m tradebot.strategies.mtf_entry.ob_entry [days] [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from tradebot.strategies.multi_agent.smc import order_block, structure

from .boom_watch import day_boot
from .channel_logic import atr14
from .early_signs import WARM, features
from .hunter_tiers import alerts, tier_of
from .immediate_move import alert_features, clean_label

SWING_N, MAX_AGE_H, ATR_NEAR = 3, 48, 0.5


def hourly(t, o, h, l, c):
    """Complete 1H candles from 15m arrays -> (t_end, o, h, l, c)."""
    hid = (t // 3600).astype(np.int64)
    ids, first, cnt = np.unique(hid, return_index=True, return_counts=True)
    rows = [(ids[k] * 3600 + 3600, o[first[k]], h[first[k]:first[k] + 4].max(), l[first[k]:first[k] + 4].min(), c[first[k] + 3])
            for k in range(len(ids)) if cnt[k] == 4 and first[k] + 3 < len(c)]
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]


def ob_state(i, t, h, l, c, th, oh, hh, lh, ch, events, atr_h):
    """(near, first_tap) for the 15m bar i using only data known at its close."""
    now = t[i] + 900
    cand = [e for e in events if e["dir"] == 1 and th[e["i"]] <= now and th[e["i"]] >= now - MAX_AGE_H * 3600]
    if not cand:
        return False, False
    e = cand[-1]
    ob = order_block(e, oh, hh, lh, ch)
    if ob is None:
        return False, False
    lo, hi = ob[0], ob[1]
    after = np.where((t >= th[e["i"]]) & (t <= t[i]))[0]
    if len(after) == 0 or (c[after] < lo).any():                      # broken (closed below the zone)
        return False, False
    ih = int(np.searchsorted(th, now, side="right")) - 1
    atr = atr_h[ih] if 0 <= ih < len(atr_h) else np.nan
    if np.isnan(atr) or not (lo <= c[i] <= hi + ATR_NEAR * atr):
        return False, False
    prior = after[after < i]
    first = bool(l[i] <= hi and not (l[prior] <= hi).any())
    return True, first


def main():
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import MexcClient

    from .paper import NON_CRYPTO

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    out = sys.argv[2] if len(sys.argv) > 2 else None
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 300_000]
    rows, nsym, live = [], 0, []
    for tk in tick:
        sym = tk["symbol"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - days * 96 * 900, now), "Min15", now)
            o, h, l, c, v, amt, t = (np.asarray(k[x], float) for x in ("open", "high", "low", "close", "vol", "amount", "time"))
            n = len(c)
            if n < WARM + 200:
                continue
            f, _, valid = features(o, h, l, c, v, t, np.array([0.0]), np.array([0.0]), np.zeros(n))
            tier = tier_of({x: f[x] for x in ("F1", "F2", "F3")}); tier[:WARM] = -1; tier[~valid] = -1
            a15 = atr14(h, l, c)
            th, oh, hh, lh, ch = hourly(t, o, h, l, c)
            _, events = structure(ch, hh, lh, SWING_N)
            atr_h = atr14(hh, lh, ch)
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
            continue
        nsym += 1
        for i, kk in alerts(tier):
            if kk > 2 or i < 100 or np.isnan(a15[i]):
                continue
            near, first = ob_state(i, t, h, l, c, th, oh, hh, lh, ch, events, atr_h)
            lab = clean_label(h, l, c, i)
            ft = alert_features(c, amt, a15, i)
            if lab is not None:
                rows.append((kk, t[i], int(t[i] // 86400), lab[0], lab[1], lab[2], near, first, ft["L"]))
            elif i >= n - 8 and near:
                live.append((sym, kk, (n - 1 - i) * 15, c[-1], first, ft["turn"]))
    cut = max(r[1] for r in rows) - 30 * 86400
    hold_days = max((max(r[1] for r in rows) - cut) / 86400, 1)
    base_h = np.mean([r[3] for r in rows if r[1] >= cut])
    lines = [f"شرط الأوردر بلوك على تنبيهات T1-T3 · {nsym} عقداً · {len(rows)} تنبيهاً · فاصل التحقق {time.strftime('%Y-%m-%d', time.gmtime(cut))} · القواعد: docs/OB_ENTRY.md", "",
             f"الأساس في التحقق (كل التنبيهات): {base_h * 100:.1f}%",
             f"{'المجموعة':<10}{'تنبيهات(تصميم/تحقق)':>22}{'/يوم':>7}{'فائز نظيف':>12}{'حد 95%':>18}{'مهلة +5% وسيط':>16}  الحكم"]
    tests = (("OB", lambda r: r[6]), ("OB-first", lambda r: r[7]), ("T1+OB", lambda r: r[0] == 0 and r[6]), ("L+OB", lambda r: r[8] and r[6]))
    for g, test in tests:
        d = [r for r in rows if r[1] < cut and test(r)]
        h_ = [r for r in rows if r[1] >= cut and test(r)]
        if not h_:
            lines.append(f"{g:<10}{len(d):>10}/0"); continue
        win = np.array([r[3] for r in h_], float)
        lo, hi = day_boot([np.array([r[3] for r in h_ if r[2] == dd], float) for dd in sorted({r[2] for r in h_})])
        w = [r for r in h_ if r[3]]
        lead = np.median([r[4] * 15 for r in w]) if w else float("nan")
        ok = len(h_) >= 50 and win.mean() >= 1.5 * base_h and lo > base_h and lo > 0.25
        lines.append(f"{g:<10}{len(d):>10}/{len(h_):<11}{len(h_) / hold_days:>7.1f}{win.mean() * 100:>11.1f}%   [{lo * 100:.1f}%, {hi * 100:.1f}%]{lead:>13.0f}د  {'نجحت' if ok else 'رُفضت'}")
    lines += ["", "تنبيهات حالية (آخر ساعتين) والسعر قرب أوردر بلوك:"]
    for sym, kk, age, px, first, tn in sorted(live, key=lambda x: (x[1], x[2])):
        lines.append(f"  T{kk + 1} {sym:<14} منذ {age}د · سعر {px:.6g} · {'أول لمسة' if first else 'قرب المنطقة'} · سيولة 24س {tn / 1e6:.1f}M")
    if not live:
        lines.append("  لا شيء")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
