"""Exploratory check of pre-explosion signs that HAVE history on MEXC (rules fixed in docs/EARLY_SIGNS.md). LONG explosions only, tiers 10/20/30% within 1h.
Usage: python -m tradebot.strategies.mtf_entry.early_signs [days] [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from . import model as M
from .boom_watch import day_boot

TIERS = (10.0, 20.0, 30.0)
AHEAD, V96, V7D, BW_WIN, WARM = 4, 96, 672, 200, 700
FLAGS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
NAMES = {"F1": "تكرار 48س", "F2": "تراكم حجم", "F3": "قفزة حجم", "F4": "ضغط", "F5": "قرب القاع", "F6": "تمويل سالب", "F7": "فرق سالب", "F8": "هدوء 24س",
         "C1": "تكرار+قفزة حجم", "C2": "قرب القاع+قفزة حجم", "C3": "ضغط+قرب القاع+تمويل سالب", "C4": "قفزة حجم+تراكم"}
COMBOS = {"C1": ("F1", "F3"), "C2": ("F5", "F3"), "C3": ("F4", "F5", "F6"), "C4": ("F3", "F2")}


def _pad(x: np.ndarray, n: int, lead: int) -> np.ndarray:
    out = np.full(n, np.nan)
    out[lead:lead + len(x)] = x
    return out


def up_events(o, c, merge=48):
    ret = (c / o - 1) * 100
    ev, last = [], -10**9
    for j in np.where(ret >= 10.0)[0]:
        if j - last >= merge:
            ev.append(int(j))
        last = j
    return np.array(ev, int)


def features(o, h, l, c, v, t, fund_t, fund_r, basis):
    """Per-bar flags/labels (arrays of length n; NaN where undefined). All use data up to and including bar i only."""
    n = len(c)
    f = {}
    ev = up_events(o, c)
    cnt = np.zeros(n)
    for j in ev:
        cnt[j:j + 192] += 1
    f["F1"] = cnt >= 1
    med96 = np.full(n, np.nan)
    med96[V96:] = np.median(swv(v, V96)[:-1], axis=1)                  # median of the 96 bars BEFORE bar i
    f["F3"] = v / med96 >= 4.0
    s4 = np.full(n, np.nan)
    s4[3:] = np.convolve(v, np.ones(4), "valid")
    med7 = np.full(n, np.nan)
    med7[V7D + 3:] = np.median(swv(s4[3:], V7D)[:-1], axis=1)
    f["F2"] = s4 / med7 >= 3.0
    mid = M._sma(c, M.SQ_BB_LEN)
    sd = np.full(n, np.nan)
    sd[M.SQ_BB_LEN - 1:] = swv(c, M.SQ_BB_LEN).std(axis=1)
    bw = 4 * sd / mid
    rank = np.full(n, np.nan)
    w = swv(bw, BW_WIN + 1)                                            # last element = current bar
    ok = ~np.isnan(w).any(axis=1)
    r = np.where(ok, (w[:, :-1] < w[:, -1:]).mean(axis=1), np.nan)
    rank[BW_WIN:] = r
    f["F4"] = rank <= 0.20
    lo96 = np.full(n, np.nan); hi96 = np.full(n, np.nan)
    lo96[V96 - 1:] = swv(l, V96).min(axis=1); hi96[V96 - 1:] = swv(h, V96).max(axis=1)
    f["F5"] = (c - lo96) / np.where(hi96 > lo96, hi96 - lo96, np.nan) <= 0.15
    idx = np.searchsorted(fund_t, t, side="right") - 1
    fr = np.where(idx >= 0, fund_r[np.clip(idx, 0, len(fund_r) - 1)] if len(fund_r) else np.nan, np.nan)
    f["F6"] = fr <= -0.00005
    f["F7"] = basis <= -0.001
    ret24 = np.full(n, np.nan)
    ret24[V96:] = c[V96:] / c[:-V96] - 1
    f["F8"] = np.abs(ret24) <= 0.03
    valid = ~np.isnan(med96) & ~np.isnan(med7) & ~np.isnan(rank) & ~np.isnan(ret24) & ~np.isnan(fr) & ~np.isnan(basis)
    fmax = np.full(n, np.nan)
    fmax[:n - AHEAD] = swv(h[1:], AHEAD).max(axis=1)[: n - AHEAD] / c[: n - AHEAD] - 1
    for k, parts in COMBOS.items():
        f[k] = np.logical_and.reduce([f[p] for p in parts])
    f = {k: np.nan_to_num(a.astype(float)).astype(bool) if a.dtype != bool else a for k, a in f.items()}
    return f, fmax * 100, valid


def fetch_basis(client, sym, t, now):
    """(fair-index)/index aligned to the 15m open times t (NaN if missing)."""
    def series(kind):
        out, cur, step, chunk = {}, int(t[0]), 900, 1999 * 900
        while cur <= now:
            d = client._request("GET", f"/api/v1/contract/kline/{kind}/{sym}", {"interval": "Min15", "start": cur, "end": min(cur + chunk, now)}) or {}
            for tt, cc in zip(d.get("time", []), d.get("close", [])):
                out[int(tt)] = float(cc)
            cur += chunk + step
        return out
    fair, idx = series("fair_price"), series("index_price")
    return np.array([(fair[x] - idx[x]) / idx[x] if x in fair and x in idx and idx[x] else np.nan for x in t.astype(int)])


def study(days=90, out=None):
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import MexcClient

    from .paper import NON_CRYPTO

    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 300_000]
    cols = {k: [] for k in FLAGS + list(COMBOS)}
    lab, daycol, tcol, nsym = [], [], [], 0
    for tk in tick:
        sym = tk["symbol"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - days * 96 * 900, now), "Min15", now)
            o, h, l, c, v, t = (np.asarray(k[x], float) for x in ("open", "high", "low", "close", "vol", "time"))
            if len(c) < WARM + 200:
                continue
            fund = cl.funding_history(sym, max_pages=40)
            ft, fr = np.array([x[0] for x in fund], float), np.array([x[1] for x in fund], float)
            if len(ft) < 5:
                continue
            f, fmax, valid = features(o, h, l, c, v, t, ft, fr, fetch_basis(cl, sym, t, now))
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
            continue
        sel = valid & ~np.isnan(fmax)
        sel[:WARM] = False
        for key in cols:
            cols[key].append(f[key][sel])
        lab.append(fmax[sel]); daycol.append((t[sel] // 86400).astype(int)); tcol.append(t[sel]); nsym += 1
    lab, day, tt = np.concatenate(lab), np.concatenate(daycol), np.concatenate(tcol)
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    cut = tt.max() - 30 * 86400
    dev, hold = tt < cut, tt >= cut
    lines = [f"علامات ما قبل الانفجار الصاعد · {nsym} عقداً · {len(lab)} نقطة قرار · تصميم {int(dev.sum())} / تحقق {int(hold.sum())} · القواعد: docs/EARLY_SIGNS.md", ""]
    base = {x: float((lab[hold] >= x).mean()) for x in TIERS}
    lines.append("النسبة الأساسية (تحقق): " + " · ".join(f"≥{int(x)}%: {base[x] * 100:.2f}%" for x in TIERS) + "\n")
    lines.append(f"{'الرمز':<4}{'العلامة':<26}{'معلَّمة(تصميم)':>14}{'معلَّمة(تحقق)':>14}{'≥10%':>8}{'رفع':>6}{'≥20%':>8}{'رفع':>6}{'≥30%':>8}{'رفع':>6}{'استدعاء10':>10}  حد ثقة ≥10%")
    for key in FLAGS + list(COMBOS):
        m = cols[key]
        nd, nh = int((m & dev).sum()), int((m & hold).sum())
        if nh == 0:
            lines.append(f"{key:<4}{NAMES[key]:<26}{nd:>14}{nh:>14}   لا نقاط")
            continue
        sel = m & hold
        hits = [float((lab[sel] >= x).mean()) for x in TIERS]
        lifts = [h_ / base[x] if base[x] else float("nan") for h_, x in zip(hits, TIERS)]
        pos10 = hold & (lab >= 10)
        recall = float((m & pos10).sum() / max(pos10.sum(), 1))
        byday = [(lab[sel & (day == d)] >= 10).astype(float) for d in np.unique(day[sel])]
        lo, hi = day_boot(byday)
        lines.append(f"{key:<4}{NAMES[key]:<26}{nd:>14}{nh:>14}{hits[0] * 100:>7.1f}%{lifts[0]:>6.1f}{hits[1] * 100:>7.1f}%{lifts[1]:>6.1f}{hits[2] * 100:>7.1f}%{lifts[2]:>6.1f}{recall * 100:>9.0f}%  [{lo * 100:.1f}%, {hi * 100:.1f}%]")
    worth = [k for k in FLAGS + list(COMBOS) if (cols[k] & hold).sum() >= 100 and base[10.0] and (lab[cols[k] & hold] >= 10).mean() / base[10.0] >= 2
             and day_boot([(lab[cols[k] & hold & (day == d)] >= 10).astype(float) for d in np.unique(day[cols[k] & hold])])[0] > base[10.0]]
    lines += ["", "تستحق المتابعة (رفع ≥ 2 على ≥ 100 نقطة وحد ثقة أدنى فوق الأساس، لدرجة 10%): " + (("، ".join(f"{k} ({NAMES[k]})" for k in worth)) or "لا شيء")]
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    study(int(sys.argv[1]) if len(sys.argv) > 1 else 90, sys.argv[2] if len(sys.argv) > 2 else None)
