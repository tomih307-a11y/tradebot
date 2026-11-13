"""Check of the squeeze-breakout alert on MEXC history (rules fixed in docs/BOOM_CHECK.md). Usage: python -m tradebot.strategies.mtf_entry.boom_study [days] [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from tradebot.config import load_config
from tradebot.data.klines import drop_unclosed, fetch_range
from tradebot.mexc.client import MexcClient

from . import model as M
from .paper import NON_CRYPTO

EXPLODE_PCT, MERGE_BARS, LOOK_BARS, WIN_PCT, WIN_BARS = 10.0, 48, 96, 10.0, 96
hhmm = lambda x: time.strftime("%m-%d %H:%M", time.gmtime(int(x)))


def analyse_symbol(o, h, l, c, v, t):
    pre = M.squeeze_precompute(o, h, l, c, v)
    sigs = []                                           # (i, dir, info)
    for i in range(215, len(c)):
        s = M.squeeze_at(pre, o, h, l, c, v, i)
        if s:
            sigs.append((i, s["dir"], s))
    ret = (c / o - 1) * 100
    ev, last = [], {}
    for j in np.where(np.abs(ret) >= EXPLODE_PCT)[0]:
        d = 1 if ret[j] > 0 else -1
        if j - last.get(d, -10**9) < MERGE_BARS:
            continue
        last[d] = j
        ev.append((int(j), d))
    return sigs, ev


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    out = sys.argv[2] if len(sys.argv) > 2 else None
    cfg = load_config("config.toml")
    cl, now = MexcClient(cfg.mexc), int(time.time())
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 300_000]
    events, all_sigs, nsym = [], [], 0
    for t in tick:
        sym = t["symbol"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - days * 96 * 900 - 300 * 900, now), "Min15", now)
            o, h, l, c, v, tt = (np.asarray(k[f], float) for f in ("open", "high", "low", "close", "vol", "time"))
            if len(c) < 400:
                continue
            sigs, ev = analyse_symbol(o, h, l, c, v, tt)
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
            continue
        nsym += 1
        for i, d, s in sigs:                            # signal quality: reached +10% in its direction within 24h?
            fut_h, fut_l = h[i + 1:i + 1 + WIN_BARS], l[i + 1:i + 1 + WIN_BARS]
            if len(fut_h) < WIN_BARS:
                continue
            fav = (fut_h.max() / c[i] - 1) * 100 if d == 1 else (1 - fut_l.min() / c[i]) * 100
            hit = np.argmax((fut_h / c[i] - 1) * 100 >= WIN_PCT) if d == 1 else np.argmax((1 - fut_l / c[i]) * 100 >= WIN_PCT)
            adverse = ((1 - fut_l[:hit + 1].min() / c[i]) if d == 1 else (fut_h[:hit + 1].max() / c[i] - 1)) * 100 if fav >= WIN_PCT else float("nan")
            all_sigs.append({"sym": sym, "t": tt[i], "dir": d, "fav": fav, "win": fav >= WIN_PCT, "adverse": adverse, "trend": s["trend"]})
        sig_by_dir = {1: [x for x in sigs if x[1] == 1], -1: [x for x in sigs if x[1] == -1]}
        for j, d in ev:
            if j < 215 + 96:
                continue
            prior = [x for x in sig_by_dir[d] if j - LOOK_BARS <= x[0] < j]
            on_bar = any(x[0] == j for x in sig_by_dir[d])
            lead = (tt[j] - (tt[prior[-1][0]] + 900)) / 60 if prior else None   # minutes from the signal bar CLOSE to the explosion bar OPEN
            events.append({"sym": sym, "t": tt[j], "dir": d, "pct": (c[j] / o[j] - 1) * 100, "lead_min": lead, "on_bar": on_bar,
                           "fav": ((h[j:j + 96].max() / o[j] - 1) * 100 if d == 1 else (1 - l[j:j + 96].min() / o[j]) * 100)})
    lines = [f"فحص منبّه الانفجارات · {nsym} عقداً · آخر {days} يوماً (15د) · القواعد: docs/BOOM_CHECK.md", ""]
    events.sort(key=lambda e: -e["t"])
    early = [e for e in events if e["lead_min"] is not None]
    lines += [f"الانفجارات (شمعة 15د ≥ {EXPLODE_PCT:.0f}%): {len(events)} · اكتُشفت مبكراً: {len(early)} ({len(early) / max(len(events), 1) * 100:.0f}%) · "
              f"على الشمعة نفسها فقط: {sum(1 for e in events if e['lead_min'] is None and e['on_bar'])}", ""]
    if early:
        lm = np.array([e["lead_min"] for e in early])
        lines.append(f"زمن الاكتشاف قبل الانفجار: وسيط {np.median(lm) / 60:.1f} ساعة · أقصر {lm.min() / 60:.1f} · أطول {lm.max() / 60:.1f}")
    ws = [x for x in all_sigs]
    if ws:
        wins = [x for x in ws if x["win"]]
        lines.append(f"إشارات المنبّه: {len(ws)} · نجحت (+{WIN_PCT:.0f}% خلال 24س): {len(wins)} ({len(wins) / len(ws) * 100:.0f}%) · "
                     f"أسوأ نزول وسيط قبل النجاح: {np.nanmedian([x['adverse'] for x in wins]):.1f}%" if wins else f"إشارات المنبّه: {len(ws)} · لم ينجح أي منها")
        for tr in ("with", "mixed", "against"):
            g = [x for x in ws if x["trend"] == tr]
            if g:
                lines.append(f"  {tr}: {len(g)} إشارة، نجاح {sum(x['win'] for x in g) / len(g) * 100:.0f}%")
    lines += ["", "أحدث 30 انفجاراً (بغض النظر عن الاكتشاف):", "الوقت UTC | العقد | الجهة | الشمعة | أقصى حركة 24س | الاكتشاف المبكر"]
    for e in events[:30]:
        det = f"قبل {e['lead_min'] / 60:.1f} ساعة" if e["lead_min"] is not None else ("على الشمعة نفسها" if e["on_bar"] else "لم يُكتشف")
        lines.append(f"{hhmm(e['t'])} | {e['sym']} | {'صعود' if e['dir'] == 1 else 'هبوط'} | {e['pct']:+.1f}% | {e['fav']:+.0f}% | {det}")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
