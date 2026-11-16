"""Fact-check + pattern description of copied channel signals using 1-minute MEXC candles (rules: docs/CHANNEL_CHECK2.md).
Usage: python -m tradebot.strategies.mtf_entry.channel_check2 [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from tradebot.config import load_config
from tradebot.data.klines import drop_unclosed, fetch_range
from tradebot.mexc.client import MexcClient

DAYS, WIN, LIQ = 45, 3, 0.045


def hm(s):                                   # "HH:MM" Riyadh -> minute-of-day UTC
    h, m = map(int, s.split(":"))
    return ((h * 60 + m) - 180) % 1440


SIGNALS = [
    dict(sym="IDOL_USDT", entry=0.034382, stop=0.032663, tps=[0.035242, 0.036445, 0.037648, 0.038852, 0.040055, 0.041258],
         hits={0: "19:25", 1: "19:27", 3: "19:43", 4: "19:46", 5: "19:48"}),
    dict(sym="HMSTR_USDT", entry=0.000134, stop=0.00012702, tps=[0.00013658, 0.00013792, 0.0001406, 0.00014039, 0.00014707],
         hits={0: "22:47", 1: "22:47", 2: "23:15", 3: "23:15", 4: "23:18"}),
    dict(sym="AIOT_USDT", entry=0.11449, stop=0.10877, tps=[0.11735, 0.12136, 0.12537, 0.12937, 0.13338, 0.13739],
         hits={0: "03:48", 1: "04:11", 2: "07:48", 3: "10:14", 5: "12:51"}),
    dict(sym="JCT_USDT", entry=0.003773, stop=0.003584, tps=[0.004131, 0.004263, 0.004396, 0.004528],
         hits={0: "12:46", 1: "12:46", 2: "12:47", 3: "12:50"}),
    dict(sym="Q_USDT", entry=0.03107, stop=0.03072, tps=[0.03213, 0.03292, 0.0337, 0.03449], hits={0: "16:06"}),
    dict(sym="ON_USDT", entry=0.1817, stop=0.1726, tps=[0.2117, 0.218], hits={0: "18:25", 1: "18:25"}),
    dict(sym="ARIA_USDT", entry=0.16018, stop=0.15217, tps=[0.16499, 0.17043, 0.17588, 0.18132, 0.18677, 0.19222],
         hits={0: "00:35", 1: "17:25", 2: "18:04"}),
    dict(sym="AVAX_USDT", entry=8.4274, stop=8.006, tps=[8.6381, 8.933, 9.228, 9.523, 9.8179, 10.1129], hits={0: "16:24", 1: "17:49"}),
    dict(sym="BARD_USDT", entry=0.1253, stop=0.119, tps=[0.1291, 0.1333], hits={0: "15:38"}),
    dict(sym="BTW_USDT", entry=0.695, stop=0.66, tps=[0.716, 0.739, 0.763], hits={0: "16:36", 1: "20:26"}),
]


def touches(h, tp, look=30):
    idx = []
    for i in range(look, len(h)):
        if h[i] >= tp and h[i - look:i].max() < tp:
            idx.append(i)
    return np.array(idx, int)


def locate(h, t, sig):
    by_day = {}
    for k, s in sig["hits"].items():
        for i in touches(h, sig["tps"][k]):
            mod = int((t[i] % 86400) // 60)
            if min((mod - hm(s)) % 1440, (hm(s) - mod) % 1440) <= WIN:
                by_day.setdefault(int(t[i] // 86400), {}).setdefault(k, int(i))
    if not by_day:
        return None, 0, 0
    best = max(by_day, key=lambda d: (len(by_day[d]), d))
    return by_day[best], len(by_day[best]), len([d for d in by_day if len(by_day[d]) == len(by_day[best])])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    lines = ["فحص توصيات القناة بشموع الدقيقة · القواعد: docs/CHANNEL_CHECK2.md", ""]
    rows = []
    for sig in SIGNALS:
        sym = sig["sym"]
        try:
            k = drop_unclosed(fetch_range(cl, sym, "Min1", now // 60 * 60 - DAYS * 86400, now), "Min1", now)
            o, h, l, c, v, t = (np.asarray(k[f], float) for f in ("open", "high", "low", "close", "vol", "time"))
        except Exception as e:
            lines.append(f"{sym}: فشل الجلب {e}"); continue
        m, nmatch, ncand = locate(h, t, sig)
        if m is None or (nmatch < 2 and len(sig["hits"]) > 1) or ncand > 1 and nmatch < 2:
            lines.append(f"{sym}: غامض (تطابق {nmatch} هدف، مرشحون {ncand})"); continue
        i1 = min(m.values())
        e = sig["entry"]
        lo = max(i1 - 1440, 0)
        reach = np.where(l[lo:i1 + 1] <= e * 1.002)[0]
        if len(reach) == 0:
            lines.append(f"{sym}: {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t[i1]))} UTC · تطابق {nmatch} · السعر لم يلمس الدخول خلال 24س قبل الهدف الأول"); continue
        j = lo + int(reach[-1])
        lead = (t[i1] - t[j]) / 60
        r24 = (c[j] / c[max(j - 1440, 0)] - 1) * 100
        r6 = (c[j] / c[max(j - 360, 0)] - 1) * 100
        n15 = (j - 96 * 15) // 15
        s15 = np.array([v[a:a + 15].sum() for a in range(max(j - 96 * 60, 0), j - 14, 15)])
        vr = v[j - 14:j + 1].sum() / max(np.median(s15), 1e-9) if len(s15) > 10 else float("nan")
        seg = slice(max(j - 2880, 0), j + 1)
        c15 = [(c[a + 14] / o[a] - 1) * 100 for a in range(seg.start, seg.stop - 14, 15)]
        ex48 = sum(1 for x in c15 if x >= 10)
        mae = (l[j:i1 + 1].min() / e - 1) * 100
        liq = l[j:i1 + 1].min() <= e * (1 - LIQ)
        stop_i = None
        for q in range(j, min(j + 4320, len(c))):                     # first 4H close below the announced stop
            if (t[q] + 60) % 14400 == 0 and c[q] < sig["stop"]:
                stop_i = q; break
        stop_before = stop_i is not None and stop_i < i1
        mfe = (h[j:j + 1440].max() / e - 1) * 100
        rows.append(dict(sym=sym, lead=lead, r24=r24, r6=r6, vr=vr, ex48=ex48, mae=mae, liq=liq, mfe=mfe))
        lines.append(f"{sym}: {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t[i1]))} UTC (TP1) · تطابق {nmatch} · دخول متاح قبل TP1 بـ {lead:.0f} د · "
                     f"عائد 24س قبل الدخول {r24:+.1f}% · 6س {r6:+.1f}% · حجم آخر 15د ×{vr:.1f} · انفجارات 48س: {ex48} · "
                     f"أسوأ هبوط قبل TP1 {mae:.1f}%{' (لمس مستوى التصفية 20x)' if liq else ''} · إغلاق 4H تحت الوقف قبل TP1: {'نعم' if stop_before else 'لا'} · أقصى صعود 24س {mfe:+.0f}%")
    if rows:
        a = lambda key: np.array([r[key] for r in rows], float)
        lines += ["", f"ملخص ({len(rows)} توصية محدّدة): وسيط مهلة الدخول المتاح قبل TP1 {np.median(a('lead')):.0f} د · وسيط عائد 24س قبل الدخول {np.median(a('r24')):+.1f}% · "
                  f"عندها انفجار 15د≥10% خلال 48س: {int((a('ex48') >= 1).sum())}/{len(rows)} · قفزة حجم ≥3×: {int((a('vr') >= 3).sum())}/{len(rows)} · "
                  f"لمست مستوى تصفية 20x قبل TP1: {int(a('liq').sum())}/{len(rows)}"]
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w").write(text)


if __name__ == "__main__":
    main()
