"""Fact-check a Telegram channel's claimed take-profit hits against MEXC 1-minute candles.
Claims are copied from the pasted messages (times are Riyadh = UTC+3, day 2026-10-17).
For each claim: first time the 1m HIGH reached the level that day (Riyadh time) vs the claimed time;
daily high/low; whether the level was ever reached."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tradebot.config import load_config                  # noqa: E402
from tradebot.mexc.client import MexcClient              # noqa: E402

RIYADH = timezone(timedelta(hours=3))
DAY = datetime(2026, 10, 17, tzinfo=RIYADH)

# (symbol, label, level, claimed time "HH:MM" Riyadh or None)
CLAIMS = [
    ("SAND_USDT", "TP1", 0.06031, "14:22"), ("SAND_USDT", "TP2", 0.06178, "14:50"),
    ("SAND_USDT", "TP3", 0.06325, "17:17"), ("SAND_USDT", "TP4", 0.06472, "18:13"),
    ("Q_USDT", "TP1 (entry 0.070963)", 0.072737, "15:12"), ("Q_USDT", "TP2", 0.074511, "15:22"),
    ("Q_USDT", "TP3", 0.076285, None), ("Q_USDT", "TP4", 0.078059, "17:05"),
    ("Q_USDT", "TP7 (+43.2%)", 0.0779, "16:54"), ("Q_USDT", "TP8 (+50%)", 0.0816, "17:45"),
    ("PROM_USDT", "TP1", 5.097, "15:32"),
    ("ICNT_USDT", "TP1 (entry 0.1901)", 0.1949, "19:04"), ("ICNT_USDT", "TP2", 0.1996, "19:08"),
    ("ICNT_USDT", "TP3", 0.2044, "19:09"), ("ICNT_USDT", "TP4", 0.2091, "19:12"),
    ("BAT_USDT", "TP1 (entry 0.18156)", 0.1861, "19:50"),
]
ENTRIES = {"Q_USDT": 0.070963, "ICNT_USDT": 0.1901, "BAT_USDT": 0.18156, "SAND_USDT": 0.05845}


def main():
    cfg = load_config("config.toml")
    c = MexcClient(cfg.mexc)
    start = int(DAY.timestamp())
    end = start + 24 * 3600
    L = ["# فحص ادعاءات القناة بشموع MEXC (دقيقة واحدة، يوم 2026-10-17، توقيت الرياض)", ""]
    data = {}
    for sym in sorted({x[0] for x in CLAIMS}):
        try:
            d = c.klines(sym, "Min1", start, end)
            data[sym] = d if len(d.get("time", [])) else None
        except Exception as e:
            data[sym] = None
            L.append(f"- {sym}: فشل الجلب: {e}")
    for sym, d in data.items():
        if d is None:
            L.append(f"- {sym}: لا توجد شموع دقيقة متاحة (قد لا يحتفظ MEXC بها لهذا اليوم أو العقد غير موجود)")
            continue
        t, hi, lo = np.asarray(d["time"]), np.asarray(d["high"], float), np.asarray(d["low"], float)
        L.append(f"- {sym}: {len(t)} شمعة؛ أعلى سعر في اليوم {hi.max()} · أدنى {lo.min()}")
    L += ["", "| العملة | الهدف | المستوى | الوقت المُعلن | أول لمس فعلي (1م) | الفرق | بلغه السعر؟ |", "|---|---|---|---|---|---|---|"]
    for sym, lab, lvl, claimed in CLAIMS:
        d = data.get(sym)
        if d is None:
            L.append(f"| {sym} | {lab} | {lvl} | {claimed or '-'} | لا بيانات | | |")
            continue
        t, hi = np.asarray(d["time"]), np.asarray(d["high"], float)
        idx = np.where(hi >= lvl)[0]
        if len(idx) == 0:
            L.append(f"| {sym} | {lab} | {lvl} | {claimed or '-'} | **لم يصل أبداً** (أعلى {hi.max()}) | | ❌ |")
            continue
        first = datetime.fromtimestamp(int(t[idx[0]]), RIYADH)
        diff = ""
        if claimed:
            ch, cm = map(int, claimed.split(":"))
            claimed_dt = DAY.replace(hour=ch, minute=cm)
            diff = f"{(first - claimed_dt).total_seconds() / 60:+.0f} د"
        L.append(f"| {sym} | {lab} | {lvl} | {claimed or '-'} | {first:%H:%M} | {diff} | ✅ |")
    # context: price around the claimed entries
    L += ["", "## السعر حول الأوقات المُعلنة (إغلاق الدقيقة، توقيت الرياض)", ""]
    for sym, t0 in (("ICNT_USDT", "19:04"), ("Q_USDT", "15:12"), ("BAT_USDT", "19:50"), ("SAND_USDT", "14:22")):
        d = data.get(sym)
        if d is None:
            continue
        t, cl = np.asarray(d["time"]), np.asarray(d["close"], float)
        h, m = map(int, t0.split(":"))
        c0 = DAY.replace(hour=h, minute=m)
        row = []
        for off in (-60, -30, -10, 0, 10, 30):
            ts = int((c0 + timedelta(minutes=off)).timestamp())
            k = np.searchsorted(t, ts)
            if k < len(t):
                row.append(f"{off:+d}د: {cl[k]}")
        L.append(f"- {sym} حول {t0}: " + " · ".join(row) + f" (سعر الدخول المُعلن {ENTRIES[sym]})")
    out = Path("reports/channel")
    out.mkdir(parents=True, exist_ok=True)
    (out / "channel_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
