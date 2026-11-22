"""Per coin: radar alert price vs the HIGHEST price reached after the alert (until now), plus the current price. Alert dates located by matching 1H candles (see alert_batch.py).
Peak measured on 15-minute candles after the alert; time in UTC+4."""
import re
import statistics as st
import time
from datetime import datetime, timedelta, timezone
import scanner as S

src = open("alert_batch.py", encoding="utf-8").read()
A = eval("[" + src.split("\nA = [", 1)[1].split("\n]\nUTC3", 1)[0] + "\n]")
UTC3, UTC4 = timedelta(hours=3), timedelta(hours=4)
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + UTC4
now = int(time.time())
day0 = datetime(2026, 10, 24, 0, 0, 0, tzinfo=timezone.utc)


def fetch(sym, itv, step, a, b_end):
    out = {k: [] for k in ("time", "open", "high", "low", "close")}
    while a < b_end:
        b = min(a + step * 1000, b_end)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            raise SystemExit("fetch failed " + sym)
        for k in out:
            out[k] += d[k]
        a = b
    return out


t_lo, t_hi = int((day0 - timedelta(hours=6)).timestamp()), now
syms = sorted({a[1] for a in A})
H1 = {s: fetch(s, "Min60", 3600, t_lo, int((day0 + timedelta(days=5)).timestamp())) for s in syms}
M15 = {s: fetch(s, "Min15", 900, t_lo, t_hi) for s in syms}
rows = []
for di, sym, close, aprice, hhmmss, prob, r1, r4, vol, drop in A:
    h = H1[sym]
    cands = []
    for i in range(4, len(h["time"])):
        close_utc4 = h["time"][i] + 3600
        if abs(h["close"][i] / close - 1) > 0.003:
            continue
        radar_close = datetime.fromtimestamp(close_utc4, timezone.utc) + UTC3
        ret1 = (h["close"][i] / h["open"][i] - 1) * 100
        ret4 = (h["close"][i] / h["open"][i - 3] - 1) * 100
        score = abs(ret1 - r1) + abs(ret4 - r4) if r1 is not None else 0
        if r1 is not None and score > 3:
            continue
        cands.append((abs((radar_close.date() - (day0 + timedelta(days=di)).date()).days), score, close_utc4))
    if not cands:
        rows.append({"sym": sym, "found": False, "t": hhmmss}); continue
    cands.sort()
    close_utc4 = cands[0][2]
    hh, mm, ss = (int(x) for x in hhmmss.split(":"))
    a0 = int(datetime.fromtimestamp(close_utc4, timezone.utc).replace(minute=mm, second=ss).timestamp())
    m = M15[sym]
    idx = [i for i in range(len(m["time"])) if m["time"][i] >= (a0 // 900 + 1) * 900]
    pk = max(idx, key=lambda i: m["high"][i])
    lo = min(idx, key=lambda i: m["low"][i])
    rows.append({"sym": sym, "found": True, "a0": a0, "price": aprice, "peak": m["high"][pk], "peak_t": m["time"][pk], "low": m["low"][lo], "low_t": m["time"][lo],
                 "now": m["close"][-1], "prob": prob})
ok = [r for r in rows if r["found"]]
print(f"تنبيهات الرادار: {len(A)} · طُبقت: {len(ok)} · لم تُطبّق: {len(A)-len(ok)} ({', '.join(sorted({r['sym'][:-5] for r in rows if not r['found']}))}) · حتى {dt4(now):%d %b %H:%M} (UTC+4)\n")
for sym in sorted({r["sym"] for r in ok}):
    g = sorted([r for r in ok if r["sym"] == sym], key=lambda r: r["a0"])
    print(f"== {sym[:-5]}: {len(g)} تنبيه · السعر الآن {g[0]['now']:.6g}")
    print(f"   {'وقت التنبيه (UTC+4)':20s} {'سعر التنبيه':>11s} | {'أعلى سعر وصله':>13s} {'الارتفاع':>9s} {'متى':>14s} {'بعد':>8s} | {'الآن':>9s}")
    for r in g:
        print(f"   {dt4(r['a0']):%d %b %H:%M:%S}   {r['price']:11.6g} | {r['peak']:13.6g} {(r['peak']/r['price']-1)*100:+8.1f}% {dt4(r['peak_t']):%d %b %H:%M} {(r['peak_t']-r['a0'])/3600:6.1f} س | {(r['now']/r['price']-1)*100:+8.1f}%")
gains = [(r["peak"] / r["price"] - 1) * 100 for r in ok]
nowd = [(r["now"] / r["price"] - 1) * 100 for r in ok]
print(f"\nالحصيلة ({len(ok)} تنبيهاً): الارتفاع حتى أعلى سعر: وسيط {st.median(gains):+.1f}% · متوسط {st.mean(gains):+.1f}% · الأدنى {min(gains):+.1f}% · الأعلى {max(gains):+.1f}% · وصل ≥+25%: {sum(1 for g in gains if g>=25)} · ≥+50%: {sum(1 for g in gains if g>=50)} · ≥+100%: {sum(1 for g in gains if g>=100)}")
print(f"الفرق بين السعر الآن وسعر التنبيه: وسيط {st.median(nowd):+.1f}% · فوق سعر التنبيه الآن: {sum(1 for d in nowd if d>0)} من {len(ok)}")
hrs = [(r["peak_t"] - r["a0"]) / 3600 for r in ok]
print(f"زمن الوصول لأعلى سعر: وسيط {st.median(hrs):.0f} ساعة ({st.median(hrs)/24:.1f} يوم) · خلال 24 ساعة: {sum(1 for h in hrs if h<=24)} · خلال 3 أيام: {sum(1 for h in hrs if h<=72)}")

import json
print("JSON:" + json.dumps([{"sym": r["sym"], "a0": r["a0"], "price": r["price"], "peak": r["peak"], "peak_t": r["peak_t"], "now": r["now"], "prob": r["prob"]} for r in ok]))
