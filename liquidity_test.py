"""Liquidity-entry test (long only). For every 15m PMO(35,20,10) zero-touch (up) in the last 14 days, mark whether the 1H and 4H frames show
"start or approach of liquidity entry" (definition agreed with the user), then print raw forward price numbers (no verdicts).
Frame OK = MACD ok OR PMO ok.  MACD ok: hist>0, or hist<0 and shrinking (rising).  PMO ok: blue>orange, or blue<orange, rising, gap shrinking.
Extra: 4H PMO blue higher than 2 candles ago.  Pass = 1H OK and 4H OK and 4H PMO rising.  Frames include the forming candle (price at the touch)."""
import json
import statistics as st
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import scanner as S

TZ = 4
SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400}
DAYS = {"Min15": 14, "Min60": 40, "Hour4": 120}
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)


def kline(sym, itv, now):
    for a in range(3):
        try:
            d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={now - DAYS[itv] * 86400}&end={now}")["data"]
            return d["time"], d["high"], d["low"], d["close"]
        except Exception:
            time.sleep(1 + a)
    return None


def frame(times, closes, period, tc, price):
    series = [closes[k] for k in range(len(times)) if times[k] + period <= tc] + [price]
    if len(series) < 120:
        return None
    p, s = S.pmo(series)
    ml = [a - b for a, b in zip(S.ema(series, 12), S.ema(series, 26))]
    sg = S.ema(ml, 9)
    h = [a - b for a, b in zip(ml, sg)]
    n = len(series) - 1
    macd_ok = h[n] > 0 or h[n] > h[n - 1]
    gap, gp = p[n] - s[n], p[n - 1] - s[n - 1]
    pmo_ok = gap > 0 or (p[n] > p[n - 1] and gap > gp)
    return {"macd": macd_ok, "pmo": pmo_ok, "ok": macd_ok or pmo_ok, "rise2": p[n] > p[n - 2]}


def work(sym):
    now = int(time.time())
    k15, k60, k4 = kline(sym, "Min15", now), kline(sym, "Min60", now), kline(sym, "Hour4", now)
    if not (k15 and k60 and k4):
        return []
    t, hi, lo, c = k15
    keep = [i for i in range(len(t)) if t[i] + 900 <= now]
    t, hi, lo, c = [[x[i] for i in keep] for x in (t, hi, lo, c)]
    p, _ = S.pmo(c)
    out = []
    for i in range(400, len(c)):
        if not (p[i - 1] <= 0 < p[i]):
            continue
        tc = t[i] + 900
        f1, f4 = frame(k60[0], k60[3], 3600, tc, c[i]), frame(k4[0], k4[3], 14400, tc, c[i])
        if not f1 or not f4:
            continue
        ent = c[i]
        fwd = i + 48 < len(c)
        up = dn = first = None
        if fwd:
            up = (max(hi[i + 1:i + 49]) / ent - 1) * 100
            dn = (min(lo[i + 1:i + 49]) / ent - 1) * 100
            first = 0
            for j in range(i + 1, i + 49):                       # which came first: +3% or -3%
                if hi[j] >= ent * 1.03 and lo[j] <= ent * 0.97:
                    first = 0; break
                if hi[j] >= ent * 1.03:
                    first = 1; break
                if lo[j] <= ent * 0.97:
                    first = -1; break
        out.append({"sym": sym, "t": t[i], "f1": f1, "f4": f4, "pass": f1["ok"] and f4["ok"] and f4["rise2"], "up": up, "dn": dn, "first": first, "fwd": fwd})
    return out


def report(name, rows):
    rows = [r for r in rows if r["fwd"]]
    if not rows:
        print(f"{name:28s} 0")
        return
    ups, dns = [r["up"] for r in rows], [r["dn"] for r in rows]
    a = sum(1 for r in rows if r["first"] == 1) / len(rows) * 100
    b = sum(1 for r in rows if r["first"] == -1) / len(rows) * 100
    print(f"{name:28s} {len(rows):5d}   وسيط أعلى صعود 12س {st.median(ups):+6.2f}%   وسيط أعمق هبوط {st.median(dns):+6.2f}%   +3% أولاً {a:4.1f}%   -3% أولاً {b:4.1f}%")


def main():
    detail = {d["symbol"]: d for d in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000]
    print(f"عملات مفحوصة: {len(tick)} · آخر 14 يوماً · دخول = إغلاق شمعة لمس الصفر · التوقيت UTC+{TZ}\n")
    with ThreadPoolExecutor(6) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    print(f"{'المجموعة':28s} {'لمسات':>5s}")
    report("كل اللمسات", rows)
    report("مرّت بالشرط الكامل", [r for r in rows if r["pass"]])
    report("لم تمرّ", [r for r in rows if not r["pass"]])
    print()
    report("الساعة OK فقط", [r for r in rows if r["f1"]["ok"]])
    report("الساعة غير OK", [r for r in rows if not r["f1"]["ok"]])
    report("4س OK فقط", [r for r in rows if r["f4"]["ok"]])
    report("4س غير OK", [r for r in rows if not r["f4"]["ok"]])
    report("4س PMO صاعد", [r for r in rows if r["f4"]["rise2"]])
    report("4س PMO غير صاعد", [r for r in rows if not r["f4"]["rise2"]])
    print("\nلمسات آخر 36 ساعة (مباشر):")
    now = time.time()
    for r in sorted((r for r in rows if now - r["t"] < 36 * 3600), key=lambda r: r["t"]):
        f1, f4 = r["f1"], r["f4"]
        print(f"  {dt(r['t']):%d %H:%M} {r['sym']:18s} {'مرّت' if r['pass'] else 'لا'}  | 1س مكدي {'✓' if f1['macd'] else '✗'} pmo {'✓' if f1['pmo'] else '✗'} | 4س مكدي {'✓' if f4['macd'] else '✗'} pmo {'✓' if f4['pmo'] else '✗'} صاعد {'✓' if f4['rise2'] else '✗'}")
    json.dump(rows, open("liquidity_rows.json", "w"))


if __name__ == "__main__":
    main()
