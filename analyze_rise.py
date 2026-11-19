"""Why did a coin rise? Timeline of the move through the user's lens (PMO 35/20/10, MACD 12/26/9, TEMA50, LuxAlgo OB, volume). Usage: python3 analyze_rise.py QUERY"""
import statistics as st
import sys
import time
from datetime import datetime, timedelta, timezone

import luxalgo_ob as LX
import scanner as S

TZ = 4
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)
SEC = {"Min5": 300, "Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}


def kl(sym, itv, days, now):
    out = {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
    a = now - days * 86400
    while a < now:
        b = min(a + SEC[itv] * 1000, now)
        d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={b}")["data"]
        for k in out:
            out[k] += d[k]
        a = b
    seen, idx = set(), []
    for i, tt in enumerate(out["time"]):
        if tt not in seen and tt + SEC[itv] <= now:
            seen.add(tt); idx.append(i)
    return {k: [out[k][i] for i in idx] for k in out}


def state(d, now_idx=None):
    c = d["close"]
    p, s = S.pmo(c)
    m = [a - b for a, b in zip(S.ema(c, 12), S.ema(c, 26))]
    g = S.ema(m, 9)
    return p, s, m, g


def line(name, d):
    p, s, m, g = state(d)
    n = len(p) - 1
    h = m[n] - g[n]
    rise = p[n] > p[n - 1]
    return (f"  {name}: PMO أزرق {p[n]:+.2f} ({'فوق' if p[n] > 0 else 'تحت'} الصفر) · {'فوق' if p[n] > s[n] else 'تحت'} البرتقالي · {'صاعد' if rise else 'نازل'}"
            f" | مكدي {'أخضر' if h > 0 else 'أحمر'} ({'خط فوق الإشارة' if m[n] > g[n] else 'خط تحت الإشارة'})")


q = sys.argv[1].upper()
now = int(time.time())
detail = S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]
cands = [x for x in detail if q in x["symbol"].upper() and x["symbol"].endswith("_USDT")]
print("مطابقات:", [x["symbol"] for x in cands])
if not cands:
    sys.exit()
det = next((x for x in cands if x["symbol"].upper() == f"{q}_USDT"), cands[0])
sym = det["symbol"]
tk = next(x for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"] == sym)
print(f"\n== {sym} · {det.get('baseCoinName')} · مصادر السعر: {det.get('indexOrigin')}")
print(f"السعر {tk['lastPrice']} · تغيّر 24س {float(tk.get('riseFallRate') or 0)*100:+.1f}% · حجم 24س {float(tk.get('amount24') or 0)/1e6:.2f}M USDT · مراكز مفتوحة (عقود) {tk.get('holdVol')} · تمويل {float(tk.get('fundingRate') or 0)*100:+.4f}%")

d15 = kl(sym, "Min15", 20, now)
d5 = kl(sym, "Min5", 6, now)
d1 = kl(sym, "Min60", 60, now)
d4 = kl(sym, "Hour4", 200, now)
dd = kl(sym, "Day1", 250, now)
t, c, hi, lo, v = d15["time"], d15["close"], d15["high"], d15["low"], d15["vol"]

# the move: highest high in the last 72h, and the lowest low before it within 48h
n72 = 288
pk = max(range(len(t) - n72, len(t)), key=lambda i: hi[i])
bt = min(range(max(pk - 192, 0), pk + 1), key=lambda i: lo[i])
print(f"\nالحركة: قاع {lo[bt]:.8g} عند {dt(t[bt]):%d %H:%M} → قمة {hi[pk]:.8g} عند {dt(t[pk]):%d %H:%M} (UTC+4) = {(hi[pk]/lo[bt]-1)*100:+.1f}% خلال {(t[pk]-t[bt])/3600:.1f} ساعة · السعر الآن {(c[-1]/hi[pk]-1)*100:+.1f}% عن القمة")

# 15m PMO touches / crosses around the move
p, s, m, g = state(d15)
tm = S.tema(c, 50)
obs, snaps = LX.run(d15["open"], hi, lo, c)
vm = st.median(v[-960:]) or 1
print("\nلمسات PMO للصفر (15د) حول الحركة:")
for i in range(max(bt - 24, 1), min(pk + 1, len(c))):
    if p[i - 1] <= 0 < p[i]:
        bull = [ob for ob in snaps[i] if ob["bias"] == 1]
        ins = any(ob["lo"] <= c[i] <= ob["hi"] for ob in bull)
        print(f"  {dt(t[i]):%d %H:%M} إغلاق {c[i]:.8g} ({(c[i]/lo[bt]-1)*100:+.1f}% عن القاع) · مكدي {'فوق' if m[i]>g[i] else 'تحت'} الإشارة · {'فوق' if c[i]>tm[i] else 'تحت'} TEMA50 · {'داخل أوردر بلوك' if ins else 'خارج أوردر بلوك'}")
print("\nأكبر شموع الحجم في الحركة (15د، مضاعف الوسيط):")
big = sorted(range(max(bt - 12, 0), min(pk + 12, len(c))), key=lambda i: -v[i])[:5]
for i in sorted(big):
    print(f"  {dt(t[i]):%d %H:%M} حجم ×{v[i]/vm:.1f} · شمعة {(c[i]/d15['open'][i]-1)*100:+.1f}%")

print("\nقبل القاع وحوله (أقرب 15د): حالة الفريمات عند القاع")
def at(d, per, tc):
    k = max(i for i in range(len(d['time'])) if d['time'][i] + per <= tc)
    return {kk: d[kk][:k + 1] for kk in d}
tc = t[bt] + 900
for nm, d, per in (("4س", d4, 14400), ("يومي", dd, 86400), ("1س", d1, 3600)):
    print(line(nm, at(d, per, tc)))
print("\nالحالة الآن:")
for nm, d in (("5د", d5), ("15د", d15), ("1س", d1), ("4س", d4), ("يومي", dd)):
    print(line(nm, d))
print("\nمستويات: أوردر بلوك صاعد حي الآن (15د):", "؛ ".join(f"[{ob['lo']:.8g}-{ob['hi']:.8g}]" for ob in snaps[-1] if ob["bias"] == 1) or "لا يوجد")
