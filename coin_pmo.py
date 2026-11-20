"""Analyse one coin with our PMO-only rule: current PMO state on 5m/15m/1H/4H/daily, last 15m zero-touches (last 7 days) with what the rule would have done (stop 1.5xATR, exit = blue crosses below orange, max 24h)."""
import sys
import time
from datetime import datetime, timedelta, timezone
import scanner as S
import scanner_pmo as P

TZ = 4
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)
q = sys.argv[1].upper()
now = int(time.time())
detail = S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]
c_ = [x for x in detail if q in x["symbol"].upper() and x["symbol"].endswith("_USDT")]
print("مطابقات:", [x["symbol"] for x in c_])
det = next((x for x in c_ if x["symbol"].upper() == f"{q}_USDT"), c_[0])
sym = det["symbol"]
tk = next(x for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"] == sym)
print(f"== {sym[:-5]}USDT.P · مصادر السعر {det.get('indexOrigin')} · كريبتو حسب الفلتر: {S.is_crypto(det)}")
print(f"السعر {tk['lastPrice']} · 24س {float(tk.get('riseFallRate') or 0)*100:+.1f}% · حجم 24س {float(tk.get('amount24') or 0)/1e6:.2f}M · تمويل {float(tk.get('fundingRate') or 0)*100:+.4f}%")
d = {k: P.kl(sym, k, now) for k in ("Min5", "Min15", "Min60", "Hour4", "Day1")}
P.DAYS["Min15"] = 20
t, hi, lo, c = d["Min15"]["time"], d["Min15"]["high"], d["Min15"]["low"], d["Min15"]["close"]
n = len(c)
p, ps = S.pmo(c)
tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
atr = S.ema(tr, 14)
print("\nحالة PMO الآن:")
for nm, key, per in (("5د", "Min5", 300), ("15د", "Min15", 900), ("1س", "Min60", 3600), ("4س", "Hour4", 14400), ("يومي", "Day1", 86400)):
    print(f"  {nm}: {P.fmt(P.fstate(d[key], per, now, c[-1]))}")
daily = P.fstate(d["Day1"], 86400, now, c[-1])
print(f"\nهل يطابق شرطنا (اليومي فوق الصفر)؟ {'نعم' if daily and daily['pos'] else 'لا'}")
sp = p[-1] - p[-2]
print(f"15د الآن: PMO {p[-1]:+.3f} · {'فوق' if p[-1] > ps[-1] else 'تحت'} البرتقالي · {'صاعد' if sp > 0 else 'نازل'}" + (f" · يبعد عن الصفر {(-p[-1]/sp):.1f} شمعة بالسرعة الحالية" if p[-1] < 0 < sp else ""))
print("\nلمسات الصفر على 15 دقيقة في آخر 7 أيام وماذا كانت ستفعل القاعدة:")
cnt = 0
for i in range(n - 1, max(n - 7 * 96, 300), -1):
    if p[i - 1] <= 0 < p[i]:
        ent = c[i]
        stp = min(max(ent - 1.5 * atr[i], ent * 0.94), ent * 0.995)
        risk = ent - stp
        out, why = None, "مفتوحة"
        for j in range(i + 1, min(i + 97, n)):
            if lo[j] <= stp:
                out, why = stp, "وقف"; break
            if p[j] < ps[j] and p[j - 1] >= ps[j - 1]:
                out, why = c[j], "إشارة خروج"; break
        else:
            if i + 96 < n:
                out, why = c[i + 96], "24 ساعة"
        px = out if out else c[-1]
        R = (px - ent) / risk - 0.001 * ent / risk
        d1 = P.fstate(d["Day1"], 86400, t[i] + 900, ent)
        print(f"  {dt(t[i]):%d %H:%M} سعر {ent:.6g} · اليومي {'فوق' if d1 and d1['pos'] else 'تحت'} الصفر · {why} · R {R:+.2f}")
        cnt += 1
        if cnt >= 12:
            break
if cnt == 0:
    print("  لا لمسات في آخر 7 أيام")
