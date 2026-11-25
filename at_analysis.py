"""Reverse-engineer a channel signal: what did the chart look like at the signal time (data cut at that moment)? PMO/MACD/TEMA50 per frame, volume, range position, breakout check,
swing levels, order blocks, and where the stop / TP ladder came from. Times UTC+4."""
import statistics as st
import sys
import time
from datetime import datetime, timedelta, timezone
import luxalgo_ob as LX
import scanner as S

SYM = "AT_USDT"
ENTRY, STOP = 0.0544, 0.0517
TPS = [0.056, 0.0579, 0.0597, 0.0616, 0.0634, 0.0653]
SIG = datetime(2026, 11, 25, 21, 0, 0, tzinfo=timezone.utc) - timedelta(hours=4)        # 25 Nov 21:00 UTC+4
t_sig = int(SIG.timestamp())
now = int(time.time())
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)
SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}


def kl(itv, days):
    out = {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
    a = now - days * 86400
    while a < now:
        b = min(a + SEC[itv] * 1000, now)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{SYM}?interval={itv}&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            raise SystemExit("fetch failed")
        for k in out:
            out[k] += d[k]
        a = b
    seen, keep = set(), []
    for i, t in enumerate(out["time"]):
        if t not in seen:
            seen.add(t); keep.append(i)
    D = {k: [out[k][i] for i in keep] for k in out}
    cut = [i for i in range(len(D["time"])) if D["time"][i] + SEC[itv] <= t_sig]            # candles CLOSED at or before the signal time
    return {k: [D[k][i] for i in cut] for k in D}


D = {"Min15": kl("Min15", 20), "Min60": kl("Min60", 40), "Hour4": kl("Hour4", 120), "Day1": kl("Day1", 400)}
h1 = D["Min60"]
print(f"{SYM} · الإشارة {dt4(t_sig):%d %b %H:%M} · دخول {ENTRY} · وقف {STOP} ({(STOP/ENTRY-1)*100:.2f}%) · آخر شمعة ساعة مغلقة قبل الإشارة: إغلاقها {h1['close'][-1]} (وقتها {dt4(h1['time'][-1]):%d %b %H:%M})")
print(f"شموع يومية متاحة قبل الإشارة: {len(D['Day1']['close'])} · أول شمعة {dt4(D['Day1']['time'][0]):%d %b %Y}\n")
print("آخر 14 شمعة ساعة قبل الإشارة:")
vmed = st.median(h1["vol"][-24 * 7:-1])
for i in range(len(h1["close"]) - 14, len(h1["close"])):
    o, h, l, c, v = h1["open"][i], h1["high"][i], h1["low"][i], h1["close"][i], h1["vol"][i]
    print(f"  {dt4(h1['time'][i]):%d %H:%M} فتح {o:.5g} أعلى {h:.5g} أدنى {l:.5g} إغلاق {c:.5g} ({(c/o-1)*100:+.2f}%) · حجم ×{v/vmed:.1f}")
c1 = h1["close"]
print(f"\nالأداء قبل الإشارة: آخر 4 ساعات {(c1[-1]/c1[-5]-1)*100:+.1f}% · 24 ساعة {(c1[-1]/c1[-25]-1)*100:+.1f}% · 7 أيام {(c1[-1]/c1[-169]-1)*100:+.1f}%")
for nm, n in (("24 ساعة", 24), ("48 ساعة", 48), ("7 أيام", 168)):
    hh, ll = max(h1["high"][-n:]), min(h1["low"][-n:])
    print(f"  نطاق {nm}: {ll:.5g} – {hh:.5g} · الدخول عند {(ENTRY-ll)/(hh-ll)*100:.0f}% من النطاق · الدخول {(ENTRY/hh-1)*100:+.1f}% عن القمة")
prev_hi = max(h1["high"][-25:-1]); prev_hi48 = max(h1["high"][-49:-1])
print(f"  اختراق؟ إغلاق آخر شمعة {c1[-1]:.5g} مقابل أعلى 24س قبلها {prev_hi:.5g} ({'اخترقها' if c1[-1] > prev_hi else 'لم يخترقها'}) · أعلى 48س قبلها {prev_hi48:.5g} ({'اخترقها' if c1[-1] > prev_hi48 else 'لم يخترقها'})")
print(f"  حجم آخر ساعة {h1['vol'][-1]/vmed:.1f}× وسيط الساعة (7 أيام) · حجم آخر 4 ساعات {sum(h1['vol'][-4:])/(4*vmed):.1f}×")

print("\nحالة المؤشرات لحظة الإشارة (PMO 35,20,10 · مكدي 12,26,9 · TEMA50):")
for nm, k in (("15 دقيقة", "Min15"), ("ساعة", "Min60"), ("4 ساعات", "Hour4"), ("يومي", "Day1")):
    d = D[k]; cl = d["close"]
    if len(cl) < 200:
        print(f"  {nm}: بيانات قليلة ({len(cl)})"); continue
    p, ps = S.pmo(cl)
    m = [a - b for a, b in zip(S.ema(cl, 12), S.ema(cl, 26))]
    g = S.ema(m, 9)
    te = S.tema(cl, 50)
    h = m[-1] - g[-1]
    cross = any(p[j - 1] <= 0 < p[j] for j in range(len(p) - 6, len(p)))
    print(f"  {nm}: PMO {p[-1]:+.2f} ({'فوق' if p[-1] > 0 else 'تحت'} الصفر) {'↑' if p[-1] > p[-2] else '↓'} {'فوق' if p[-1] > ps[-1] else 'تحت'} البرتقالي{' · لمس الصفر صعوداً خلال آخر 6 شموع' if cross else ''} · مكدي {'أخضر' if h > 0 else 'أحمر'} {'↑' if h > m[-2] - g[-2] else '↓'} · السعر {'فوق' if cl[-1] > te[-1] else 'تحت'} TEMA50 ({(cl[-1]/te[-1]-1)*100:+.2f}%)")

print("\nقمم/قيعان بارزة قريبة من السعر (±12%):")
lv = []
for nm, k, w in (("ساعة", "Min60", 5), ("4 ساعات", "Hour4", 4), ("يومي", "Day1", 3)):
    d = D[k]
    for i in range(w, len(d["close"]) - w):
        if d["high"][i] == max(d["high"][i - w:i + w + 1]):
            lv.append((d["high"][i], "قمة", nm, d["time"][i]))
        if d["low"][i] == min(d["low"][i - w:i + w + 1]):
            lv.append((d["low"][i], "قاع", nm, d["time"][i]))
near = sorted([x for x in lv if abs(x[0] / ENTRY - 1) <= 0.12], key=lambda x: -x[0])
for p_, kind, nm, t_ in near[:22]:
    print(f"  {p_:.5g} {kind} ({nm}, {dt4(t_):%d %b}) · {(p_/ENTRY-1)*100:+.1f}% عن الدخول")

print("\nأوردر بلوك (LuxAlgo) الحية لحظة الإشارة:")
for nm, k in (("ساعة", "Min60"), ("4 ساعات", "Hour4"), ("يومي", "Day1")):
    d = D[k]
    if len(d["close"]) < 250:
        print(f"  {nm}: بيانات غير كافية ({len(d['close'])})"); continue
    _, sn = LX.run(d["open"], d["high"], d["low"], d["close"])
    for ob in sn[-1]:
        inside = ob["lo"] <= ENTRY <= ob["hi"]
        print(f"  {nm}: {'صاعد' if ob['bias'] == 1 else 'هابط'} [{ob['lo']:.5g} – {ob['hi']:.5g}] · {(ob['lo']/ENTRY-1)*100:+.1f}% إلى {(ob['hi']/ENTRY-1)*100:+.1f}% عن الدخول{' · الدخول داخله' if inside else ''}")

print("\nمن أين جاء الوقف والأهداف؟")
print(f"  الوقف {STOP}: -{(1-STOP/ENTRY)*100:.2f}% · أدنى قاع آخر 6 ساعات {min(h1['low'][-6:]):.5g} · 12 ساعة {min(h1['low'][-12:]):.5g} · 24 ساعة {min(h1['low'][-24:]):.5g} · 48 ساعة {min(h1['low'][-48:]):.5g}")
print("  الأهداف كنسب من الدخول: " + " · ".join(f"TP{n} {(t/ENTRY-1)*100:+.2f}%" for n, t in enumerate(TPS, 1)))
steps = [(TPS[i] / TPS[i - 1] - 1) * 100 for i in range(1, 6)]
print(f"  فروق الأهداف المتتالية: " + ", ".join(f"{x:.2f}%" for x in steps) + " → سلّم شبه ثابت (≈ +3.3% لكل هدف من +3% إلى +20%)")
for n, t in enumerate(TPS, 1):
    close_lv = [x for x in lv if abs(x[0] / t - 1) <= 0.006 and x[0] > ENTRY]
    print(f"  TP{n} {t}: " + (", ".join(f"{x[1]} {x[2]} عند {x[0]:.5g}" for x in close_lv[:3]) if close_lv else "لا قمة أو قاع بارز على بعد 0.6%"))
