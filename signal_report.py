"""Generic channel-signal report: locate the signal (latest 1H close within 0.5% of the entry), follow-up on 1m candles (TPs, stop, highs/lows, current), and the chart state at the signal time
(data cut at that moment): PMO/MACD/TEMA50 per frame, volume, range, breakout, swing levels, order blocks, origin of stop/TP ladder. Usage: signal_report.py SYMBOL ENTRY STOP TP1,TP2,.. [HH:MM UTC+4 override date YYYY-MM-DD]"""
import statistics as st
import sys
import time
from datetime import datetime, timedelta, timezone
import luxalgo_ob as LX
import scanner as S

SYM, ENTRY, STOP = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
TPS = [float(x) for x in sys.argv[4].split(",")]
now = int(time.time())
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)
SEC = {"Min1": 60, "Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}


def kl(itv, days, cut=None):
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
    if cut is not None:
        ok = [i for i in range(len(D["time"])) if D["time"][i] + SEC[itv] <= cut]
        D = {k: [D[k][i] for i in ok] for k in D}
    return D


h1_all = kl("Min60", 21)
cands = [h1_all["time"][i] + 3600 for i in range(len(h1_all["time"])) if abs(h1_all["close"][i] / ENTRY - 1) <= 0.005]
print(f"{SYM}: شموع ساعة أغلقت ضمن ±0.5% من الدخول {ENTRY} خلال 21 يوماً: {len(cands)}" + (f" · الأحدث: {dt4(cands[-1]):%d %b %H:%M}" if cands else ""))
if not cands:
    raise SystemExit("لا مطابقة")
print("  كل اللحظات: " + ", ".join(f"{dt4(t):%d %b %H:%M}" for t in cands[-8:]))
t_sig = cands[-1]
print(f"الإشارة المفترضة: {dt4(t_sig):%d %b %H:%M} (UTC+4) · مرّ عليها {(now-t_sig)/3600:.2f} ساعة\n")

# ------- follow-up -------
m = kl("Min1", 2 + int((now - t_sig) / 86400))
idx = [i for i in range(len(m["time"])) if m["time"][i] >= t_sig]
last = m["close"][-1]
hit, st_t = {}, None
for i in idx:
    if st_t is None and m["low"][i] <= STOP:
        st_t = m["time"][i]
    for n, tp in enumerate(TPS, 1):
        if n not in hit and m["high"][i] >= tp and (st_t is None or m["time"][i] < st_t):
            hit[n] = m["time"][i]
mx = max(idx, key=lambda i: m["high"][i]); mn = min(idx, key=lambda i: m["low"][i])
print(f"== المتابعة · السعر الآن {last} ({(last/ENTRY-1)*100:+.2f}% عن الدخول) · وقف {STOP} ({(STOP/ENTRY-1)*100:+.2f}%)")
for n, tp in enumerate(TPS, 1):
    print(f"   TP{n} {tp} ({(tp/ENTRY-1)*100:+.2f}%): " + (f"بُلغ {dt4(hit[n]):%d %b %H:%M} (بعد {(hit[n]-t_sig)/60:.0f} دقيقة)" if n in hit else "لم يُبلغ"))
print("   الوقف: " + (f"ضُرب {dt4(st_t):%d %b %H:%M} (بعد {(st_t-t_sig)/60:.0f} دقيقة)" if st_t else "لم يُضرب"))
print(f"   أعلى سعر منذ الدخول {m['high'][mx]:.6g} ({(m['high'][mx]/ENTRY-1)*100:+.2f}%) {dt4(m['time'][mx]):%d %b %H:%M} · أدنى {m['low'][mn]:.6g} ({(m['low'][mn]/ENTRY-1)*100:+.2f}%) {dt4(m['time'][mn]):%d %b %H:%M}\n")

# ------- chart state at the signal time -------
D = {"Min15": kl("Min15", 20, t_sig), "Min60": kl("Min60", 40, t_sig), "Hour4": kl("Hour4", 120, t_sig), "Day1": kl("Day1", 400, t_sig)}
h1 = D["Min60"]
print("== حالة الشارت لحظة الإشارة (بيانات حتى تلك اللحظة فقط)")
vmed = st.median(h1["vol"][-24 * 7:-1])
print("آخر 8 شموع ساعة:")
for i in range(len(h1["close"]) - 8, len(h1["close"])):
    o, h, l, c, v = h1["open"][i], h1["high"][i], h1["low"][i], h1["close"][i], h1["vol"][i]
    print(f"  {dt4(h1['time'][i]):%d %H:%M} فتح {o:.5g} أعلى {h:.5g} أدنى {l:.5g} إغلاق {c:.5g} ({(c/o-1)*100:+.2f}%) · حجم ×{v/vmed:.1f}")
c1 = h1["close"]
print(f"الأداء قبل الإشارة: 4 ساعات {(c1[-1]/c1[-5]-1)*100:+.1f}% · 24 ساعة {(c1[-1]/c1[-25]-1)*100:+.1f}% · 7 أيام {(c1[-1]/c1[-169]-1)*100:+.1f}%")
for nm, n in (("24 ساعة", 24), ("48 ساعة", 48), ("7 أيام", 168)):
    hh, ll = max(h1["high"][-n:]), min(h1["low"][-n:])
    print(f"  نطاق {nm}: {ll:.5g} – {hh:.5g} · الدخول عند {(ENTRY-ll)/(hh-ll)*100:.0f}% من النطاق · {(ENTRY/hh-1)*100:+.1f}% عن القمة")
print(f"  أعلى 24س قبل الأخيرة {max(h1['high'][-25:-1]):.5g} ({'اخترقها' if c1[-1] > max(h1['high'][-25:-1]) else 'لم يخترقها'}) · حجم آخر ساعة ×{h1['vol'][-1]/vmed:.1f} · آخر 4 ساعات ×{sum(h1['vol'][-4:])/(4*vmed):.1f}")
print("حالة المؤشرات:")
for nm, k in (("15 دقيقة", "Min15"), ("ساعة", "Min60"), ("4 ساعات", "Hour4"), ("يومي", "Day1")):
    cl = D[k]["close"]
    if len(cl) < 200:
        print(f"  {nm}: بيانات قليلة ({len(cl)})"); continue
    p, ps = S.pmo(cl)
    mm = [a - b for a, b in zip(S.ema(cl, 12), S.ema(cl, 26))]
    g = S.ema(mm, 9)
    te = S.tema(cl, 50)
    h = mm[-1] - g[-1]
    print(f"  {nm}: PMO {p[-1]:+.2f} ({'فوق' if p[-1] > 0 else 'تحت'} الصفر) {'↑' if p[-1] > p[-2] else '↓'} {'فوق' if p[-1] > ps[-1] else 'تحت'} البرتقالي · مكدي {'أخضر' if h > 0 else 'أحمر'} {'↑' if h > mm[-2] - g[-2] else '↓'} · السعر {'فوق' if cl[-1] > te[-1] else 'تحت'} TEMA50 ({(cl[-1]/te[-1]-1)*100:+.2f}%)")
lv = []
for nm, k, w in (("ساعة", "Min60", 5), ("4 ساعات", "Hour4", 4), ("يومي", "Day1", 3)):
    d = D[k]
    for i in range(w, len(d["close"]) - w):
        if d["high"][i] == max(d["high"][i - w:i + w + 1]):
            lv.append((d["high"][i], "قمة", nm, d["time"][i]))
        if d["low"][i] == min(d["low"][i - w:i + w + 1]):
            lv.append((d["low"][i], "قاع", nm, d["time"][i]))
print("قمم/قيعان بارزة قريبة (±12%):")
for p_, kind, nm, t_ in sorted([x for x in lv if abs(x[0] / ENTRY - 1) <= 0.12], key=lambda x: -x[0])[:16]:
    print(f"  {p_:.5g} {kind} ({nm}, {dt4(t_):%d %b}) · {(p_/ENTRY-1)*100:+.1f}%")
print("أوردر بلوك الحية:")
for nm, k in (("ساعة", "Min60"), ("4 ساعات", "Hour4"), ("يومي", "Day1")):
    d = D[k]
    if len(d["close"]) < 250:
        print(f"  {nm}: بيانات غير كافية ({len(d['close'])})"); continue
    _, sn = LX.run(d["open"], d["high"], d["low"], d["close"])
    for ob in sn[-1]:
        if abs(ob["lo"] / ENTRY - 1) <= 0.35:
            print(f"  {nm}: {'صاعد' if ob['bias'] == 1 else 'هابط'} [{ob['lo']:.5g} – {ob['hi']:.5g}] · {(ob['lo']/ENTRY-1)*100:+.1f}% إلى {(ob['hi']/ENTRY-1)*100:+.1f}%{' · الدخول داخله' if ob['lo'] <= ENTRY <= ob['hi'] else ''}")
print(f"الوقف {STOP}: أدنى قاع 6/12/24/48 ساعة = {min(h1['low'][-6:]):.5g} / {min(h1['low'][-12:]):.5g} / {min(h1['low'][-24:]):.5g} / {min(h1['low'][-48:]):.5g}")
print("الأهداف: " + " · ".join(f"TP{n} {(t/ENTRY-1)*100:+.2f}%" for n, t in enumerate(TPS, 1)))
