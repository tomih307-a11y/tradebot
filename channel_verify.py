"""Verify the Gulf Digital Elite channel record against MEXC 1m candles. Claimed TP times are Riyadh (UTC+3); output times UTC+4 (user clock).
For each signal: entry/stop/TPs (ladder 3.0/6.4/9.8/13.2/16.6/20 %), start = hourly close nearest the posted entry before the first claimed TP (mode latest/earliest in a window),
then: TPs actually reached (first minute), claimed vs actual time, worst drawdown, stop by wick, stop by 4H close under the stop (the channel's own rule), now."""
import time
from datetime import datetime, timedelta, timezone
import scanner as S

now = int(time.time())
R = lambda d, h, m: int(datetime(2026, 11, d, h, m, tzinfo=timezone.utc).timestamp()) - 3 * 3600   # Riyadh clock -> epoch
LAD = [3.0, 6.4, 9.8, 13.2, 16.6, 20.0]
u4 = lambda x: (datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)).strftime("%d %H:%M")
# name, entry, claims {tp_index: epoch}, window_hours, mode, fixed_start(epoch or None), inferred
SIG = [
    ("CYS", 0.0609 / 1.132, {4: R(24, 1, 26)}, 72, "latest", None, True),
    ("AIOT", 0.07409 / 1.03, {1: R(24, 2, 35), 2: R(24, 7, 2), 3: R(24, 9, 35), 4: R(24, 12, 2), 5: R(24, 12, 31), 6: R(24, 12, 41)}, 72, "latest", None, True),
    ("LISTA", 0.10771, {1: R(24, 16, 7), 2: R(24, 16, 17), 3: R(25, 19, 22), 4: R(25, 19, 24)}, 24, "earliest", None, False),
    ("BTW", 0.787 / 1.132, {4: R(24, 22, 19), 5: R(25, 2, 11), 6: R(25, 6, 13)}, 72, "latest", None, True),
    ("MOODENG", 0.02792, {1: R(25, 13, 17), 3: R(25, 13, 24), 4: R(25, 13, 27), 6: R(25, 13, 28)}, 6, "latest", None, False),
    ("AT", 0.0544, {1: R(25, 20, 10), 2: R(25, 20, 17), 3: R(25, 20, 28), 5: R(25, 20, 48), 6: R(25, 20, 50)}, 6, "latest", None, False),
    ("EVAA", 2.785, {1: R(26, 0, 54), 2: R(26, 9, 13), 3: R(26, 9, 20)}, 24, "earliest", None, False),
    ("ELSA", 0.09577, {}, 0, "latest", None, False),
]


def get(url):
    for r in range(4):
        try:
            return S.get(url)["data"]
        except Exception:
            time.sleep(1.5 + r)
    raise SystemExit("fetch failed " + url)


def kl(sym, itv, a, b, step):
    out = {k: [] for k in ("time", "high", "low", "close")}
    while a < b:
        e = min(a + step * 1000, b)
        d = get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={e}")
        for k in out:
            out[k] += d[k]
        a = e
    return out


rows = []
for name, entry, claims, win, mode, fixed, inferred in SIG:
    sym = name + "_USDT"
    stop = entry * 0.95
    tps = [entry * (1 + x / 100) for x in LAD]
    first_claim = min(claims.values()) if claims else None
    hi = int(now // 3600) * 3600
    if first_claim:
        a, b = first_claim - win * 3600, first_claim
    else:
        a, b = R(26, 9, 20), hi
    h1 = kl(sym, "Min60", a - 3600, b + 3600, 3600)
    cand = [(h1["time"][i] + 3600, abs(h1["close"][i] / entry - 1)) for i in range(len(h1["time"])) if a <= h1["time"][i] + 3600 <= b]
    if not cand:
        print(f"== {name}: لا شموع"); continue
    m_all = kl(sym, "Min1", a - 3600, now, 60)
    fc = first_claim or now

    def stop_before(t0, tend):
        return any(m_all["low"][i] <= stop for i in range(len(m_all["time"])) if t0 <= m_all["time"][i] < tend)

    near = [c for c in cand if c[1] <= 0.01 and not stop_before(c[0], fc)]
    note = ""
    if near:
        t0 = (min(near) if mode == "earliest" else max(near))[0]
    else:
        t0 = min(cand, key=lambda c: c[1])[0]
        note = " (لا إغلاق قريب من الدخول، أقرب إغلاق)"
    m = {k: [v for tt, v in zip(m_all["time"], m_all[k]) if tt >= t0 - 60] for k in m_all}
    idx = [i for i in range(len(m["time"])) if m["time"][i] >= t0]
    h4 = kl(sym, "Hour4", t0 - 4 * 3600, now, 14400)
    # stop by wick
    wick = next((m["time"][i] for i in idx if m["low"][i] <= stop), None)
    # stop by closed 4H candle under stop (candle closes at time+14400 <= now)
    c4 = next((h4["time"][i] + 14400 for i in range(len(h4["time"])) if h4["time"][i] + 14400 > t0 and h4["time"][i] + 14400 <= now and h4["close"][i] < stop), None)
    reach = {}
    for n, tp in enumerate(tps, 1):
        reach[n] = next((m["time"][i] for i in idx if m["high"][i] >= tp), None)
    lo = min(m["low"][i] for i in idx); mx = max(m["high"][i] for i in idx)
    # until the 4H-close stop (channel's rule), TPs counted only before it
    cut = c4 or now
    ok = {n: t for n, t in reach.items() if t is not None and t < cut}
    rows.append((name, entry, t0, len(ok), wick, c4, lo, mx))
    print(f"== {name}{' (الدخول مُستنتج من TP)' if inferred else ''} · دخول {entry:.6g} · وقف {stop:.6g} · بداية مفترضة {u4(t0)}{note}")
    print("   الأهداف الفعلية: " + " | ".join(f"TP{n} {u4(t)}" if (t := reach[n]) else f"TP{n} ✘" for n in range(1, 7)))
    for n, ct in claims.items():
        t = reach[n]
        diff = f"{(t - ct) / 60:+.0f}د" if t else "لم يتحقق"
        print(f"   مقارنة TP{n}: المعلن {u4(ct)} · الفعلي {u4(t) if t else '—'} · فرق {diff}")
    print(f"   أدنى سعر {(lo/entry-1)*100:+.2f}% · أعلى {(mx/entry-1)*100:+.2f}% · وقف بالفتيل: {u4(wick) if wick else 'لا'} · وقف بإغلاق 4س: {u4(c4) if c4 else 'لا'} · الآن {(m['close'][-1]/entry-1)*100:+.2f}%\n")
# MOODENG spot check (claimed TP4..TP6 at 13:27-13:28 Riyadh = 14:27-14:28 UTC+4, 25 Nov)
try:
    st = int(datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc).timestamp()) * 1000
    d = S.get(f"https://api.mexc.com/api/v3/klines?symbol=MOODENGUSDT&interval=1m&startTime={st}&endTime={st + 60 * 60 * 1000}&limit=100")
    hi = max(float(x[2]) for x in d)
    print(f"سبوت MOODENG بين 14:00 و15:00 (UTC+4) 25 نوفمبر: أعلى سعر {hi} ({(hi/0.02792-1)*100:+.2f}%) · هدف TP4 {0.02792*1.132:.5f} · TP6 {0.02792*1.2:.5f}")
except Exception as e:
    print("سبوت MOODENG: فشل الجلب", str(e)[:80])
print("== ملخص (الأهداف قبل إغلاق 4س تحت الوقف)")
done = [r for r in rows if r[0] != "ELSA"]
for name, entry, t0, k, wick, c4, lo, mx in rows:
    print(f"  {name:8s} أهداف {k}/6 · وقف فتيل {'نعم' if wick else 'لا'} · وقف 4س {'نعم' if c4 else 'لا'} · أدنى {(lo/entry-1)*100:+.1f}%")
print(f"  صفقات بلغت TP1 على الأقل: {sum(r[3] >= 1 for r in done)}/{len(done)} · ضُرب وقف الفتيل: {sum(1 for r in done if r[4])}/{len(done)} · وقف إغلاق 4س: {sum(1 for r in done if r[5])}/{len(done)}")
