"""What if the stop had been placed at the entry price? Re-simulates the tracked virtual trades with 15m data.
A) stop = entry from the start. B) original stop, moved to entry once price reached +1R. C) actual rule (original stop). Net of 0.1% costs. Output in % and R of the original risk."""
import json
import time
import scanner as S
import scanner_pmo as P

state = json.load(open("tracker.json"))
now = int(time.time())
print(f"{'العملة':10s} {'المجموعة':8s} | {'فعلي (وقف أصلي)':>22s} | {'وقف عند سعر الدخول':>22s} | {'وقف ينتقل لسعر الدخول بعد +1R':>30s}")
tot = {"C": 0.0, "A": 0.0, "B": 0.0}
totp = {"C": 0.0, "A": 0.0, "B": 0.0}
for tr in state:
    d = None
    for _ in range(5):
        d = P.kl(tr["sym"], "Min15", now)
        if d:
            break
        time.sleep(5)
    if not d:
        print("تعذّر جلب بيانات", tr["sym"]); continue
    t, hi, lo, c = d["time"], d["high"], d["low"], d["close"]
    p, ps = S.pmo(c)
    ent, stop = tr["entry"], tr["stop"]
    risk = ent - stop
    cost_pct = 0.1
    res = {}
    for name in ("C", "A", "B"):
        cur_stop = ent if name == "A" else stop
        moved = False
        out = None
        for j in range(1, len(c)):
            if t[j] < tr["detect_t"]:
                continue
            age = (t[j] + 900 - tr["detect_t"]) / 3600
            if lo[j] <= cur_stop:
                out = (cur_stop, "وقف"); break
            if p[j] < ps[j] and p[j - 1] >= ps[j - 1]:
                out = (c[j], "إشارة خروج"); break
            if age >= 24:
                out = (c[j], "24 ساعة"); break
            if name == "B" and not moved and hi[j] >= ent + risk:
                cur_stop = ent; moved = True
        if out is None:
            out = (c[-1], "مفتوحة")
        pct = (out[0] / ent - 1) * 100 - cost_pct
        res[name] = (pct, pct / (risk / ent * 100), out[1])
    for k in res:
        tot[k] += res[k][1]; totp[k] += res[k][0]
    f = lambda r: f"{r[0]:+.2f}% / {r[1]:+.2f}R ({r[2]})"
    print(f"{tr['sym'][:-5]:10s} {tr['group']:8s} | {f(res['C']):>22s} | {f(res['A']):>22s} | {f(res['B']):>30s}")
print(f"\nالمجموع: فعلي {totp['C']:+.2f}% / {tot['C']:+.2f}R | وقف عند الدخول {totp['A']:+.2f}% / {tot['A']:+.2f}R | وقف ينتقل {totp['B']:+.2f}% / {tot['B']:+.2f}R  (عدد الصفقات {len(state)})")
