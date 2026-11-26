"""What is common at post time? 17 signals with known post times (13 Oct 14-17 posted Riyadh clock, plus MOODENG/AT/EVAA/ELSA Nov by their hourly entry).
Spot 5m candles, 4 days before each post. For each feature: percentile of the signal's value among all 5m moments of the same coin in the prior 3 days. Uniform = 50."""
import time
from datetime import datetime, timezone
import numpy as np
import scanner as S

now = int(time.time())
P = lambda mo, d, h, mi, off=3: int(datetime(2026, mo, d, h, mi, tzinfo=timezone.utc).timestamp()) - off * 3600
SIG = [("BTW", P(10, 14, 15, 58)), ("MITO", P(10, 14, 16, 11)), ("BTW", P(10, 16, 3, 12)), ("STBL", P(10, 16, 3, 19)), ("ZRO", P(10, 16, 3, 21)),
       ("WLD", P(10, 16, 11, 42)), ("ZK", P(10, 16, 11, 43)), ("AKE", P(10, 16, 11, 54)), ("TIA", P(10, 16, 12, 19)), ("SIREN", P(10, 17, 1, 11)),
       ("WLD", P(10, 17, 1, 14)), ("DNA", P(10, 17, 2, 3)), ("Q", P(10, 17, 3, 14)),
       ("MOODENG", P(11, 25, 14, 0, 4)), ("AT", P(11, 25, 21, 0, 4)), ("EVAA", P(11, 25, 22, 0, 4)), ("ELSA", P(11, 26, 13, 0, 4))]


def spot(sym, a, b):
    out = []
    while a < b:
        e = min(a + 300 * 1000, b)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://api.mexc.com/api/v3/klines?symbol={sym}&interval=5m&startTime={a*1000}&endTime={e*1000}&limit=1000"); break
            except Exception:
                time.sleep(1.5 + r)
        out += d or []
        a = e
    seen, rows = set(), []
    for x in out:
        if x[0] not in seen:
            seen.add(x[0]); rows.append(x)
    return np.array([[int(x[0]) // 1000, float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7])] for x in rows])


def feats(d, i):
    c, h, l, q = d[:, 4], d[:, 2], d[:, 3], d[:, 5]
    w = slice(i - 287, i + 1)
    base_q = np.median(q[i - 287:i - 2]) or 1e-9
    return {"عائد 15د": (c[i] / c[i - 3] - 1) * 100, "عائد ساعة": (c[i] / c[i - 12] - 1) * 100, "عائد 4س": (c[i] / c[i - 48] - 1) * 100, "عائد 24س": (c[i] / c[i - 287] - 1) * 100,
            "حجم آخر 15د ÷ المعتاد": q[i - 2:i + 1].sum() / (3 * base_q), "حجم آخر ساعة ÷ المعتاد": q[i - 11:i + 1].sum() / (12 * base_q),
            "موقع من نطاق 24س": (c[i] - l[w].min()) / ((h[w].max() - l[w].min()) or 1e-12), "بعد عن قمة 24س %": (c[i] / h[w].max() - 1) * 100,
            "فوق قاع 24س %": (c[i] / l[w].min() - 1) * 100, "مدى آخر ساعة %": (h[i - 11:i + 1].max() / l[i - 11:i + 1].min() - 1) * 100}


pcts = {}
for name, t in SIG:
    d = spot(name + "USDT", t - 4 * 86400, t + 600)
    if len(d) < 600:
        print(f"{name}: بيانات غير كافية"); continue
    idx = np.where(d[:, 0] < t)[0]
    i = idx[-1]
    cur = feats(d, i)
    base = [feats(d, j) for j in range(i - 864, i - 6, 3) if j >= 300]
    for k, v in cur.items():
        arr = np.array([b[k] for b in base])
        pcts.setdefault(k, []).append((arr < v).mean() * 100)
    print(f"{name} {datetime.fromtimestamp(t + 4*3600, timezone.utc):%d %H:%M}: " + " · ".join(f"{k} {v:.2f}" for k, v in list(cur.items())[:5]))
print("\n== النسبة المئوية المتوسطة لقيمة التوصية بين لحظات نفس العملة (50 = عادي)")
n = len(next(iter(pcts.values())))
for k, v in pcts.items():
    print(f"  {k:26s} متوسط {np.mean(v):5.1f} · فوق 80%: {sum(x > 80 for x in v)}/{n} · تحت 20%: {sum(x < 20 for x in v)}/{n}")
print(f"  (الخطأ المعتاد لمتوسط {n} توصيات ≈ ±{29/np.sqrt(n):.0f}، فالفرق عن 50 يُعتبر ذا معنى إذا زاد عن ≈ ±{2*29/np.sqrt(n):.0f})")
hrs = [(datetime.fromtimestamp(t, timezone.utc).hour + 3) % 24 for _, t in SIG]
print("  ساعات النشر (الرياض): " + ", ".join(map(str, sorted(hrs))))
