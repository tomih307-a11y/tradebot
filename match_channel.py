"""How much would the spike rule match the channel? For each channel signal (coin + post time) look for a trigger on the same coin (15m volume >=4x the usual median of the last 24h, 15m return >=+2%) in a window around the post.
Oct signals: [post-60min, post+30min]; Nov signals (hourly-entry time only): [t-30min, t+90min]. Also loose variants (>=3x & >=1.5%) and the lead time of the trigger before the post. Futures 5m candles."""
import time
from datetime import datetime, timezone
import numpy as np
import scanner as S

now = int(time.time())
P = lambda mo, d, h, mi, off=3: int(datetime(2026, mo, d, h, mi, tzinfo=timezone.utc).timestamp()) - off * 3600
SIG = [("BTW", P(10, 14, 15, 58), 1), ("MITO", P(10, 14, 16, 11), 1), ("BTW", P(10, 16, 3, 12), 1), ("STBL", P(10, 16, 3, 19), 1), ("ZRO", P(10, 16, 3, 21), 1),
       ("WLD", P(10, 16, 11, 42), 1), ("ZK", P(10, 16, 11, 43), 1), ("AKE", P(10, 16, 11, 54), 1), ("TIA", P(10, 16, 12, 19), 1), ("SIREN", P(10, 17, 1, 11), 1),
       ("WLD", P(10, 17, 1, 14), 1), ("DNA", P(10, 17, 2, 3), 1), ("Q", P(10, 17, 3, 14), 1),
       ("MOODENG", P(11, 25, 14, 0, 4), 0), ("AT", P(11, 25, 21, 0, 4), 0), ("EVAA", P(11, 25, 22, 0, 4), 0), ("ELSA", P(11, 26, 13, 0, 4), 0)]


def fetch(sym, a, b):
    out = {k: [] for k in ("time", "close", "vol")}
    while a < b:
        e = min(a + 300 * 1000, b)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min5&start={a}&end={e}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d:
            for k in out:
                out[k] += d[k]
        a = e
    seen, keep = set(), []
    for i, t in enumerate(out["time"]):
        if t not in seen:
            seen.add(t); keep.append(i)
    return {k: np.array([out[k][i] for i in keep], float) for k in out}


def trig(d, rt, rr):
    T, C, V = d["time"], d["close"], d["vol"]
    res = []
    for i in range(290, len(T)):
        base = np.median(V[i - 288:i - 2]) * 3 or 1e-9
        if (V[i] + V[i - 1] + V[i - 2]) / base >= rt and (C[i] / C[i - 3] - 1) * 100 >= rr:
            res.append(int(T[i]) + 300)
    return res


hit = {"صارم (≥4× وعائد ≥2%)": 0, "مرن (≥3× وعائد ≥1.5%)": 0}
n = 0
for name, t, oct_ in SIG:
    d = fetch(name + "_USDT", t - 2 * 86400, t + 4 * 3600)
    if len(d["time"]) < 400:
        print(f"{name}: بيانات غير كافية"); continue
    n += 1
    lo, hi = (t - 3600, t + 1800) if oct_ else (t - 1800, t + 5400)
    line = f"{name} {datetime.fromtimestamp(t + 4*3600, timezone.utc):%d %H:%M}:"
    for lab, (rt, rr) in (("صارم (≥4× وعائد ≥2%)", (4, 2)), ("مرن (≥3× وعائد ≥1.5%)", (3, 1.5))):
        ts = [x for x in trig(d, rt, rr) if lo <= x <= hi]
        if ts:
            hit[lab] += 1
            line += f" {lab.split()[0]}: نعم (أقرب إشارة قبل/بعد النشر {(min(ts, key=lambda x: abs(x - t)) - t) / 60:+.0f}د)"
        else:
            line += f" {lab.split()[0]}: لا"
    print(line)
print(f"\n== توصيات القناة التي كانت قاعدتي ستلتقطها (من {n})")
for lab, k in hit.items():
    print(f"  {lab}: {k}/{n} = {k/n*100:.0f}%")
