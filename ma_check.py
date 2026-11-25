"""Does the 'MA5/10/30/60' picture show up in the three channel signals at signal time? Simple moving averages of the close on 15m / 1H / 4H (data cut at the signal), alignment, price vs each,
hours since MA5 crossed MA10 and MA10 crossed MA30, and the base rate of the same alignment on all 1H candles of the coin. Times UTC+4."""
import time
from datetime import datetime, timedelta, timezone
import scanner as S

U = lambda y, mo, d, h: int((datetime(y, mo, d, h, tzinfo=timezone.utc) - timedelta(hours=4)).timestamp())
SIG = [("AT_USDT", U(2026, 11, 25, 21), 0.0544), ("MOODENG_USDT", U(2026, 11, 25, 21), 0.02792), ("EVAA_USDT", U(2026, 11, 25, 22), 2.785)]
now = int(time.time())
SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400}


def kl(sym, itv, days, cut=None):
    out = {"time": [], "close": [], "high": [], "low": []}
    a = now - days * 86400
    while a < now:
        b = min(a + SEC[itv] * 1000, now)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            raise SystemExit("fetch failed")
        for k in out:
            out[k] += d[k]
        a = b
    seen, keep = set(), []
    for i, t in enumerate(out["time"]):
        if t not in seen and t + SEC[itv] <= (cut or now):
            seen.add(t); keep.append(i)
    return {k: [out[k][i] for i in keep] for k in out}


def sma(x, n):
    return [None] * (n - 1) + [sum(x[i - n + 1:i + 1]) / n for i in range(n - 1, len(x))]


def state(cl, i):
    m = {n: sma(cl, n) for n in (5, 10, 30, 60)}
    if any(m[n][i] is None for n in m):
        return None
    return m


def aligned(cl, i, m):
    return cl[i] > m[5][i] > m[10][i] > m[30][i] > m[60][i]


print("المتوسطات البسيطة (5/10/30/60) لحظة الإشارة:\n")
for sym, t, entry in SIG:
    print(f"== {sym[:-5]} · دخول {entry}")
    for nm, k in (("15 دقيقة", "Min15"), ("ساعة", "Min60"), ("4 ساعات", "Hour4")):
        d = kl(sym, k, 40 if k != "Hour4" else 120, t)
        cl = d["close"]
        i = len(cl) - 1
        m = state(cl, i)
        if not m:
            print(f"  {nm}: بيانات قليلة"); continue
        order = sorted([(m[n][i], f"MA{n}") for n in m] + [(cl[i], "السعر")], reverse=True)
        c510 = next((j for j in range(i, 0, -1) if m[5][j] is not None and m[10][j] is not None and (m[5][j - 1] <= m[10][j - 1]) and m[5][j] > m[10][j]), None)
        c1030 = next((j for j in range(i, 0, -1) if m[30][j - 1] is not None and (m[10][j - 1] <= m[30][j - 1]) and m[10][j] > m[30][j]), None)
        hrs = lambda j, per=SEC[k] / 3600: f"{(i - j) * per:.0f} ساعة" if j is not None else "—"
        print(f"  {nm}: الترتيب (من الأعلى): {' > '.join(f'{lab} {v:.5g}' for v, lab in order)}")
        print(f"     MA5 {'فوق' if m[5][i] > m[10][i] else 'تحت'} MA10 · MA10 {'فوق' if m[10][i] > m[30][i] else 'تحت'} MA30 · MA30 {'فوق' if m[30][i] > m[60][i] else 'تحت'} MA60 · السعر فوق MA5/10/30/60: {'/'.join('✔' if cl[i] > m[n][i] else '✘' for n in (5, 10, 30, 60))} · ترتيب صاعد كامل: {'نعم' if aligned(cl, i, m) else 'لا'} · منذ تقاطع MA5 فوق MA10: {hrs(c510)} · منذ MA10 فوق MA30: {hrs(c1030)}")
    h1 = kl(sym, "Min60", 60)
    cl = h1["close"]; m = {n: sma(cl, n) for n in (5, 10, 30, 60)}
    n_al = sum(1 for i in range(60, len(cl)) if aligned(cl, i, {k: m[k] for k in m}))
    print(f"  نسبة الشموع (ساعة، آخر 60 يوماً) التي فيها ترتيب كامل صاعد: {n_al/(len(cl)-60)*100:.0f}% من {len(cl)-60} شمعة\n")
