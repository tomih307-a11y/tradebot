"""What do the channel signals have in common? Same measurements at each signal time (data cut at the signal): last-candle move & volume, 4h/24h/7d returns, range position, PMO/MACD/TEMA per frame,
hourly ATR vs the stop distance, order block containing the entry, breakout check, BTC context. Times UTC+4."""
import statistics as st
import time
from datetime import datetime, timedelta, timezone
import luxalgo_ob as LX
import scanner as S

U = lambda y, mo, d, h: int((datetime(y, mo, d, h, tzinfo=timezone.utc) - timedelta(hours=4)).timestamp())
SIG = [("AT_USDT", U(2026, 11, 25, 21), 0.0544, 0.0517), ("MOODENG_USDT", U(2026, 11, 25, 21), 0.02792, 0.02652), ("EVAA_USDT", U(2026, 11, 25, 22), 2.785, 2.646), ("ELSA_USDT", U(2026, 11, 26, 13), 0.09577, 0.09098)]
now = int(time.time())
SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)


def kl(sym, itv, days, cut):
    out = {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
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
        if t not in seen and t + SEC[itv] <= cut:
            seen.add(t); keep.append(i)
    return {k: [out[k][i] for i in keep] for k in out}


def frame_txt(d):
    cl = d["close"]
    if len(cl) < 200:
        return "—"
    p, ps = S.pmo(cl)
    m = [a - b for a, b in zip(S.ema(cl, 12), S.ema(cl, 26))]
    g = S.ema(m, 9)
    te = S.tema(cl, 50)
    return f"PMO {p[-1]:+.1f}{'↑' if p[-1] > p[-2] else '↓'}{'>' if p[-1] > ps[-1] else '<'}برتقالي · مكدي {'أخضر' if m[-1] > g[-1] else 'أحمر'} · {(cl[-1]/te[-1]-1)*100:+.0f}% عن TEMA50"


rows = {}
for sym, t, entry, stop in SIG:
    h1, h4, d1, m15 = kl(sym, "Min60", 40, t), kl(sym, "Hour4", 120, t), kl(sym, "Day1", 400, t), kl(sym, "Min15", 20, t)
    btc = kl("BTC_USDT", "Min60", 10, t)
    c = h1["close"]
    vmed = st.median(h1["vol"][-24 * 7:-1])
    rng_h = [(h1["high"][i] / h1["low"][i] - 1) * 100 for i in range(len(c) - 24, len(c))]
    r = {}
    r["الشمعة الأخيرة (ساعة)"] = f"{(c[-1]/h1['open'][-1]-1)*100:+.1f}% · حجم ×{h1['vol'][-1]/vmed:.1f} · مدى {rng_h[-1]:.1f}%"
    r["قاع الشمعة الأخيرة → الدخول"] = f"{(entry/h1['low'][-1]-1)*100:+.1f}%"
    r["عائد 4 ساعات / 24 ساعة / 7 أيام"] = f"{(c[-1]/c[-5]-1)*100:+.1f}% / {(c[-1]/c[-25]-1)*100:+.1f}% / {(c[-1]/c[-169]-1)*100:+.1f}%"
    hh7, ll7 = max(h1["high"][-168:]), min(h1["low"][-168:])
    r["الدخول عن قمة 7 أيام / فوق قاعها"] = f"{(entry/hh7-1)*100:+.0f}% / {(entry/ll7-1)*100:+.0f}%"
    hh24, ll24 = max(h1["high"][-24:]), min(h1["low"][-24:])
    r["موقع الدخول من نطاق 24 ساعة"] = f"{(entry-ll24)/(hh24-ll24)*100:.0f}%"
    r["اختراق أعلى 24س السابقة؟"] = "نعم" if c[-1] > max(h1["high"][-25:-1]) else "لا"
    r["وسيط مدى الشمعة (24 ساعة) vs الوقف"] = f"{st.median(rng_h):.1f}% vs {(1-stop/entry)*100:.1f}%"
    r["PMO 15د"] = frame_txt(m15)
    r["PMO ساعة"] = frame_txt(h1)
    r["PMO 4س"] = frame_txt(h4)
    r["PMO يومي"] = frame_txt(d1)
    if len(h1["close"]) > 250:
        _, sn = LX.run(h1["open"], h1["high"], h1["low"], h1["close"])
        r["أوردر بلوك ساعة يحوي الدخول"] = next((f"{'صاعد' if ob['bias']==1 else 'هابط'} [{ob['lo']:.5g}–{ob['hi']:.5g}]" for ob in sn[-1] if ob["lo"] <= entry <= ob["hi"]), "لا")
    bc = btc["close"]
    r["BTC عائد 4س / 24س قبلها"] = f"{(bc[-1]/bc[-5]-1)*100:+.1f}% / {(bc[-1]/bc[-25]-1)*100:+.1f}%"
    rows[sym] = r
keys = list(next(iter(rows.values())).keys())
for k in keys:
    print(f"\n{k}:")
    for sym in rows:
        print(f"  {sym[:-5]:9s} {rows[sym].get(k, '—')}")
