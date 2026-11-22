"""Where did the price bounce before rising to its peak - at a bullish order block (LuxAlgo-style internal OBs) or not?
Part A: each radar alert (matched to MEXC candles): bounce low = lowest 15m low between the alert and the peak; is that low inside/at a bullish OB on 15m / 1H / 4H?
Part B (control, all crypto coins): every 1H pivot low (lowest of +-6 candles): share inside a 1H/4H OB, and the rise in the next 24h/48h, OB vs non-OB.
Part C: 'first touch of a bullish 1H OB' strategy (limit at OB high, stop below OB low) vs the same stop distance at random bars.
Times UTC+4."""
import random
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import luxalgo_ob as LX
import scanner as S
import numpy as np

UTC3, UTC4 = timedelta(hours=3), timedelta(hours=4)
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + UTC4
now = int(time.time())
src = open("alert_batch.py", encoding="utf-8").read()
A = eval("[" + src.split("\nA = [", 1)[1].split("\n]\nUTC3", 1)[0] + "\n]")
day0 = datetime(2026, 10, 24, 0, 0, 0, tzinfo=timezone.utc)
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX", "USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}


def fetch(sym, itv, step, a, b_end):
    out = {k: [] for k in ("time", "open", "high", "low", "close")}
    while a < b_end:
        b = min(a + step * 1000, b_end)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            return None
        for k in out:
            out[k] += d[k]
        a = b
    seen, keep = set(), []
    for i, t in enumerate(out["time"]):
        if t not in seen:
            seen.add(t); keep.append(i)
    return {k: [out[k][i] for i in keep] for k in out}


class TF:
    def __init__(self, d, per):
        self.d, self.per = d, per
        _, self.snaps = LX.run(d["open"], d["high"], d["low"], d["close"])

    def idx(self, ts):
        t = self.d["time"]
        lo_, hi_ = 0, len(t)
        while lo_ < hi_:
            mid = (lo_ + hi_) // 2
            if t[mid] <= ts:
                lo_ = mid + 1
            else:
                hi_ = mid
        return lo_ - 1

    def hit(self, ts, price):
        """Bullish OB alive just before the bar that contains ts, that the price low touched/held (inside the zone, or a wick up to 1% under its low)."""
        i = self.idx(ts)
        if i < 2:
            return None
        best = None
        for ob in self.snaps[i - 1]:
            if ob["bias"] != 1:
                continue
            if ob["lo"] * 0.99 <= price <= ob["hi"] * 1.002:
                inside = ob["lo"] <= price <= ob["hi"]
                cand = (0 if inside else 1, ob)
                if best is None or cand[0] < best[0]:
                    best = cand
        return best

    def nearest_below(self, ts, price):
        i = self.idx(ts)
        c = [ob for ob in self.snaps[max(i - 1, 0)] if ob["bias"] == 1 and ob["hi"] < price]
        return min(((price / ob["hi"] - 1) * 100, ob) for ob in c)[0] if c else None


# ---------------- Part A ----------------
syms = sorted({a[1] for a in A})
def prep(sym):
    return sym, {"1h_match": fetch(sym, "Min60", 3600, int((day0 - timedelta(hours=6)).timestamp()), int((day0 + timedelta(days=5)).timestamp())),
                 "15m": fetch(sym, "Min15", 900, int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()), now),
                 "1h": fetch(sym, "Min60", 3600, int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()), now),
                 "4h": fetch(sym, "Hour4", 14400, int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()), now)}
with ThreadPoolExecutor(4) as ex:
    P = dict(ex.map(prep, syms))
TFS = {s: {"15m": TF(P[s]["15m"], 900), "1h": TF(P[s]["1h"], 3600), "4h": TF(P[s]["4h"], 14400)} for s in syms}
print("== الجزء أ: أين ارتد السعر قبل أن يصعد إلى قمته؟ (تنبيهات الرادار)")
print(f"{'العملة':8s} {'وقت التنبيه':14s} {'سعر التنبيه':>10s} | {'قاع الارتداد':>11s} {'متى':>13s} {'تحت سعر التنبيه':>14s} | {'القمة':>10s} {'من القاع للقمة':>13s} | أوردر بلوك صاعد عند القاع: 15د / 1س / 4س")
rowsA = []
for di, sym, close, aprice, hhmmss, prob, r1, r4, vol, drop in A:
    h = P[sym]["1h_match"]
    cands = []
    for i in range(4, len(h["time"])):
        cu = h["time"][i] + 3600
        if abs(h["close"][i] / close - 1) > 0.003:
            continue
        rc = datetime.fromtimestamp(cu, timezone.utc) + UTC3
        a1 = (h["close"][i] / h["open"][i] - 1) * 100
        a4 = (h["close"][i] / h["open"][i - 3] - 1) * 100
        sc = abs(a1 - r1) + abs(a4 - r4) if r1 is not None else 0
        if r1 is not None and sc > 3:
            continue
        cands.append((abs((rc.date() - (day0 + timedelta(days=di)).date()).days), sc, cu))
    if not cands:
        continue
    cands.sort()
    hh, mm, ss = (int(x) for x in hhmmss.split(":"))
    a0 = int(datetime.fromtimestamp(cands[0][2], timezone.utc).replace(minute=mm, second=ss).timestamp())
    m = P[sym]["15m"]
    idx = [i for i in range(len(m["time"])) if m["time"][i] >= (a0 // 900 + 1) * 900]
    pk = max(idx, key=lambda i: m["high"][i])
    seg = [i for i in idx if i <= pk]
    lo_i = min(seg, key=lambda i: m["low"][i])
    lo_p, lo_t, pk_p = m["low"][lo_i], m["time"][lo_i], m["high"][pk]
    hits = {}
    for tf in ("15m", "1h", "4h"):
        T = TFS[sym][tf]
        hx = T.hit(lo_t, lo_p)
        hits[tf] = ("داخل" if hx[0] == 0 else "ذيل تحته") if hx else ("لا")
    rowsA.append({"sym": sym, "hits": hits, "below": (lo_p / aprice - 1) * 100, "rise": (pk_p / lo_p - 1) * 100, "alert_rise": (pk_p / aprice - 1) * 100, "first_bar": lo_i == seg[0]})
    print(f"{sym[:-5]:8s} {dt4(a0):%d %b %H:%M}  {aprice:10.6g} | {lo_p:11.6g} {dt4(lo_t):%d %b %H:%M} {(lo_p/aprice-1)*100:+13.1f}% | {pk_p:10.6g} {(pk_p/lo_p-1)*100:+12.0f}% | {hits['15m']:8s} / {hits['1h']:8s} / {hits['4h']}")
okA = len(rowsA)
for tf in ("15m", "1h", "4h"):
    n = sum(1 for r in rowsA if r["hits"][tf] != "لا")
    print(f"  عند قاع الارتداد كان داخل/عند أوردر بلوك صاعد على {tf}: {n} من {okA} ({n/okA*100:.0f}%)")
anyob = sum(1 for r in rowsA if any(v != "لا" for v in r["hits"].values()))
print(f"  على أي فريم من الثلاثة: {anyob} من {okA} ({anyob/okA*100:.0f}%) · وسيط البعد عن سعر التنبيه: {st.median(r['below'] for r in rowsA):+.1f}% (أي القاع أرخص من سعر التنبيه بهذا القدر) · وسيط الارتفاع من القاع للقمة {st.median(r['rise'] for r in rowsA):+.0f}% مقابل {st.median(r['alert_rise'] for r in rowsA):+.0f}% من سعر التنبيه")

# ---------------- Part B & C ----------------
detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
        and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]


def per_coin(sym):
    h1 = fetch(sym, "Min60", 3600, int(datetime(2026, 8, 20, tzinfo=timezone.utc).timestamp()), now)
    h4 = fetch(sym, "Hour4", 14400, int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()), now)
    if not h1 or not h4 or len(h1["time"]) < 600:
        return [], []
    T1, T4 = TF(h1, 3600), TF(h4, 14400)
    t, o, hi, lo, c = h1["time"], h1["open"], h1["high"], h1["low"], h1["close"]
    n = len(c)
    piv, trades = [], []
    rnd = random.Random(hash(sym) % 10000)
    used = set()
    for i in range(400, n - 49):
        if lo[i] == min(lo[i - 6:i + 7]) if i + 7 <= n else False:
            h = T1.hit(t[i], lo[i]); h4h = T4.hit(t[i], lo[i])
            piv.append({"in1": h is not None, "in4": h4h is not None, "r24": (max(hi[i + 1:i + 25]) / lo[i] - 1) * 100, "r48": (max(hi[i + 1:i + 49]) / lo[i] - 1) * 100})
        # first touch of a bullish 1H OB: limit at OB high, stop under OB low
        for ob in T1.snaps[i - 1]:
            if ob["bias"] != 1 or (sym, ob["t"], ob["created"]) in used:
                continue
            if lo[i] <= ob["hi"] and c[i - 1] > ob["hi"]:
                used.add((sym, ob["t"], ob["created"]))
                ent, stp = ob["hi"], ob["lo"] * 0.997
                risk = ent - stp
                if risk <= 0 or risk / ent > 0.15 or risk / ent < 0.003:
                    continue
                res = None
                for j in range(i, min(i + 49, n)):
                    if lo[j] <= stp:
                        res = (-risk / ent * 100 - 0.2, -1.0); break
                    if hi[j] >= ent * 1.10:
                        res = (10 - 0.2, (ent * 0.10) / risk); break
                if res is None:
                    px = c[min(i + 48, n - 1)]
                    res = ((px / ent - 1) * 100 - 0.2, (px - ent) / risk)
                # control: same % stop distance, random bar entry at its close
                rb = rnd.randrange(400, n - 49)
                e2 = c[rb]; s2 = e2 * (1 - risk / ent); r2 = None
                for j in range(rb + 1, rb + 49):
                    if lo[j] <= s2:
                        r2 = (-risk / ent * 100 - 0.2, -1.0); break
                    if hi[j] >= e2 * 1.10:
                        r2 = (10 - 0.2, (e2 * 0.10) / (e2 - s2)); break
                if r2 is None:
                    r2 = ((c[rb + 48] / e2 - 1) * 100 - 0.2, (c[rb + 48] - e2) / (e2 - s2))
                trades.append({"ob": res, "ctl": r2, "risk": risk / ent * 100})
    return piv, trades


with ThreadPoolExecutor(4) as ex:
    out = list(ex.map(per_coin, tick))
piv = [p for a, _ in out for p in a]
trd = [x for _, b in out for x in b]
print(f"\n== الجزء ب: كل قيعان الساعة (قاع هو أدنى سعر بين ±6 شموع) في {len(tick)} عملة: {len(piv)} قاعاً")
def grp(nm, g):
    if not g:
        return
    print(f"  {nm}: {len(g)} قاع · وسيط الصعود خلال 24س {st.median(x['r24'] for x in g):+.1f}% · خلال 48س {st.median(x['r48'] for x in g):+.1f}% · وصل +10% خلال 48س: {np.mean([x['r48']>=10 for x in g])*100:.0f}% · +20%: {np.mean([x['r48']>=20 for x in g])*100:.0f}% · +50%: {np.mean([x['r48']>=50 for x in g])*100:.1f}%")
grp("كل القيعان", piv)
grp("القاع داخل/عند أوردر بلوك صاعد على 1س", [x for x in piv if x["in1"]])
grp("القاع ليس عند أوردر بلوك 1س", [x for x in piv if not x["in1"]])
grp("القاع عند أوردر بلوك 4س", [x for x in piv if x["in4"]])
grp("القاع ليس عند أوردر بلوك 4س", [x for x in piv if not x["in4"]])
grp("عند أوردر بلوك 1س و4س معاً", [x for x in piv if x["in1"] and x["in4"]])
print(f"  نسبة القيعان التي كانت عند أوردر بلوك 1س: {np.mean([x['in1'] for x in piv])*100:.0f}% · 4س: {np.mean([x['in4'] for x in piv])*100:.0f}%")
print(f"\n== الجزء ج: دخول عند أول لمس لأوردر بلوك صاعد على الساعة (أمر معلّق عند أعلى الأوردر بلوك، وقف تحت أدناه، هدف +10%، حد أقصى 48س، تكلفة 0.2%): {len(trd)} صفقة")
def stat(nm, k):
    r = [x[k] for x in trd]
    pct = np.array([a for a, _ in r]); rr = np.array([b for _, b in r])
    print(f"  {nm}: رابحة بالهدف {np.mean(pct >= 9.5)*100:.0f}% · ضُرب الوقف {np.mean(rr <= -0.99)*100:.0f}% · متوسط الصفقة {pct.mean():+.2f}% · متوسط R {rr.mean():+.3f} · وسيط {np.median(pct):+.2f}%")
stat("دخول عند أوردر بلوك صاعد", "ob")
stat("ضابط: دخول عشوائي بنفس مسافة الوقف", "ctl")
print(f"  وسيط مسافة الوقف: {st.median(x['risk'] for x in trd):.1f}%")
