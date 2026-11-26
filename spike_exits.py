"""Exit-method search on the volume-spike triggers (same as spike_test.py: A = 15m volume >=4x and 15m return >=+2%, next 5m open +0.1% slippage, 0.2% cost, 45 days, 86 coins).
Exits: fixed stop/TP, trailing stop, time exit with hard stop, breakeven then trail, half at +3% then trail. 48h horizon, stop checked before TP in the same candle.
Compare triggers vs random control (same coins), week-block bootstrap CI, halves."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scanner as S

now = int(time.time())
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX"}
DAYS = 45
COST = 0.2
rng = np.random.default_rng(2)
EXITS = [("fixed", (2, 2)), ("fixed", (3, 3)), ("fixed", (2, 4)), ("fixed", (3, 6)), ("fixed", (1.5, 3)),
         ("trail", (3, 3)), ("trail", (4, 5)), ("trail", (5, 8)), ("trail", (8, 8)),
         ("time", (3, 5)), ("time", (12, 5)), ("time", (48, 5)), ("time", (288, 5)),
         ("be", (4, 3, 3)), ("be", (5, 4, 5)), ("half", (4, 3, 4)), ("half", (5, 3, 6))]
NAME = {"fixed": "هدف {1}% / وقف {0}%", "trail": "وقف متحرك: بدء -{0}% ثم {1}% تحت القمة", "time": "خروج بعد {0} شمعة 5د (وقف {1}%)",
        "be": "تعادل بعد +{1}% ثم متحرك {2}% (وقف أول -{0}%)", "half": "نصف عند +{1}% والباقي متحرك {2}% (وقف أول -{0}%)"}


def fetch(sym):
    a, out = now - DAYS * 86400, {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
    while a < now:
        b = min(a + 300 * 1000, now)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min5&start={a}&end={b}")["data"]; break
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
    return {k: np.array([out[k][i] for i in keep], float) for k in out}


def sim(d, i, kind, p):
    H, L, C = d["high"], d["low"], d["close"]
    e = d["open"][i + 1] * 1.001
    end = i + 1 + 576
    mark = lambda k: (C[k] / e - 1) * 100
    if kind == "fixed":
        s, t = p
        for k in range(i + 1, end + 1):
            if L[k] <= e * (1 - s / 100): return -s
            if H[k] >= e * (1 + t / 100): return t
        return mark(end)
    if kind == "trail":
        s0, tr = p
        lvl, pk = e * (1 - s0 / 100), e
        for k in range(i + 1, end + 1):
            if L[k] <= lvl: return (lvl / e - 1) * 100
            pk = max(pk, H[k]); lvl = max(lvl, pk * (1 - tr / 100))
        return mark(end)
    if kind == "time":
        n, s = p
        for k in range(i + 1, i + 1 + n):
            if L[k] <= e * (1 - s / 100): return -s
        return mark(i + n)
    if kind == "be":
        s0, trg, tr = p
        lvl, pk, armed = e * (1 - s0 / 100), e, False
        for k in range(i + 1, end + 1):
            if L[k] <= lvl: return (lvl / e - 1) * 100
            pk = max(pk, H[k])
            if H[k] >= e * (1 + trg / 100): armed = True
            if armed: lvl = max(lvl, e * 1.005, pk * (1 - tr / 100))
        return mark(end)
    if kind == "half":
        s0, t1, tr = p
        lvl, pk, half = e * (1 - s0 / 100), e, False
        for k in range(i + 1, end + 1):
            if not half:
                if L[k] <= lvl: return -s0
                if H[k] >= e * (1 + t1 / 100): half = True; pk = H[k]; lvl = max(e, pk * (1 - tr / 100))
            else:
                if L[k] <= lvl: return (t1 + (lvl / e - 1) * 100) / 2
                pk = max(pk, H[k]); lvl = max(lvl, pk * (1 - tr / 100))
        return (t1 + mark(end)) / 2 if half else mark(end)


def work(sym):
    d = fetch(sym)
    if d is None or len(d["time"]) < 3000:
        return []
    c, v = d["close"], d["vol"]
    n = len(c)
    pts, last = [], -99
    for i in range(300, n - 600):
        base = np.median(v[i - 288:i - 2]) * 3 or 1e-9
        if (v[i] + v[i - 1] + v[i - 2]) / base >= 4 and (c[i] / c[i - 3] - 1) * 100 >= 2 and i - last >= 12:
            last = i; pts.append((i, "T"))
    pts += [(int(rng.integers(300, n - 600)), "C") for _ in range(3 * len(pts))]
    return [(sym, int(d["time"][i]), kind, [sim(d, i, k, p) - COST for k, p in EXITS]) for i, kind in pts]


detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
        and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
with ThreadPoolExecutor(6) as ex:
    res = [r for part in ex.map(work, tick) for r in part]
T = [r for r in res if r[2] == "T"]; Cn = [r for r in res if r[2] == "C"]
print(f"عملات {len(tick)} · إشارات {len(T)} · عشوائي {len(Cn)}\n")
ts = sorted(r[1] for r in T); mid = ts[len(ts) // 2]
wk = np.array([r[1] // (7 * 86400) for r in T]); wu = np.unique(wk)
rows = []
for j, (k, p) in enumerate(EXITS):
    a = np.array([r[3][j] for r in T]); b = np.array([r[3][j] for r in Cn])
    bs = [np.concatenate([a[wk == w] for w in rng.choice(wu, len(wu))]).mean() for _ in range(800)]
    h1 = np.mean([r[3][j] for r in T if r[1] < mid]); h2 = np.mean([r[3][j] for r in T if r[1] >= mid])
    rows.append((a.mean(), b.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), h1, h2, np.mean(a > 0) * 100, NAME[k].format(*p)))
print(f"{'طريقة الخروج':52s} {'إشارات':>7s} {'عشوائي':>7s} {'CI إشارات':>14s} {'نصف1':>6s} {'نصف2':>6s} {'موجب':>5s}")
for r in sorted(rows, reverse=True):
    print(f"{r[7]:52s} {r[0]:+6.2f}% {r[1]:+6.2f}% {r[2]:+6.2f}..{r[3]:+5.2f} {r[4]:+5.2f}% {r[5]:+5.2f}% {r[6]:4.0f}%")
