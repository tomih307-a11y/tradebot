"""Verify the 27-28 Nov channel claims (Riyadh clock UTC+3 in the messages) on BOTH spot and futures 1m candles: EVAA #2 (1.3215), BANK (0.01648), BTW X (entry inferred 1.1973) and BTW Y (1.2296).
Start = earliest 1H close within 1% of the entry with no stop wick before the first claim (stop = -5%). Prints claimed vs first-cross time per TP on spot and futures, plus extremes."""
import time
from datetime import datetime, timedelta, timezone
import scanner as S

now = int(time.time())
R = lambda d, h, m: int(datetime(2026, 11, d, h, m, tzinfo=timezone.utc).timestamp()) - 3 * 3600
u4 = lambda x: (datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)).strftime("%d %H:%M") if x else "—"
SIG = [("EVAA #2", "EVAA", 1.3215, {1: (1.3611, R(27, 22, 55)), 2: (1.4061, R(28, 1, 11)), 3: (1.451, R(28, 1, 11)), 4: (1.4959, R(28, 6, 13)), 5: (1.5409, R(28, 8, 16)), 6: (1.5858, R(28, 8, 37))}),
       ("BANK", "BANK", 0.01648, {1: (0.01697, R(28, 0, 56))}),
       ("BTW X", "BTW", 1.1973, {4: (1.353, R(28, 9, 27)), 5: (1.395, R(28, 9, 52)), 6: (1.436, R(28, 12, 49))}),
       ("BTW Y", "BTW", 1.2296, {3: (1.3464, R(28, 9, 21)), 4: (1.3894, R(28, 9, 51)), 5: (1.4325, R(28, 12, 51)), 6: (1.4755, R(28, 14, 8))})]


def fut(sym, a, b):
    out = {k: [] for k in ("time", "high", "low", "close")}
    while a < b:
        e = min(a + 60000, b)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}_USDT?interval=Min1&start={a}&end={e}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d:
            for k in out:
                out[k] += d[k]
        a = e
    return out


def spot(sym, a, b):
    out = {k: [] for k in ("time", "high", "low", "close")}
    while a < b:
        e = min(a + 60000, b)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://api.mexc.com/api/v3/klines?symbol={sym}USDT&interval=1m&startTime={a*1000}&endTime={e*1000}&limit=1000"); break
            except Exception:
                time.sleep(1.5 + r)
        for x in d or []:
            out["time"].append(int(x[0]) // 1000); out["high"].append(float(x[2])); out["low"].append(float(x[3])); out["close"].append(float(x[4]))
        a = e
    return out


for name, sym, entry, claims in SIG:
    fc = min(c[1] for c in claims.values())
    a = fc - 36 * 3600
    print(f"== {name} · دخول {entry} · أول إعلان {u4(fc)}")
    for kind, fn in (("سبوت", spot), ("عقود", fut)):
        try:
            m = fn(sym, a, now)
        except Exception as e:
            print(f"  {kind}: فشل {str(e)[:60]}"); continue
        T = m["time"]
        if len(T) < 100:
            print(f"  {kind}: بيانات غير كافية"); continue
        stop = entry * 0.95
        # hourly closes from the 1m series
        hcl = {}
        for i, t in enumerate(T):
            hcl[t // 3600 * 3600] = i
        cands = []
        for h, i in sorted(hcl.items()):
            t_close = h + 3600
            if t_close <= fc and abs(m["close"][i] / entry - 1) <= 0.01 and not any(m["low"][j] <= stop for j in range(len(T)) if t_close <= T[j] < fc):
                cands.append(t_close)
        t0 = cands[0] if cands else None
        if t0 is None:
            near = min((abs(m["close"][i] / entry - 1), h + 3600) for h, i in hcl.items() if h + 3600 <= fc)
            t0 = near[1]; note = f" (لا إغلاق قريب بلا وقف؛ أقرب إغلاق، فرق {near[0]*100:.1f}%)"
        else:
            note = ""
        post = [i for i in range(len(T)) if T[i] >= t0]
        lo = min(m["low"][i] for i in post); hi = max(m["high"][i] for i in post)
        wick = next((T[i] for i in post if m["low"][i] <= stop), None)
        line = f"  {kind}: بداية {u4(t0)}{note} · أعلى {(hi/entry-1)*100:+.1f}% · أدنى {(lo/entry-1)*100:+.1f}% · وقف فتيل: {u4(wick) if wick else 'لا'}"
        print(line)
        for n, (px, ct) in claims.items():
            ft = next((T[i] for i in post if m["high"][i] >= px), None)
            print(f"     TP{n} ({px}): المعلن {u4(ct)} · الفعلي {u4(ft)}" + (f" ({(ft-ct)/60:+.0f}د)" if ft else ""))
    print()
