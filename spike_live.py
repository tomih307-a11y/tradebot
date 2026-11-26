"""Live paper tracker of the 'volume spike' rule. Trigger on a closed 5m candle: 15m volume >= 4x the usual (median of the last 24h) AND 15m return >= +2%; cooldown 1h per coin.
Entry = next 5m open +0.1% slippage; exit after 4 hours at the close (-0.1% slippage) or at the -5% stop; costs: fees 0.2% + actual MEXC funding between entry and exit.
State in spike.json (restored from the spike-results branch). Each run scans closed candles since the last scan (so gaps in the schedule lose nothing) and settles open trades."""
import json
import os
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import scanner as S

now = int(time.time())
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX"}
st = json.load(open("spike.json")) if os.path.exists("spike.json") else {"last_scan": now - 2 * 3600, "trades": []}
dt4 = lambda x: (datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)).strftime("%d %H:%M")


def get(url):
    for r in range(4):
        try:
            return S.get(url)["data"]
        except Exception:
            time.sleep(1.5 + r)
    return None


def kl(sym, a, b):
    d = get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min5&start={a}&end={b}")
    if not d:
        return None
    seen, keep = set(), []
    for i, t in enumerate(d["time"]):
        if t not in seen:
            seen.add(t); keep.append(i)
    return {k: np.array([d[k][i] for i in keep], float) for k in ("time", "open", "high", "low", "close", "vol")}


def fund(sym, t1, t2):
    d = get(f"https://contract.mexc.com/api/v1/contract/funding_rate/history?symbol={sym}&page_num=1&page_size=30")
    tot, s = 0.0, (t1 // 28800 + 1) * 28800
    rec = {int(x["settleTime"]) // 1000: float(x["fundingRate"]) for x in (d or {}).get("resultList", [])}
    while s <= t2:
        k = min(rec, key=lambda z: abs(z - s)) if rec else None
        if k is not None and abs(k - s) < 1800:
            tot += rec[k] * 100
        s += 28800
    return tot


detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
        and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
last_scan = st["last_scan"]
new = 0
for sym in tick:
    d = kl(sym, now - 27 * 3600, now)
    if d is None or len(d["time"]) < 300:
        continue
    T, O, H, L, C, V = (d[k] for k in ("time", "open", "high", "low", "close", "vol"))
    n = len(T)
    prev = [t["trig_t"] for t in st["trades"] if t["sym"] == sym]
    for i in range(290, n - 1):
        if T[i] + 300 > now or T[i] + 300 <= last_scan:
            continue
        base = np.median(V[i - 288:i - 2]) * 3 or 1e-9
        ratio = (V[i] + V[i - 1] + V[i - 2]) / base
        ret = (C[i] / C[i - 3] - 1) * 100
        if ratio >= 4 and ret >= 2 and not any(0 <= T[i] - p < 3600 for p in prev):
            e = float(O[i + 1]) * 1.001
            st["trades"].append({"sym": sym, "trig_t": int(T[i]), "ratio": round(float(ratio), 1), "ret15": round(float(ret), 2), "entry_t": int(T[i + 1]), "entry": e, "status": "open"})
            prev.append(int(T[i])); new += 1
st["last_scan"] = now
# settle open trades
for tr in st["trades"]:
    if tr["status"] != "open":
        continue
    d = kl(tr["sym"], tr["entry_t"] - 300, now)
    if d is None:
        continue
    T, H, L, C = d["time"], d["high"], d["low"], d["close"]
    e, sl = tr["entry"], tr["entry"] * 0.95
    exit_t = tr["entry_t"] + 4 * 3600
    done = None
    for k in range(len(T)):
        if T[k] < tr["entry_t"]:
            continue
        if L[k] <= sl:
            done = (sl, int(T[k]), "وقف"); break
        if T[k] + 300 >= exit_t and T[k] + 300 <= now:
            done = (float(C[k]), int(T[k]) + 300, "4 ساعات"); break
    if done:
        px, tx, why = done
        f = fund(tr["sym"], tr["entry_t"], tx)
        tr.update(status="closed", exit=px, exit_t=tx, why=why, funding=round(f, 4), pct=round((px * 0.999 / e - 1) * 100 - 0.2 - f, 3))
    else:
        tr["now_pct"] = round((float(C[-1]) / e - 1) * 100, 2)
json.dump(st, open("spike.json", "w"))
cl = [t for t in st["trades"] if t["status"] == "closed"]
op = [t for t in st["trades"] if t["status"] == "open"]
with open("spike.txt", "w") as o:
    o.write(f"تحديث {dt4(now)} (UTC+4) · عملات تحت الرصد {len(tick)} · إشارات جديدة هذه الجولة {new}\n")
    o.write(f"إجمالي الصفقات {len(st['trades'])}: مغلقة {len(cl)} · مفتوحة {len(op)} · أول إشارة {dt4(min(t['trig_t'] for t in st['trades'])) if st['trades'] else '—'}\n")
    if cl:
        p = np.array([t["pct"] for t in cl])
        o.write(f"المغلقة: متوسط الصفقة بعد الرسوم والتمويل {p.mean():+.2f}% · موجبة {np.mean(p > 0)*100:.0f}% · الوسيط {np.median(p):+.2f}% · مجموع {p.sum():+.1f}% · بالوقف {sum(t['why']=='وقف' for t in cl)} · متوسط التمويل {np.mean([t['funding'] for t in cl]):+.3f}%\n")
    o.write("\nآخر 25 صفقة:\n")
    for t in st["trades"][-25:][::-1]:
        res_ = f"{t['pct']:+.2f}% ({t['why']}, تمويل {t['funding']:+.3f}%)" if t["status"] == "closed" else f"مفتوحة {t.get('now_pct', 0):+.2f}%"
        o.write(f"  {t['sym'][:-5]:10s} إشارة {dt4(t['trig_t'])} · حجم ×{t['ratio']} · عائد 15د {t['ret15']:+.1f}% · دخول {t['entry']:.6g} · {res_}\n")
print(open("spike.txt").read())
