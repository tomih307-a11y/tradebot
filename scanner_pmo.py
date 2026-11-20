"""PMO-only scanner (35,20,10), MEXC USDT futures, crypto only, long only. No MACD/TEMA/order blocks. Does NOT recommend or claim profit
(70-day test: PMO zero-touch + daily PMO above zero was ~break-even, +0.017R with CI containing zero).
Rule shown: 15m PMO blue crosses zero UP; "matches" when the daily PMO is above zero. Also shows 1H/4H/5m PMO state, 'about to touch' list, exit status
(15m blue crossed below orange since the signal), stop 1.5xATR. Usage: python3 scanner_pmo.py [hours_back=6]  (times UTC+4)"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import scanner as S

TZ = 4
SEC = {"Min5": 300, "Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}
DAYS = {"Min5": 5, "Min15": 20, "Min60": 60, "Hour4": 200, "Day1": 250}
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "TQQQ", "SQQQ", "TSLL", "NVDL", "MSTU", "CONL", "SPCX", "NVIDIA", "US30", "TESLA", "BITO"}
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)


def kl(sym, itv, now):
    out = {k: [] for k in ("time", "high", "low", "close")}
    a = now - DAYS[itv] * 86400
    while a < now:
        b = min(a + SEC[itv] * 1000, now)
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
    seen, idx = set(), []
    for i, tt in enumerate(out["time"]):
        if tt not in seen and tt + SEC[itv] <= now:
            seen.add(tt); idx.append(i)
    return {k: [out[k][i] for i in idx] for k in out}


def fstate(d, per, tc, price):
    times, closes = d["time"], d["close"]
    lo_, hi_ = 0, len(times)
    while lo_ < hi_:
        mid = (lo_ + hi_) // 2
        if times[mid] + per <= tc:
            lo_ = mid + 1
        else:
            hi_ = mid
    series = closes[max(0, lo_ - 700):lo_] + [price]
    if len(series) < 120:
        return None
    p, s = S.pmo(series)
    n = len(series) - 1
    return {"p": p[n], "pos": p[n] > 0, "above": p[n] > s[n], "up": p[n] > p[n - 1]}


def fmt(f):
    if not f:
        return "؟"
    return f"PMO {f['p']:+.1f} ({'فوق' if f['pos'] else 'تحت'} الصفر) {'↑' if f['up'] else '↓'} {'فوق' if f['above'] else 'تحت'} البرتقالي"


def work(args):
    sym, hours = args
    now = int(time.time())
    try:
        k5, k15, k1, k4, kd = (kl(sym, x, now) for x in ("Min5", "Min15", "Min60", "Hour4", "Day1"))
    except Exception:
        return None
    if not all((k5, k15, k1, k4, kd)) or len(k15["close"]) < 500:
        return None
    t, hi, lo, c = k15["time"], k15["high"], k15["low"], k15["close"]
    n = len(c)
    p, ps = S.pmo(c)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    touch = None
    for i in range(n - 1, max(n - 1 - hours * 4, 300), -1):
        if p[i - 1] <= 0 < p[i]:
            touch = i; break
    sp = p[-1] - p[-2]
    soon = p[-1] < 0 and p[-1] > ps[-1] and sp > 0 and -p[-1] <= 3 * sp
    if touch is None and not soon:
        return None
    i = touch if touch is not None else n - 1
    ent = c[i]
    tc = t[i] + 900
    out = {"sym": sym, "t": t[i], "touch": touch is not None, "soon": soon and touch is None, "price": ent, "now": c[-1], "p15": p[i], "p15now": p[-1], "above_now": p[-1] > ps[-1],
           "stop": max(ent - 1.5 * atr[i], ent * 0.94), "h1": fstate(k1, 3600, tc, ent), "h4": fstate(k4, 14400, tc, ent), "d1": fstate(kd, 86400, tc, ent),
           "m5": fstate(k5, 300, now, c[-1])}
    out["stop"] = min(out["stop"], ent * 0.995)
    if touch is not None:
        out["exit"] = any(p[j] < ps[j] and p[j - 1] >= ps[j - 1] for j in range(i + 1, n))
        out["low_since"] = min(lo[i + 1:]) if i + 1 < n else ent
        out["hit_stop"] = out["low_since"] <= out["stop"]
    return out


def main():
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    now = int(time.time())
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    with ThreadPoolExecutor(5) as ex:
        res = [r for r in ex.map(work, [(x["symbol"], hours) for x in tick]) if r]
    chg = {x["symbol"]: float(x.get("riseFallRate") or 0) * 100 for x in tick}
    print(f"ماسح PMO فقط · {len(tick)} عقد كريبتو · {dt(now):%d %H:%M} (UTC+{TZ})")
    print("القاعدة: لمس PMO الأزرق للصفر صعوداً على 15 دقيقة. 'يطابق' = PMO اليومي فوق الصفر. ليست توصية؛ أفضل ما وجدناه في الاختبار كان تعادلاً.\n")

    def show(r):
        d1ok = r["d1"] and r["d1"]["pos"]
        head = f"== {r['sym'][:-5]}USDT.P · {dt(r['t']):%d %H:%M} · سعر الإشارة {r['price']:.6g} (الآن {r['now']:.6g}، {(r['now']/r['price']-1)*100:+.1f}%، 24س {chg.get(r['sym'],0):+.1f}%)"
        print(head)
        print(f"   15د الآن: PMO {r['p15now']:+.2f} {'فوق' if r['above_now'] else 'تحت'} البرتقالي · وقف للعرض {r['stop']:.6g} (بعد {(r['price']/r['stop']-1)*100:.1f}%)")
        print(f"   يومي: {fmt(r['d1'])}\n   4س: {fmt(r['h4'])}\n   1س: {fmt(r['h1'])}\n   5د الآن: {fmt(r['m5'])}")
        if r["touch"]:
            st = "⚠ ضُرب الوقف" if r["hit_stop"] else ("⚠ ظهرت إشارة الخروج (الأزرق قطع البرتقالي هبوطاً)" if r["exit"] else "لا إشارة خروج بعد")
            print(f"   حالة الصفقة لو دخلت عند الإشارة: {st}")

    A = sorted([r for r in res if r["touch"] and r["d1"] and r["d1"]["pos"]], key=lambda r: -r["t"])
    B = sorted([r for r in res if r["touch"] and not (r["d1"] and r["d1"]["pos"])], key=lambda r: -r["t"])
    C = sorted([r for r in res if r["soon"]], key=lambda r: -abs(r["p15now"]))
    print(f"######## لمس الصفر + اليومي فوق الصفر (يطابق القاعدة) — آخر {hours} ساعات: {len(A)}")
    for r in A: show(r)
    print(f"\n######## لمس الصفر لكن اليومي تحت الصفر (للعلم فقط): {len(B)}")
    for r in B: show(r)
    print(f"\n######## على وشك لمس الصفر الآن (PMO فوق البرتقالي، يرتفع، ≤3 شموع): {len(C)}")
    for r in C: show(r)


if __name__ == "__main__":
    main()
