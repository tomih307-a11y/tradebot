"""Compact study of today's biggest 24h gainers (crypto, >=1M USDT volume): where the move started, first 15m PMO zero-touch, larger-frame state at the bottom."""
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import luxalgo_ob as LX
import scanner as S

TZ = 4
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)
SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}


def kl(sym, itv, days, now):
    out = {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
    a = now - days * 86400
    while a < now:
        b = min(a + SEC[itv] * 1000, now)
        d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={a}&end={b}")["data"]
        for k in out:
            out[k] += d[k]
        a = b
    seen, idx = set(), []
    for i, tt in enumerate(out["time"]):
        if tt not in seen and tt + SEC[itv] <= now:
            seen.add(tt); idx.append(i)
    return {k: [out[k][i] for i in idx] for k in out}


def st_at(d, per, tc):
    k = max((i for i in range(len(d["time"])) if d["time"][i] + per <= tc), default=None)
    if k is None or k < 60:
        return None
    c = d["close"][:k + 1]
    p, s = S.pmo(c)
    m = [a - b for a, b in zip(S.ema(c, 12), S.ema(c, 26))]
    g = S.ema(m, 9)
    return {"p": p[k], "pos": p[k] > 0, "up": p[k] > p[k - 1], "above": p[k] > s[k], "green": m[k] > g[k]}


def tag(x):
    if not x:
        return "؟"
    return f"{'+' if x['pos'] else '-'}{abs(x['p']):.1f} {'↑' if x['up'] else '↓'} {'مكدي أخضر' if x['green'] else 'مكدي أحمر'}"


def study(item):
    sym, chg, vol, fund = item
    now = int(time.time())
    d15, d1, d4, dd = kl(sym, "Min15", 14, now), kl(sym, "Min60", 40, now), kl(sym, "Hour4", 150, now), kl(sym, "Day1", 250, now)
    t, c, hi, lo, v, o = d15["time"], d15["close"], d15["high"], d15["low"], d15["vol"], d15["open"]
    n = len(t)
    pk = max(range(n - 288, n), key=lambda i: hi[i])
    bt = min(range(max(pk - 192, 0), pk + 1), key=lambda i: lo[i])
    if pk <= bt:
        return None
    p, _ = S.pmo(c)
    first = next((i for i in range(bt, pk + 1) if i > 0 and p[i - 1] <= 0 < p[i]), None)
    _, snaps = LX.run(o, hi, lo, c)
    vm = st.median(v[-960:]) or 1
    tc = t[bt] + 900
    ins = None
    if first is not None:
        ins = any(ob["bias"] == 1 and ob["lo"] <= c[first] <= ob["hi"] for ob in snaps[first])
    return {"sym": sym, "chg": chg, "vol": vol, "fund": fund, "move": (hi[pk] / lo[bt] - 1) * 100, "hrs": (t[pk] - t[bt]) / 3600, "bt": t[bt],
            "first": None if first is None else (t[first], (c[first] / lo[bt] - 1) * 100, ins), "maxvol": max(v[bt:pk + 1]) / vm,
            "d": st_at(dd, 86400, tc), "h4": st_at(d4, 14400, tc), "h1": st_at(d1, 3600, tc), "now_off": (c[-1] / hi[pk] - 1) * 100}


def main():
    detail = {d["symbol"]: d for d in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK")]
    tick.sort(key=lambda x: -float(x.get("riseFallRate") or 0))
    top = [(x["symbol"], float(x["riseFallRate"]) * 100, float(x["amount24"]) / 1e6, float(x.get("fundingRate") or 0) * 100) for x in tick[:14]]
    with ThreadPoolExecutor(5) as ex:
        res = [r for r in ex.map(study, top) if r]
    for r in res:
        print(f"== {r['sym']} · 24س {r['chg']:+.1f}% · حجم {r['vol']:.1f}M · تمويل {r['fund']:+.3f}%")
        print(f"   الحركة {r['move']:+.0f}% خلال {r['hrs']:.0f}س من قاع {dt(r['bt']):%d %H:%M} · أعلى حجم ×{r['maxvol']:.0f} · الآن {r['now_off']:+.1f}% عن القمة")
        f = r["first"]
        print("   أول لمس صفر 15د بعد القاع: " + (f"{dt(f[0]):%d %H:%M} (+{f[1]:.1f}% عن القاع) {'داخل أوردر بلوك' if f[2] else 'خارج أوردر بلوك'}" if f else "لا يوجد"))
        print(f"   عند القاع: يومي {tag(r['d'])} | 4س {tag(r['h4'])} | 1س {tag(r['h1'])}")
    for k, nm in (("d", "يومي"), ("h4", "4س"), ("h1", "1س")):
        xs = [r[k] for r in res if r[k]]
        print(f"\nملخص {nm} عند القاع ({len(xs)} عملة): فوق الصفر {sum(x['pos'] for x in xs)} · صاعد {sum(x['up'] for x in xs)} · مكدي أخضر {sum(x['green'] for x in xs)}")
    fl = [r["first"] for r in res if r["first"]]
    print(f"\nلمس صفر 15د حدث بعد القاع في {len(fl)} من {len(res)} · وسيط بعده عن القاع {st.median(x[1] for x in fl):+.1f}% · داخل أوردر بلوك {sum(1 for x in fl if x[2])}")


main()
