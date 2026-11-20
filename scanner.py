"""Scanner v1 (long only). Rules agreed with the user:
 1. PMO(35,20,10) blue line crosses zero UP on 15m.            2. At that bar the close is INSIDE a bullish internal order block (LuxAlgo logic).
 3. MACD(12,26,9) bullish at that bar (line above signal).     4. Close above TEMA(50).
 Larger frames (4H, daily) are reported: above zero / about to touch ("inevitable": blue>orange, rising, distance <= 3 candles of current speed) / below zero.
 Stop = low of the order block. Usage: python3 scanner.py [hours_back]"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import luxalgo_ob as LX

SEC = {"Min15": 900, "Hour4": 14400, "Day1": 86400}
TZ = 4


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


NON_SPOT = {"BINANCETICKER", "KAIKO", "ALLTICK", "INFOWAYCOMMON", "LIGHTER", "HYPERLIQUID"}


STABLE = {"USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}


def is_crypto(detail):
    """Crypto = has at least one SPOT price source (MEXC, BINANCE, BITGET...), and is not a commodity-style name like GOLD(PAXG)."""
    spot = [o for o in detail.get("indexOrigin") or [] if not o.endswith("_FUTURE") and o not in NON_SPOT]
    return bool(spot) and "(" not in (detail.get("baseCoinName") or "") and (detail.get("baseCoin") or (detail.get("symbol") or "").split("_")[0]).upper() not in STABLE


def kl(sym, itv, days, now):
    d = get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval={itv}&start={now - days * 86400}&end={now}")["data"]
    keep = [i for i, tt in enumerate(d["time"]) if tt + SEC[itv] <= now]                  # closed candles only
    return [[d[k][i] for i in keep] for k in ("time", "open", "high", "low", "close")]


def ema(x, n):
    a, e, out = 2 / (n + 1), x[0], []
    for v in x:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


def smooth(x, n):
    a, e, out = 2 / n, x[0], []
    for v in x:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


def pmo(c, l1=35, l2=20, sg=10):
    roc = [0.0] + [(c[i] / c[i - 1] - 1) * 100 for i in range(1, len(c))]
    p = smooth([10 * v for v in smooth(roc, l1)], l2)
    return p, ema(p, sg)


def tema(x, n):
    e1 = ema(x, n); e2 = ema(e1, n); e3 = ema(e2, n)
    return [3 * a - 3 * b + c for a, b, c in zip(e1, e2, e3)]


def frame_state(p, s, j):
    """Larger-frame PMO state at closed bar j."""
    if j < 2:
        return "؟"
    if p[j] > 0:
        return "فوق الصفر"
    slope = p[j] - p[j - 1]
    if p[j] > s[j] and slope > 0 and -p[j] <= 3 * slope:
        return "على وشك لمس الصفر"
    return "تحت الصفر"


def last_closed_before(times, period, t_close):
    j = -1
    for k, tt in enumerate(times):
        if tt + period <= t_close:
            j = k
        else:
            break
    return j


def analyse(sym, now, hours_back):
    t, o, h, l, c = kl(sym, "Min15", 14, now)
    if len(c) < 400:
        return []
    t4, _, _, _, c4 = kl(sym, "Hour4", 60, now)
    td, _, _, _, cd = kl(sym, "Day1", 250, now)
    p, s = pmo(c)
    p4, s4 = pmo(c4) if len(c4) > 40 else ([0] * len(c4), [0] * len(c4))
    pd, sd = pmo(cd) if len(cd) > 40 else ([0] * len(cd), [0] * len(cd))
    macd = [a - b for a, b in zip(ema(c, 12), ema(c, 26))]
    msig = ema(macd, 9)
    t50 = tema(c, 50)
    obs, snaps = LX.run(o, h, l, c)
    out = []
    for i in range(max(1, len(c) - int(hours_back * 4)), len(c)):
        if not (p[i - 1] <= 0 < p[i]):
            continue
        tc = t[i] + 900
        inside = [ob for ob in snaps[i] if ob["bias"] == 1 and ob["lo"] <= c[i] <= ob["hi"]]
        j4, jd = last_closed_before(t4, 14400, tc), last_closed_before(td, 86400, tc)
        out.append(dict(sym=sym, t=t[i], age=(now - tc) / 60, px=c[i], pmo=p[i],
                        ob=bool(inside), ob_zone=(inside[0]["lo"], inside[0]["hi"]) if inside else None,
                        macd=macd[i] > msig[i], macd_pos=macd[i] > 0, tema=c[i] > t50[i],
                        f4=frame_state(p4, s4, j4), fd=frame_state(pd, sd, jd)))
    return out


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6
    now = int(time.time())
    detail = {d["symbol"]: d for d in get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x for x in get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000]
    res, errs = [], 0
    for x in tick:
        try:
            res += analyse(x["symbol"], now, hours)
        except Exception as e:
            errs += 1
            print("skip", x["symbol"], e, file=sys.stderr)
    f = lambda tt: (datetime.fromtimestamp(tt, timezone.utc) + timedelta(hours=TZ)).strftime("%d %H:%M")
    mk = lambda b: "✓" if b else "✗"
    allp = [r for r in res if r["ob"] and r["macd"] and r["tema"]]
    lines = [f"ماسح v1 (لونج، كريبتو فقط) · {len(tick)} عقداً (سيولة ≥ 1M) · آخر {hours:g} ساعات · {datetime.fromtimestamp(now, timezone.utc) + timedelta(hours=TZ):%d %b %H:%M} (UTC+4) · أخطاء جلب: {errs}", "",
             f"لمسات PMO للصفر صعوداً على 15 دقيقة: {len(res)} · داخل أوردر بلوك: {sum(r['ob'] for r in res)} · المكدي صاعد: {sum(r['macd'] for r in res)} · فوق TEMA50: {sum(r['tema'] for r in res)} · **كل الشروط الأربعة: {len(allp)}**", ""]
    hdr = f"{'للبحث في TV':<16}{'الوقت':<12}{'منذ(د)':>7}  OB MACD TEMA  {'4س':<18}{'يومي':<18} سعر"
    lines += ["== كل الشروط متحققة ==", hdr] + [row for row in []]
    def fmt(r):
        stop = f" · وقف (أدنى OB) {r['ob_zone'][0]:.6g}" if r["ob_zone"] else ""
        return f"{r['sym'].replace('_', '') + '.P':<16}{f(r['t']):<12}{r['age']:>7.0f}  {mk(r['ob']):^2}  {mk(r['macd']):^3}  {mk(r['tema']):^3}   {r['f4']:<18}{r['fd']:<18} {r['px']:.6g}{stop}"
    lines += [fmt(r) for r in sorted(allp, key=lambda r: r["age"])] or ["  لا شيء"]
    near = [r for r in res if sum((r["ob"], r["macd"], r["tema"])) == 2 and not (r["ob"] and r["macd"] and r["tema"])]
    lines += ["", "== شرطان من ثلاثة (قريبة) ==", hdr] + [fmt(r) for r in sorted(near, key=lambda r: r["age"])[:25]]
    text = "\\n".join(lines).replace("\\n", "\n")
    print(text)
    open("scan.txt", "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
