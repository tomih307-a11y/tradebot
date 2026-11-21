"""Feature dataset for formula discovery. One row per entry signal (PMO zero-touch up OR PMO bullish cross + MACD entry) on 15m, 70 days, crypto futures.
Features are all known at the close of the signal bar. Labels: net R of 3 trade definitions + fixed-horizon net returns + MFE/MAE."""
import json
import math
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import luxalgo_ob as LX
import scanner as S

SEC = {"Min15": 900, "Min60": 3600, "Hour4": 14400, "Day1": 86400}
DAYS = {"Min15": 70, "Min60": 130, "Hour4": 300, "Day1": 300}
NAN = float("nan")


def kline(sym, itv, now):
    out = {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
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


def frame(d, per, tc, price, tag, row):
    times, closes = d["time"], d["close"]
    k = 0
    lo_, hi_ = 0, len(times)
    while lo_ < hi_:                                                      # bisect: closed candles with time+per <= tc
        mid = (lo_ + hi_) // 2
        if times[mid] + per <= tc:
            lo_ = mid + 1
        else:
            hi_ = mid
    series = closes[max(0, lo_ - 700):lo_] + [price]
    if len(series) < 150:
        for f in ("p", "gap", "slope", "hist", "hslope", "green", "above"):
            row[f"{tag}_{f}"] = NAN
        return
    p, s = S.pmo(series)
    m = [a - b for a, b in zip(S.ema(series, 12), S.ema(series, 26))]
    g = S.ema(m, 9)
    n = len(series) - 1
    h = [(a - b) / series[n] * 100 for a, b in zip(m, g)]
    row[f"{tag}_p"] = p[n]
    row[f"{tag}_gap"] = p[n] - s[n]
    row[f"{tag}_slope"] = p[n] - p[n - 1]
    row[f"{tag}_hist"] = h[n]
    row[f"{tag}_hslope"] = h[n] - h[n - 1]
    row[f"{tag}_green"] = 1.0 if h[n] > 0 else 0.0
    row[f"{tag}_above"] = 1.0 if p[n] > s[n] else 0.0


BT = {}


def btc_prepare(now):
    d = kline("BTC_USDT", "Min15", now)
    BT["t"], BT["c"] = d["time"], d["close"]
    BT["idx"] = {tt: i for i, tt in enumerate(d["time"])}
    BT["h1"], BT["h4"] = kline("BTC_USDT", "Min60", now), kline("BTC_USDT", "Hour4", now)


def work(sym):
    now = int(time.time())
    try:
        k15, k1, k4, kd = kline(sym, "Min15", now), kline(sym, "Min60", now), kline(sym, "Hour4", now), kline(sym, "Day1", now)
    except Exception:
        return []
    if not (k15 and k1 and k4 and kd):
        return []
    t, o, hi, lo, c, vo = (k15[k] for k in ("time", "open", "high", "low", "close", "vol"))
    n = len(c)
    p, ps = S.pmo(c)
    te = S.tema(c, 50)
    m = [a - b for a, b in zip(S.ema(c, 12), S.ema(c, 26))]
    g = S.ema(m, 9)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    _, snaps = LX.run(o, hi, lo, c)
    turn = [vo[k] * c[k] for k in range(n)]
    out, last = [], -99
    lastzero = lastcross = -9999
    for i in range(1, n):
        if p[i - 1] <= 0 < p[i]:
            lastzero_new = i
        else:
            lastzero_new = None
        if p[i - 1] <= ps[i - 1] and p[i] > ps[i]:
            cross_now = True
        else:
            cross_now = False
        touch = lastzero_new is not None
        mc = any(m[k - 1] <= g[k - 1] and m[k] > g[k] for k in range(max(i - 2, 1), i + 1)) or any(m[k - 1] - g[k - 1] <= 0 < m[k] - g[k] for k in range(max(i - 2, 1), i + 1))
        early = cross_now and mc
        if touch:
            lastzero = i
        if cross_now:
            lastcross = i
        if i < 700 or not (touch or early) or i - last < 4 or False:
            continue
        last = i
        ent, tc = c[i], t[i] + 900
        row = {"sym": sym, "t": t[i], "touch": 1.0 if touch else 0.0, "early": 1.0 if early else 0.0, "pmo": p[i], "pmo_gap": p[i] - ps[i], "pmo_slope": p[i] - p[i - 1],
               "pmo_below0": 1.0 if p[i] < 0 else 0.0, "since_zero": min(i - lastzero, 400), "since_cross": min(i - lastcross, 400),
               "macd_hist": (m[i] - g[i]) / ent * 100, "macd_hslope": ((m[i] - g[i]) - (m[i - 1] - g[i - 1])) / ent * 100, "macd_pos": 1.0 if m[i] > 0 else 0.0,
               "tema_dist": (ent / te[i] - 1) * 100, "tema_slope": (te[i] / te[i - 3] - 1) * 100, "atr_pct": atr[i] / ent * 100}
        base = st.median(vo[max(0, i - 960):i - 8]) or 1e-12
        row["vol_max2h"] = max(vo[i - 7:i + 1]) / base
        row["vol_bar"] = vo[i] / base
        rng = hi[i] - lo[i] or 1e-12
        row["close_pos"] = (c[i] - lo[i]) / rng
        row["body_pct"] = (c[i] - o[i]) / ent * 100
        for nm, b in (("ret1h", 4), ("ret4h", 16), ("ret12h", 48), ("ret24h", 96)):
            row[nm] = (ent / c[i - b] - 1) * 100
        h24, l24 = max(hi[i - 95:i + 1]), min(lo[i - 95:i + 1])
        row["dd24h"] = (ent / h24 - 1) * 100
        row["up_from_low24h"] = (ent / l24 - 1) * 100
        row["pos24h"] = (ent - l24) / ((h24 - l24) or 1e-12)
        row["log_turnover"] = math.log10(st.median(turn[max(0, i - 960):i]) + 1)
        bulls = [ob for ob in snaps[i] if ob["bias"] == 1]
        bears = [ob for ob in snaps[i] if ob["bias"] == -1]
        below = [ob for ob in bulls if ob["lo"] < ent]
        row["ob_inside"] = 1.0 if any(ob["lo"] <= ent <= ob["hi"] for ob in bulls) else 0.0
        row["ob_dist_below"] = min((ent / ob["hi"] - 1) * 100 for ob in below) if below else NAN
        above = [ob for ob in bears if ob["lo"] > ent]
        row["bear_ob_dist_above"] = min((ob["lo"] / ent - 1) * 100 for ob in above) if above else NAN
        row["n_bull_ob"] = float(len(bulls))
        dtl = datetime.fromtimestamp(t[i], timezone.utc) + timedelta(hours=4)
        row["hour"] = float(dtl.hour)
        row["wday"] = float(dtl.weekday())
        frame(k1, 3600, tc, ent, "h1", row)
        frame(k4, 14400, tc, ent, "h4", row)
        frame(kd, 86400, tc, ent, "d1", row)
        bi = BT["idx"].get(t[i])
        if bi and bi >= 100:
            bc = BT["c"]
            row["btc_ret1h"] = (bc[bi] / bc[bi - 4] - 1) * 100
            row["btc_ret4h"] = (bc[bi] / bc[bi - 16] - 1) * 100
            row["btc_ret24h"] = (bc[bi] / bc[bi - 96] - 1) * 100
        else:
            row["btc_ret1h"] = row["btc_ret4h"] = row["btc_ret24h"] = NAN
        if i + 96 < n:
            # labels
            for hn, hb in (("f2h", 8), ("f4h", 16), ("f12h", 48), ("f24h", 96)):
                row[hn] = (c[i + hb] / ent - 1) * 100 - 0.1
            row["mfe24"] = (max(hi[i + 1:i + 97]) / ent - 1) * 100
            row["mae24"] = (min(lo[i + 1:i + 97]) / ent - 1) * 100
            for sn, stp in (("swing", min(lo[i - 15:i + 1]) * 0.999), ("atr", ent - 1.5 * atr[i]), ("fix2", ent * 0.98)):
                stp = max(min(stp, ent * 0.995), ent * 0.94)
                risk = ent - stp
                cost = 0.001 * ent / risk
                for xn in ("A", "B", "T12", "TR"):
                    R = None
                    hh = ent
                    for j in range(i + 1, i + 97):
                        lvl = stp
                        if xn == "TR":
                            lvl = max(stp, hh - 2 * atr[i])                      # trailing: highest high so far minus 2 ATR (uses highs before this bar)
                        if lo[j] <= lvl:
                            R = (lvl - ent) / risk; break
                        hh = max(hh, hi[j])
                        if xn == "A" and c[j] < te[j] and (m[j] - g[j]) < 0:
                            R = (c[j] - ent) / risk; break
                        if xn == "B" and m[j] < g[j] and m[j - 1] >= g[j - 1]:
                            R = (c[j] - ent) / risk; break
                        if xn == "T12" and j == i + 48:
                            R = (c[j] - ent) / risk; break
                    if R is None:
                        R = (c[i + 96] - ent) / risk
                    row[f"R_{xn}_{sn}"] = R - cost
        out.append(row)
    return out


def main():
    now = int(time.time())
    detail = {d["symbol"]: d for d in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "TQQQ", "SQQQ", "TSLL", "NVDL", "MSTU", "CONL", "SPCX", "NVIDIA", "US30", "TESLA", "BITO"}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    btc_prepare(now)
    print("coins", len(tick))
    with ThreadPoolExecutor(5) as ex:
        rows = [r for part in ex.map(work, tick) for r in part]
    json.dump(rows, open("dataset.json", "w"))
    print("rows", len(rows), "span", datetime.fromtimestamp(min(r["t"] for r in rows), timezone.utc), datetime.fromtimestamp(max(r["t"] for r in rows), timezone.utc))


if __name__ == "__main__":
    main()
