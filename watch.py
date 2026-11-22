"""Watch-list experiment. Each run: (1) maybe take a new snapshot (if the last one is older than 6h): from MEXC USDT futures (crypto only, 24h turnover >= 1M USDT) select coins with 24h change >= +1%,
sorted by 24h turnover (top 30) = SELECTED; every other coin in the universe = CONTROL. Prices at snapshot are stored. (2) evaluate all stored snapshots on 15m candles since each snapshot:
highest price within 2h / 6h / 24h / until now, time to +10% and +20%. (3) report: the monitored list, the 'exploders', and selected-vs-control rates. State: watch.json (branch watch-results). Times UTC+4."""
import json
import os
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import scanner as S

TOPN, MIN_TURN, MIN_CHG = 30, 1_000_000, 1.0
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "TQQQ", "SQQQ", "TSLL", "NVDL", "MSTU", "CONL", "SPCX", "NVIDIA", "US30", "TESLA", "BITO", "USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}
dt4 = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=4)
now = int(time.time())


def load():
    try:
        return json.load(open("watch.json")) if os.path.exists("watch.json") else []
    except Exception:
        return []


def snapshot():
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= MIN_TURN and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    rows = [{"sym": x["symbol"], "price": float(x["lastPrice"]), "chg": float(x.get("riseFallRate") or 0) * 100, "turn": float(x["amount24"])} for x in tick]
    cand = sorted([r for r in rows if r["chg"] >= MIN_CHG], key=lambda r: -r["turn"])
    sel = {r["sym"]: i + 1 for i, r in enumerate(cand[:TOPN])}
    for r in rows:
        r["sel"] = r["sym"] in sel
        r["rank"] = sel.get(r["sym"])
    return {"t": now, "rows": rows}


def klines(sym, start):
    out = {"time": [], "high": [], "low": [], "close": []}
    a = start
    while a < now:
        b = min(a + 900 * 1000, now)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min15&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            return None
        for k in out:
            out[k] += d[k]
        a = b
    return out


def evaluate(snaps):
    first = min(s["t"] for s in snaps)
    syms = sorted({r["sym"] for s in snaps for r in s["rows"]})
    with ThreadPoolExecutor(5) as ex:
        K = dict(zip(syms, ex.map(lambda s_: klines(s_, first - 900), syms)))
    res = []
    for s in snaps:
        for r in s["rows"]:
            k = K.get(r["sym"])
            if not k or not k["time"]:
                continue
            idx = [i for i in range(len(k["time"])) if k["time"][i] >= (s["t"] // 900 + 1) * 900]
            if not idx:
                continue
            p0 = r["price"]
            o = {"snap": s["t"], "sym": r["sym"], "sel": r["sel"], "rank": r["rank"], "chg": r["chg"], "turn": r["turn"], "p0": p0, "now": k["close"][-1]}
            for nm, hrs in (("2h", 2), ("6h", 6), ("24h", 24), ("all", 10_000)):
                w = [i for i in idx if k["time"][i] <= s["t"] + hrs * 3600]
                o[nm] = (max(k["high"][i] for i in w) / p0 - 1) * 100 if w else None
            pk = max(idx, key=lambda i: k["high"][i])
            o["peak_t"] = k["time"][pk]
            lo = [i for i in idx if k["time"][i] <= k["time"][pk]]
            o["dd_before_peak"] = (min(k["low"][i] for i in lo) / p0 - 1) * 100
            o["age_h"] = (now - s["t"]) / 3600
            o["t10"] = next(((k["time"][i] - s["t"]) / 3600 for i in idx if k["high"][i] >= p0 * 1.10), None)
            o["t20"] = next(((k["time"][i] - s["t"]) / 3600 for i in idx if k["high"][i] >= p0 * 1.20), None)
            res.append(o)
    return res


def main():
    snaps = load()
    if not snaps or now - snaps[-1]["t"] >= 6 * 3600:
        snaps.append(snapshot())
    snaps = [s for s in snaps if now - s["t"] <= 14 * 86400]
    json.dump(snaps, open("watch.json", "w"))
    res = evaluate(snaps)
    L = []
    last = snaps[-1]
    n_sel = sum(1 for r in last["rows"] if r["sel"])
    L.append(f"مراقبة العملات · {dt4(now):%d %b %H:%M} (UTC+4) · لقطات محفوظة: {len(snaps)} · آخر لقطة {dt4(last['t']):%d %b %H:%M}")
    L.append(f"الاختيار: عملات كريبتو على MEXC الآجلة (حجم 24س ≥ 1M$) ارتفعت +1% فأكثر خلال 24 ساعة، مرتبة حسب حجم التداول، أعلى {TOPN}. الباقي ضابط. (تجربة مراقبة وليست توصية)\n")
    ev = {(o["snap"], o["sym"]): o for o in res}
    cur_rows = sorted([r for r in last["rows"] if r["sel"]], key=lambda r: r["rank"])
    L.append(f"== المراقَبة الآن (لقطة {dt4(last['t']):%d %b %H:%M}، عمرها {(now-last['t'])/3600:.1f} ساعة): {len(cur_rows)} عملة من {len(last['rows'])} في القائمة العامة")
    L.append(f"{'#':>3s} {'العملة':10s} {'حجم 24س':>9s} {'ارتفاع 24س وقت الاختيار':>22s} {'سعر الاختيار':>12s} | {'أبعد سعر منذئذ':>14s} {'متى':>13s} | {'الآن':>8s}")
    for r in cur_rows:
        o = ev.get((last["t"], r["sym"]))
        if o:
            L.append(f"{r['rank']:3d} {r['sym'][:-5]:10s} {r['turn']/1e6:8.1f}M {r['chg']:+21.1f}% {r['price']:12.6g} | {o['all']:+13.1f}% {dt4(o['peak_t']):%d %H:%M} | {(o['now']/r['price']-1)*100:+7.1f}%")
        else:
            L.append(f"{r['rank']:3d} {r['sym'][:-5]:10s} {r['turn']/1e6:8.1f}M {r['chg']:+21.1f}% {r['price']:12.6g} | {'—':>14s} {'—':>13s} | {'—':>8s}")
    boom = sorted([o for o in res if o["all"] is not None and o["all"] >= 10], key=lambda o: -o["all"])
    L.append(f"\n== أيها «انفجر» (ارتفع ≥+10% عن سعر اللقطة في أي لقطة): {len(boom)}")
    for o in boom[:25]:
        L.append(f"   {o['sym'][:-5]:9s} لقطة {dt4(o['snap']):%d %b %H:%M} · {'مختارة #%d' % o['rank'] if o['sel'] else 'ضابط (غير مختارة)'} · أبعد ارتفاع {o['all']:+.1f}% عند {dt4(o['peak_t']):%d %b %H:%M} · وصل +10% بعد {('%.1f س' % o['t10']) if o['t10'] is not None else '—'} · الآن {(o['now']/o['p0']-1)*100:+.1f}%")
    L.append("\n== هل الفرز يلتقط الانفجار؟ (لقطات عمرها 24 ساعة فأكثر فقط)")
    old = [o for o in res if o["age_h"] >= 24]
    for nm, sel in (("مختارة (ارتفعت ≥1% وحجم كبير)", True), ("ضابط (بقية العملات)", False)):
        g = [o for o in old if o["sel"] == sel]
        if len(g) < 10:
            L.append(f"  {nm}: {len(g)} (قليل بعد)"); continue
        L.append(f"  {nm}: {len(g)} مراقبة · وصل +10% خلال 24س: {sum(1 for o in g if o['24h'] >= 10)/len(g)*100:.0f}% · +20%: {sum(1 for o in g if o['24h'] >= 20)/len(g)*100:.1f}% · وسيط أعلى ارتفاع خلال 24س {st.median(o['24h'] for o in g):+.1f}% · وسيط التغير الآن {st.median((o['now']/o['p0']-1)*100 for o in g):+.1f}%")
    if not old:
        L.append("  لا توجد لقطة عمرها 24 ساعة بعد. أعد التشغيل لاحقاً.")
    open("watch.txt", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
