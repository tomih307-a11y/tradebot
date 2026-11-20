"""Forward tracker for the PMO-only rule. Run hourly. 1) detect new 15m PMO zero-touches (last hour) on crypto futures; log them as virtual long trades
(entry = latest price at detection, stop = 1.5xATR clamp 0.5-6%). Group 'match' = daily PMO above zero; group 'other' = control. 2) update open trades:
stop hit -> -1R; 15m PMO blue crosses below orange -> exit at that close; after 24h -> exit at close. Costs 0.1%. State in tracker.json, report in track.txt.
This is a measurement, not a recommendation."""
import json
import os
import statistics as st
import time
from datetime import datetime, timedelta, timezone

import scanner as S
import scanner_pmo as P

TZ = 4
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)


def deals_buy_ratio(sym):
    try:
        d = S.get(f"https://contract.mexc.com/api/v1/contract/deals/{sym}?limit=100")["data"]
        buy = sum(x["v"] for x in d if x["T"] == 1)
        tot = sum(x["v"] for x in d) or 1
        return buy / tot, d[0]["t"] - d[-1]["t"]
    except Exception:
        return None, None


def evaluate(tr, now):
    if tr["status"] != "open":
        return
    d = P.kl(tr["sym"], "Min15", now)
    if not d:
        return
    t, hi, lo, c = d["time"], d["high"], d["low"], d["close"]
    p, ps = S.pmo(c)
    risk = tr["entry"] - tr["stop"]
    cost = 0.001 * tr["entry"] / risk
    start = tr["detect_t"]
    last = None
    for j in range(1, len(c)):
        if t[j] < start:
            continue
        last = j
        age = (t[j] + 900 - start) / 3600
        if lo[j] <= tr["stop"]:
            return close(tr, -1.0 - cost, "وقف", tr["stop"], t[j] + 900)
        if p[j] < ps[j] and p[j - 1] >= ps[j - 1]:
            return close(tr, (c[j] - tr["entry"]) / risk - cost, "إشارة خروج PMO", c[j], t[j] + 900)
        if age >= 24:
            return close(tr, (c[j] - tr["entry"]) / risk - cost, "مرور 24 ساعة", c[j], t[j] + 900)
    tr["now"] = c[-1]
    tr["open_R"] = (c[-1] - tr["entry"]) / risk - cost


def close(tr, R, why, px, ct):
    tr.update({"status": "closed", "R": R, "why": why, "exit_px": px, "closed_t": ct})


def main():
    now = int(time.time())
    try:
        state = json.load(open("tracker.json")) if os.path.exists("tracker.json") else []
    except Exception:
        state = []
    open_syms = {x["sym"] for x in state if x["status"] == "open"}
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in P.ETF]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(5) as ex:
        res = [r for r in ex.map(P.work, [(x["symbol"], 1) for x in tick]) if r]
    snap = {x["symbol"]: float(x.get("holdVol") or 0) for x in tick}
    try:
        oilog = json.load(open("oi_log.json")) if os.path.exists("oi_log.json") else []
    except Exception:
        oilog = []                                                   # empty or corrupt file: start a fresh snapshot log
    prev = next((e for e in reversed(oilog) if now - e["t"] >= 50 * 60), None)           # snapshot at least 50 min old
    oilog.append({"t": now, "oi": snap})
    oilog = [e for e in oilog if now - e["t"] <= 14 * 86400][-400:]
    json.dump(oilog, open("oi_log.json", "w"))
    fund = {x["symbol"]: float(x.get("fundingRate") or 0) * 100 for x in tick}
    known = {(x["sym"], x["signal_t"]) for x in state}
    new = []
    for r in res:
        if not r["touch"] or (r["sym"], r["t"]) in known or r["sym"] in open_syms:
            continue
        entry = r["now"]
        stop = r["stop"] if r["stop"] < entry else entry * 0.98
        d1 = r["d1"]
        tr = {"sym": r["sym"], "signal_t": r["t"], "detect_t": now, "entry": entry, "stop": min(stop, entry * 0.995), "group": "match" if d1 and d1["pos"] else "other",
              "h1pos": bool(r["h1"] and r["h1"]["pos"]), "h4pos": bool(r["h4"] and r["h4"]["pos"]), "status": "open",
              "oi": snap.get(r["sym"]), "funding": fund.get(r["sym"]),
              "oi_chg": ((snap[r["sym"]] / prev["oi"][r["sym"]] - 1) * 100 if prev and prev["oi"].get(r["sym"]) else None)}
        br, span = deals_buy_ratio(r["sym"])
        tr["buy_ratio"], tr["deals_span_ms"] = br, span
        state.append(tr); new.append(tr)
    for tr in state:
        if tr["status"] == "open":
            evaluate(tr, now)
    json.dump(state, open("tracker.json", "w"), ensure_ascii=False)

    lines = [f"متابعة PMO (لمس الصفر 15د) · {dt(now):%d %H:%M} UTC+{TZ} · دخول افتراضي بسعر لحظة الرصد · وقف 1.5×ATR · خروج: وقف | الأزرق يقطع البرتقالي | 24 ساعة · تكلفة 0.1%"]
    lines.append("هذه قياس وليس توصية.\n")
    for grp, nm in (("match", "تطابق القاعدة (اليومي فوق الصفر)"), ("other", "مقارنة (اليومي تحت الصفر)")):
        g = [x for x in state if x["group"] == grp]
        cl = [x for x in g if x["status"] == "closed"]
        op = [x for x in g if x["status"] == "open"]
        if cl:
            Rs = [x["R"] for x in cl]
            lines.append(f"{nm}: مغلقة {len(cl)} · رابحة {sum(1 for r in Rs if r > 0)} ({sum(1 for r in Rs if r > 0)/len(Rs)*100:.0f}%) · متوسط R {st.mean(Rs):+.2f} · مجموع R {sum(Rs):+.2f} · مفتوحة {len(op)}")
        else:
            lines.append(f"{nm}: مغلقة 0 · مفتوحة {len(op)}")
    lines.append("")
    if new:
        lines.append("رُصدت الآن:")
        for x in new:
            lines.append(f"  + {x['sym'][:-5]}USDT.P · {'تطابق' if x['group']=='match' else 'مقارنة'} · دخول {x['entry']:.6g} · وقف {x['stop']:.6g} (بعد {(x['entry']/x['stop']-1)*100:.1f}%)")
        lines.append("")
    lines.append("المفتوحة:")
    for x in sorted([x for x in state if x["status"] == "open"], key=lambda x: -x["detect_t"]):
        lines.append(f"  {x['sym'][:-5]}USDT.P · {dt(x['detect_t']):%d %H:%M} · {'تطابق' if x['group']=='match' else 'مقارنة'} · دخول {x['entry']:.6g} الآن {x.get('now', x['entry']):.6g} · R حالي {x.get('open_R', 0):+.2f}")
    lines.append("\nآخر المغلقة:")
    for x in sorted([x for x in state if x["status"] == "closed"], key=lambda x: -x["closed_t"])[:25]:
        lines.append(f"  {x['sym'][:-5]}USDT.P · {dt(x['detect_t']):%d %H:%M} → {dt(x['closed_t']):%d %H:%M} · {'تطابق' if x['group']=='match' else 'مقارنة'} · {x['why']} · R {x['R']:+.2f}")
    open("track.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
