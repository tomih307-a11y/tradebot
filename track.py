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
import journal as J
import paper_account as PA

TZ = 4
dt = lambda x: datetime.fromtimestamp(x, timezone.utc) + timedelta(hours=TZ)


def btc24_at(btc, ts):
    t, c = btc["time"], btc["close"]
    j = max((k for k in range(len(t)) if t[k] + 900 <= ts), default=None)
    if j is None or j < 96:
        return None
    return (c[j] / c[j - 96] - 1) * 100


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


def peaks(tr, now):
    """Farthest prices after entry, 'second wave' flags at the signal bar, and the wide-stop alternative (stop 10%, hold 24h)."""
    d = P.kl(tr["sym"], "Min15", now)
    if not d:
        return
    t, hi, lo, c = d["time"], d["high"], d["low"], d["close"]
    end = tr.get("closed_t", now)
    during = [j for j in range(len(t)) if t[j] >= tr["detect_t"] and t[j] < end]
    since = [j for j in range(len(t)) if t[j] >= tr["detect_t"]]
    if during:
        j = max(during, key=lambda k: hi[k])
        tr["peak_during"], tr["peak_during_t"] = hi[j], t[j]
        tr["trough_during"] = min(lo[k] for k in during)
    if since:
        j = max(since, key=lambda k: hi[k])
        tr["peak_since"], tr["peak_since_t"] = hi[j], t[j]
    if "w3" not in tr:                                              # flags measured at the signal bar (past data only)
        i = next((k for k in range(len(t)) if t[k] == tr["signal_t"]), None)
        if i is not None and i >= 700:
            trr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(i + 1)]
            atr = S.ema(trr, 14)
            hot = atr[i] / (st.median(atr[max(0, i - 960):i]) or 1e-12) >= 1.2
            week = c[i] / c[i - 672] - 1 > 0
            pull = (c[i] / max(hi[i - 95:i + 1]) - 1) * 100 <= -6
            d1 = tr["group"] == "match"
            tr["hot"], tr["week"], tr["pull"] = bool(hot), bool(week), bool(pull)
            tr["w3"] = bool(d1 and hot and week)
            tr["w4"] = bool(d1 and hot and week and pull)
    if "alt_pct" not in tr and now - tr["detect_t"] >= 24 * 3600:    # wide-stop alternative: stop 10% below entry, exit after 24h
        out = None
        for j in range(len(t)):
            if t[j] < tr["detect_t"]:
                continue
            if lo[j] <= tr["entry"] * 0.90:
                out = tr["entry"] * 0.90; break
            if t[j] + 900 - tr["detect_t"] >= 24 * 3600:
                out = c[j]; break
        if out is not None:
            tr["alt_pct"] = (out / tr["entry"] - 1) * 100 - 0.1


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
    btc = P.kl("BTC_USDT", "Min15", now)
    for x in state:                                              # backfill / compute BTC 24h return at each trade's detection time
        if x.get("btc24") is None and btc:
            x["btc24"] = btc24_at(btc, x["detect_t"])
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
              "oi": snap.get(r["sym"]), "funding": fund.get(r["sym"]), "btc24": btc24_at(btc, now) if btc else None,
              "oi_chg": ((snap[r["sym"]] / prev["oi"][r["sym"]] - 1) * 100 if prev and prev["oi"].get(r["sym"]) else None)}
        br, span = deals_buy_ratio(r["sym"])
        tr["buy_ratio"], tr["deals_span_ms"] = br, span
        state.append(tr); new.append(tr)
    for tr in state:
        if tr["status"] == "open":
            evaluate(tr, now)
    for tr in state:
        if tr["status"] == "open" or now - tr.get("closed_t", 0) < 48 * 3600 or "peak_since" not in tr:
            peaks(tr, now)
    json.dump(state, open("tracker.json", "w"), ensure_ascii=False)
    try:
        journal = json.load(open("journal.json")) if os.path.exists("journal.json") else []
    except Exception:
        journal = []
    J.update(journal, state)
    json.dump(journal, open("journal.json", "w"), ensure_ascii=False)

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
    mt = [x for x in state if x["group"] == "match" and x.get("btc24") is not None]
    lines.append("\nاختبار إضافي: مجموعة التطابق مقسّمة حسب حركة بيتكوين خلال 24 ساعة قبل الإشارة (فرضية مستخرجة من الأرقام، تُقاس هنا فقط):")
    for flag, nm in ((True, "بيتكوين هابط (> -1%)"), (False, "بيتكوين غير هابط")):
        g = [x for x in mt if (x["btc24"] <= -1.0) == flag]
        cl = [x for x in g if x["status"] == "closed"]
        op = [x for x in g if x["status"] == "open"]
        if cl:
            Rs = [x["R"] for x in cl]
            lines.append(f"  تطابق + {nm}: مغلقة {len(cl)} · رابحة {sum(1 for r in Rs if r > 0)} · متوسط R {st.mean(Rs):+.2f} · مجموع R {sum(Rs):+.2f} · مفتوحة {len(op)}")
        else:
            lines.append(f"  تطابق + {nm}: مغلقة 0 · مفتوحة {len(op)}")
    lines.append("\nاختبار الموجة الثانية (مجموعات متراكبة داخل التطابق):")
    for key, nm in (("w3", "W3: تطابق + متقلبة (ATR≥1.2×) + صاعدة أسبوعاً"), ("w4", "W4: W3 + تراجع ≥6% عن قمة 24س")):
        g = [x for x in state if x.get(key)]
        cl = [x for x in g if x["status"] == "closed"]
        op = [x for x in g if x["status"] == "open"]
        alt = [x["alt_pct"] for x in g if "alt_pct" in x]
        txt = f"  {nm}: {len(g)} إشارة · مغلقة {len(cl)} · مفتوحة {len(op)}"
        if cl:
            Rs = [x["R"] for x in cl]
            txt += f" · رابحة {sum(1 for r in Rs if r > 0)} · متوسط R {st.mean(Rs):+.2f}"
        if alt:
            txt += f" · البديل (وقف 10% وخروج بعد 24س): {len(alt)} صفقة، متوسط {st.mean(alt):+.2f}%"
        lines.append(txt)
    lines.append("\nالحساب الورقي (1000$، مخاطرة 1% من الرصيد الحالي، أقصى 5 صفقات، رافعة ≤5×، رسوم+انزلاق+تمويل، فرامل هبوط):")
    for g_, nm in (("match", "تطابق"), ("w3", "W3"), ("w4", "W4"), ("all", "الكل (تطابق + مقارنة)")):
        a = PA.simulate(state, 1.0, 5, g_)
        lines.append(f"  {nm}: صفقات {a['n']} · الرصيد {a['equity']:.2f}$ ({(a['equity']/1000-1)*100:+.2f}%) · مع المفتوحة {a['marked']:.2f}$ ({(a['marked']/1000-1)*100:+.2f}%) · أقصى هبوط {a['maxdd']*100:.1f}% · مفتوحة {a['open']} · تخطّي {a['skipped']}")
    lines.append("\nسجل التأمل (الذاكرة): نتائج الصفقات المغلقة حسب الوسم التلقائي:")
    lines += J.summary(journal)
    lines.append("")
    if new:
        lines.append("رُصدت الآن:")
        for x in new:
            lines.append(f"  + {x['sym'][:-5]}USDT.P · {'تطابق' if x['group']=='match' else 'مقارنة'} · دخول {x['entry']:.6g} · وقف {x['stop']:.6g} (بعد {(x['entry']/x['stop']-1)*100:.1f}%)")
        lines.append("")
    pc = lambda a, b: (a / b - 1) * 100
    lines.append("المفتوحة (سعر الدخول ← أبعد سعر وصله والنسبة ← السعر الآن):")
    for x in sorted([x for x in state if x["status"] == "open"], key=lambda x: -x["detect_t"]):
        pk = x.get("peak_during", x["entry"])
        nw = x.get("now", x["entry"])
        lines.append(f"  {x['sym'][:-5]}USDT.P · {dt(x['detect_t']):%d %H:%M} · {'تطابق' if x['group']=='match' else 'مقارنة'}{' · W4' if x.get('w4') else (' · W3' if x.get('w3') else '')}{(' · BTC24 %+.1f%%' % x['btc24']) if x.get('btc24') is not None else ''} · دخول {x['entry']:.6g} ← أبعد سعر {pk:.6g} ({pc(pk, x['entry']):+.2f}%" + (f" عند {dt(x['peak_during_t']):%d %H:%M}" if 'peak_during_t' in x else "") + f") ← الآن {nw:.6g} ({pc(nw, x['entry']):+.2f}%) · R حالي {x.get('open_R', 0):+.2f}")
    lines.append("\nآخر المغلقة (دخول ← أبعد سعر خلال الصفقة ← الخروج | أبعد سعر منذ الدخول حتى الآن):")
    for x in sorted([x for x in state if x["status"] == "closed"], key=lambda x: -x["closed_t"])[:25]:
        pk = x.get("peak_during", x["exit_px"])
        ps_ = x.get("peak_since")
        lines.append(f"  {x['sym'][:-5]}USDT.P · {dt(x['detect_t']):%d %H:%M} → {dt(x['closed_t']):%d %H:%M} · {'تطابق' if x['group']=='match' else 'مقارنة'} · {x['why']} · دخول {x['entry']:.6g} ← أبعد {pk:.6g} ({pc(pk, x['entry']):+.2f}%) ← خروج {x['exit_px']:.6g} ({pc(x['exit_px'], x['entry']):+.2f}%) · R {x['R']:+.2f}"
                     + (f" | منذ الدخول أبعد سعر {ps_:.6g} ({pc(ps_, x['entry']):+.2f}%)" if ps_ else ""))
    for grp, nm in (("match", "تطابق"), ("other", "مقارنة")):
        g = [x for x in state if x["group"] == grp and "peak_during" in x]
        if g:
            lines.append(f"\nمتوسط أبعد صعود خلال الصفقة ({nm}، {len(g)} صفقة): {st.mean(pc(x['peak_during'], x['entry']) for x in g):+.2f}% · متوسط أبعد هبوط {st.mean(pc(x['trough_during'], x['entry']) for x in g):+.2f}%")
    open("track.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
