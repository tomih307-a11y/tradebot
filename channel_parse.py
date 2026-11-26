"""Parse a Telegram Desktop export (result.json) of the channel into signals.json: coin, direction, post time, entry, stop, TPs, leverage, timeframe, market, plus the follow-up messages
(TP hits, partial closes, stop closes) linked by reply or coin name. Message dates in the export are the PC's local time (checked against the 'وقت التحقق ... الرياض' lines: Riyadh = UTC+3).
Usage: python3 channel_parse.py result.json [signals.json]"""
import json
import re
import sys

AR = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
NUM = r"([0-9]+(?:\.[0-9]+)?)"


def flat(m):
    t = m.get("text", "")
    if isinstance(t, list):
        t = "".join(x if isinstance(x, str) else x.get("text", "") for x in t)
    return t.translate(AR)


def parse_signal(t):
    coin = re.search(r"العملة:\s*([A-Z0-9]+)\s*/\s*USDT", t) or re.search(r"🔥\s*([A-Z0-9]+)\s*\|\s*LONG", t)
    if not coin:
        return None
    entry = re.search(r"(?:منطقة الدخول|الدخول)\s*:?\s*" + NUM, t)
    stop = re.search(r"(?:وقف الخسارة|الوقف)\s*:?\s*" + NUM, t)
    tps = [float(x) for x in re.findall(r"TP\d\s*[:—-]\s*" + NUM, t)]
    if not (entry and stop and tps):
        return None
    lev = re.search(r"الرافعة:\s*(\d+)\s*[Xx]", t)
    tf = re.search(r"الفريم:\s*([0-9]+\s*[MmHh])", t)
    return {"coin": coin.group(1), "direction": "SHORT" if "SHORT" in t or "🔴" in t.split("الاتجاه")[-1][:12] else "LONG", "entry": float(entry.group(1)), "stop": float(stop.group(1)),
            "tps": tps, "leverage": int(lev.group(1)) if lev else None, "tf": tf.group(1).replace(" ", "") if tf else None,
            "market": "futures" if ("فيوتشرز" in t and "سبوت" not in t) else ("spot" if "سبوت" in t and "فيوتشرز" not in t else "both" if "سبوت" in t else None)}


def main(path, out="signals.json"):
    msgs = json.load(open(path, encoding="utf-8"))["messages"]
    sigs, by_id = [], {}
    for m in msgs:
        if m.get("type") != "message":
            continue
        t = flat(m)
        s = parse_signal(t)
        if s:
            s.update(id=m["id"], date=m["date"], text=t[:300], events=[])
            sigs.append(s); by_id[m["id"]] = s
    ev_re = [("tp", r"تحقق الهدف|تم الهدف|TP\d\s*—"), ("partial", r"إغلاق جزئي"), ("stop", r"إغلاق على وقف|ضرب وقف|🛑 إغلاق|🏁 إغلاق الصفقة"), ("done", r"ختام الأهداف")]
    for m in msgs:
        if m.get("type") != "message":
            continue
        t = flat(m)
        if parse_signal(t):
            continue
        kinds = [k for k, rx in ev_re if re.search(rx, t)]
        if not kinds:
            continue
        s = by_id.get(m.get("reply_to_message_id"))
        if s is None:
            c = re.search(r"\b([A-Z0-9]{2,10})\s*•\s*LONG|^[^\n]*?\b([A-Z0-9]{2,10})\s*[—-]\s*(?:\d|تم)", t, re.M)
            name = (c.group(1) or c.group(2)) if c else None
            cand = [x for x in sigs if x["coin"] == name and x["date"] <= m["date"]]
            s = cand[-1] if cand else None
        if s:
            s["events"].append({"date": m["date"], "kinds": kinds, "text": t[:200]})
    json.dump(sigs, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"رسائل {len(msgs)} · توصيات مستخرجة {len(sigs)} · أول {sigs[0]['date'] if sigs else '-'} · آخر {sigs[-1]['date'] if sigs else '-'}")
    for s in sigs[:8]:
        print(f"  {s['date']} {s['coin']} دخول {s['entry']} وقف {s['stop']} أهداف {len(s['tps'])} رافعة {s['leverage']} فريم {s['tf']} · أحداث {len(s['events'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "signals.json")
