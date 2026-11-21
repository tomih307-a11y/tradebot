"""Memory / reflection journal. One entry per closed trade: context at entry, outcome, and automatic post-mortem tags. Weekly summary = outcome by tag."""
import statistics as st


def tags_for(x):
    pct = lambda a, b: (a / b - 1) * 100
    t = []
    entry = x["entry"]
    risk_pct = (1 - x["stop"] / entry) * 100
    peak = pct(x["peak_during"], entry) if x.get("peak_during") else 0.0
    exit_pct = pct(x["exit_px"], entry)
    since = pct(x["peak_since"], entry) if x.get("peak_since") else peak
    if x["R"] > 0:
        t.append("رابحة")
        if since >= 2 * max(exit_pct, 0.5):
            t.append("خروج مبكر (استمر الصعود بعد الخروج)")
        if peak - exit_pct >= 3:
            t.append("أعاد جزءاً كبيراً من الربح قبل الخروج")
    else:
        t.append("خاسرة")
        if x["why"] == "وقف" and since >= risk_pct:
            t.append("ضُرب الوقف ثم ارتفع السعر")
        if peak < 0.5 * risk_pct:
            t.append("لم تتحرك لصالحنا أبداً")
        elif peak >= risk_pct:
            t.append("وصلت ربحاً ثم انعكست")
    if x.get("btc24") is not None:
        t.append("بيتكوين هابط" if x["btc24"] <= -1 else "بيتكوين غير هابط")
    t.append("W4" if x.get("w4") else ("W3" if x.get("w3") else ("تطابق" if x["group"] == "match" else "مقارنة")))
    if x.get("buy_ratio") is not None:
        t.append("شراء غالب (≥55%)" if x["buy_ratio"] >= 0.55 else "بيع غالب (≤45%)" if x["buy_ratio"] <= 0.45 else "تدفق متوازن")
    return t


def update(journal, state):
    seen = {(e["sym"], e["signal_t"]) for e in journal}
    new = 0
    for x in state:
        if x["status"] == "closed" and (x["sym"], x["signal_t"]) not in seen and "peak_during" in x:
            entry = x["entry"]
            journal.append({"sym": x["sym"], "signal_t": x["signal_t"], "closed_t": x["closed_t"], "group": x["group"], "R": x["R"], "why": x["why"],
                            "exit_pct": (x["exit_px"] / entry - 1) * 100, "peak_pct": (x["peak_during"] / entry - 1) * 100,
                            "peak_since_pct": ((x["peak_since"] / entry - 1) * 100) if x.get("peak_since") else None,
                            "ctx": {k: x.get(k) for k in ("btc24", "funding", "oi_chg", "buy_ratio", "hot", "week", "pull")}, "tags": tags_for(x)})
            new += 1
    return new


def summary(journal):
    by = {}
    for e in journal:
        for tg in e["tags"]:
            by.setdefault(tg, []).append(e["R"])
    lines = []
    for tg, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {tg}: {len(rs)} صفقة · متوسط R {st.mean(rs):+.2f} · رابحة {sum(1 for r in rs if r > 0)}")
    return lines
