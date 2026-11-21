"""Reverse study of explosive moves. Event = 15m bar from which price rises >= 25% (max high) within the next 24h, first bar of each episode (cooldown 24h).
Control = random bars in the same coins >= 24h away from any event whose next-24h max rise < 5%. Features use only the past (known at that bar).
Output: AUC of each feature (event vs control), medians, and how often our PMO zero-touch preceded events (recall) / how often our match signal is followed by an event (precision)."""
import json
import math
import random
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import scanner as S
import scanner_pmo as P

P.DAYS.update({"Min15": 180, "Min60": 130, "Hour4": 300, "Day1": 300})
ETF = P.ETF | {"USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}
NAN = float("nan")
BTC = {}


def f_frame(d, per, tc, price, tag, row):
    st_ = P.fstate(d, per, tc, price)
    row[f"{tag}_pos"] = NAN if not st_ else float(st_["pos"])
    row[f"{tag}_above"] = NAN if not st_ else float(st_["above"])
    row[f"{tag}_up"] = NAN if not st_ else float(st_["up"])
    row[f"{tag}_p"] = NAN if not st_ else st_["p"]


def work(sym):
    now = int(time.time())
    k15, k1, k4, kd = P.kl(sym, "Min15", now), P.kl(sym, "Min60", now), P.kl(sym, "Hour4", now), P.kl(sym, "Day1", now)
    if not all((k15, k1, k4, kd)):
        return [], []
    t, hi, lo, c, vo = k15["time"], k15["high"], k15["low"], k15["close"], None
    n = len(c)
    p, ps = S.pmo(c)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    ev, ctl = [], []
    rnd = random.Random(hash(sym) % 1000)
    last_ev = -999
    event_idx = []
    for i in range(700, n - 100):
        mx = max(hi[i + 1:i + 97]) / c[i] - 1
        if mx >= 0.25 and i - last_ev >= 96:
            event_idx.append(i); last_ev = i
    blocked = set()
    for e in event_idx:
        for k in range(e - 96, e + 97):
            blocked.add(k)
    cands = [i for i in range(700, n - 100) if i not in blocked and max(hi[i + 1:i + 97]) / c[i] - 1 < 0.05]
    ctrl_idx = rnd.sample(cands, min(len(cands), max(3 * len(event_idx), 12)))
    for lab, idxs in ((1, event_idx), (0, ctrl_idx)):
        for i in idxs:
            ent, tc = c[i], t[i] + 900
            row = {"sym": sym, "t": t[i], "event": lab}
            atr_med = st.median(atr[max(0, i - 960):i]) or 1e-12
            row["atr_pct"] = atr[i] / ent * 100
            row["atr_ratio"] = atr[i] / atr_med
            rng24 = max(hi[i - 95:i + 1]) / min(lo[i - 95:i + 1]) - 1
            rng7d = max(hi[i - 671:i + 1]) / min(lo[i - 671:i + 1]) - 1
            row["range24h_pct"] = rng24 * 100
            row["range_ratio"] = rng24 / (rng7d or 1e-9)
            row["ret1h"], row["ret4h"], row["ret12h"], row["ret24h"], row["ret7d"] = ((ent / c[i - b] - 1) * 100 for b in (4, 16, 48, 96, 672))
            row["dd24h"] = (ent / max(hi[i - 95:i + 1]) - 1) * 100
            row["dist_7d_high"] = (ent / max(hi[i - 671:i + 1]) - 1) * 100
            row["pos7d"] = (ent - min(lo[i - 671:i + 1])) / ((max(hi[i - 671:i + 1]) - min(lo[i - 671:i + 1])) or 1e-9)
            row["pmo15"], row["pmo15_gap"], row["pmo15_slope"] = p[i], p[i] - ps[i], p[i] - p[i - 1]
            row["touch_recent8"] = float(any(p[k - 1] <= 0 < p[k] for k in range(i - 7, i + 1)))
            row["touch_recent32"] = float(any(p[k - 1] <= 0 < p[k] for k in range(i - 31, i + 1)))
            f_frame(k1, 3600, tc, ent, "h1", row); f_frame(k4, 14400, tc, ent, "h4", row); f_frame(kd, 86400, tc, ent, "d1", row)
            bi = BTC["idx"].get(t[i])
            if bi and bi > 100:
                row["btc_ret24h"] = (BTC["c"][bi] / BTC["c"][bi - 96] - 1) * 100
                row["btc_ret4h"] = (BTC["c"][bi] / BTC["c"][bi - 16] - 1) * 100
            else:
                row["btc_ret24h"] = row["btc_ret4h"] = NAN
            row["rel24"] = row["ret24h"] - row["btc_ret24h"] if row["btc_ret24h"] == row["btc_ret24h"] else NAN
            row["hour"] = float((datetime.fromtimestamp(t[i], timezone.utc) + timedelta(hours=4)).hour)
            (ev if lab else ctl).append(row)
    # recall / precision helpers
    sig = [i for i in range(700, n - 100) if p[i - 1] <= 0 < p[i]]
    d1 = []
    for i in sig:
        s = P.fstate(kd, 86400, t[i] + 900, c[i])
        d1.append(bool(s and s["pos"]))
    ev_after = [max(hi[i + 1:i + 97]) / c[i] - 1 >= 0.25 for i in sig]
    stats = {"sym": sym, "events": len(event_idx), "bars": n - 800, "touches": len(sig), "match": sum(d1),
             "ev_after_touch": sum(ev_after), "ev_after_match": sum(1 for a, b in zip(ev_after, d1) if a and b),
             "ev_with_touch8": sum(1 for r in ev if r["touch_recent8"]), "ev_with_touch32": sum(1 for r in ev if r["touch_recent32"])}
    return ev + ctl, [stats]


def auc(a, b):
    a = np.array([x for x in a if x == x]); b = np.array([x for x in b if x == x])
    if len(a) < 20 or len(b) < 20:
        return None
    allv = np.concatenate([a, b]); ranks = allv.argsort().argsort() + 1
    # average ranks for ties
    order = allv.argsort(); sorted_v = allv[order]
    r = np.empty(len(allv)); i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def main():
    now = int(time.time())
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    b = P.kl("BTC_USDT", "Min15", now)
    BTC["idx"] = {tt: i for i, tt in enumerate(b["time"])}; BTC["c"] = b["close"]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(work, tick))
    rows = [r for rr, _ in res for r in rr]
    stats = [s for _, ss in res for s in ss]
    ev = [r for r in rows if r["event"] == 1]; ct = [r for r in rows if r["event"] == 0]
    print(f"{len(tick)} عملة · 180 يوماً · أحداث انفجار (+25% خلال 24س): {len(ev)} · لحظات ضابطة: {len(ct)}")
    feats = [k for k in rows[0] if k not in ("sym", "t", "event")]
    out = []
    for f in feats:
        a = auc([r[f] for r in ev], [r[f] for r in ct])
        if a is not None:
            ea = [r[f] for r in ev if r[f] == r[f]]; ca = [r[f] for r in ct if r[f] == r[f]]
            out.append((f, a, st.median(ea), st.median(ca)))
    out.sort(key=lambda r: -abs(r[1] - 0.5))
    print("\\nالمؤشر · AUC (0.5 = لا فرق، >0.5 أعلى قبل الانفجار) · وسيط قبل الانفجار · وسيط الضابط")
    for f, a, me, mc in out[:22]:
        print(f"  {f:16s} AUC {a:.3f} | قبل الانفجار {me:+9.3f} | ضابط {mc:+9.3f}")
    tot = {k: sum(s[k] for s in stats) for k in ("events", "touches", "match", "ev_after_touch", "ev_after_match", "ev_with_touch8", "ev_with_touch32")}
    print(f"\\nالتغطية: من {tot['events']} انفجاراً، سبقها لمس صفر 15د خلال ساعتين: {tot['ev_with_touch8']} ({tot['ev_with_touch8']/tot['events']*100:.0f}%) · خلال 8 ساعات: {tot['ev_with_touch32']} ({tot['ev_with_touch32']/tot['events']*100:.0f}%)")
    print(f"الدقة: من {tot['touches']} لمسة صفر، تبعها انفجار +25%: {tot['ev_after_touch']} ({tot['ev_after_touch']/tot['touches']*100:.1f}%) · من {tot['match']} لمسة مطابقة (اليومي فوق الصفر): {tot['ev_after_match']} ({tot['ev_after_match']/tot['match']*100:.1f}%)")
    base = tot['events'] / sum(s['bars'] for s in stats) * 100
    print(f"النسبة الأساسية: انفجار +25% يبدأ في {base:.2f}% من كل شموع 15د")
    json.dump(rows, open("event_rows.json", "w"))


if __name__ == "__main__":
    main()
