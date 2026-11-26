"""Backtest of the channel's common trait: volume spike + fast up move (15m), entered at the NEXT 5m open (+0.1% slippage), 0.2% round-trip cost.
Triggers: A = vol(15m) >= 4x usual and 15m return >= +2%; B = >= 10x and >= +3%. Cooldown 1h per coin. Futures 5m candles, last 45 days.
Templates: SC = scalp (stop -4%, TP +3/6/9/12/15) ; LD = ladder (stop -5%, TP +3/6.4/9.8/13.2/16.6/20). 48h horizon, equal tranches, stop checked before TP inside a candle.
Control: random 5m moments of the same coins (3 per trigger). Reports any-TP rate, TP1-before-stop, mean position, CI by week-block bootstrap, halves, concentration."""
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scanner as S

now = int(time.time())
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX"}
DAYS = 45
SC = (4.0, [3, 6, 9, 12, 15]); LD = (5.0, [3, 6.4, 9.8, 13.2, 16.6, 20])
rng = np.random.default_rng(1)


def fetch(sym):
    a, out = now - DAYS * 86400, {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
    while a < now:
        b = min(a + 300 * 1000, now)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min5&start={a}&end={b}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            return None
        for k in out:
            out[k] += d[k]
        a = b
    seen, keep = set(), []
    for i, t in enumerate(out["time"]):
        if t not in seen:
            seen.add(t); keep.append(i)
    return {k: np.array([out[k][i] for i in keep], float) for k in out}


def outcome(d, i, tpl):
    stop_pct, tps = tpl
    e = d["open"][i + 1] * 1.001
    sl = e * (1 - stop_pct / 100)
    nxt, ex = 0, [None] * len(tps)
    end = min(len(d["time"]) - 1, i + 1 + 576)
    if i + 1 + 576 >= len(d["time"]):
        return None
    for k in range(i + 1, end + 1):
        if d["low"][k] <= sl:
            for n in range(nxt, len(tps)):
                ex[n] = -stop_pct
            return ex, nxt
        while nxt < len(tps) and d["high"][k] >= e * (1 + tps[nxt] / 100):
            ex[nxt] = tps[nxt]; nxt += 1
    for n in range(nxt, len(tps)):
        ex[n] = (d["close"][end] / e - 1) * 100
    return ex, nxt


def work(sym):
    d = fetch(sym)
    if d is None or len(d["time"]) < 3000:
        return []
    c, v, h, l = d["close"], d["vol"], d["high"], d["low"]
    n = len(c)
    rows, last = [], -99
    q3 = v[2:] + v[1:-1] + v[:-2]
    for i in range(300, n - 600):
        base = np.median(v[i - 288:i - 2]) * 3 or 1e-9
        ratio = (v[i] + v[i - 1] + v[i - 2]) / base
        ret = (c[i] / c[i - 3] - 1) * 100
        tier = "B" if (ratio >= 10 and ret >= 3) else ("A" if (ratio >= 4 and ret >= 2) else None)
        if tier and i - last >= 12:
            last = i
            rows.append((sym, int(d["time"][i]), tier, i))
    out = []
    for sym_, t, tier, i in rows:
        a, b = outcome(d, i, SC), outcome(d, i, LD)
        if a and b:
            out.append((sym_, t, tier, a, b, "T"))
    # controls: random moments
    for _ in range(3 * len(rows)):
        i = int(rng.integers(300, n - 600))
        a, b = outcome(d, i, SC), outcome(d, i, LD)
        if a and b:
            out.append((sym, int(d["time"][i]), "-", a, b, "C"))
    return out


detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
        and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
with ThreadPoolExecutor(6) as ex:
    res = [r for part in ex.map(work, tick) for r in part]
print(f"عملات: {len(tick)} · لحظات: {len(res)}")


def pos(o, tpl):
    ex_, _ = o
    return float(np.mean(ex_)) - 0.2


def summary(sel, tpl_idx, label):
    if len(sel) < 20:
        print(f"  {label}: عدد قليل ({len(sel)})"); return
    o = [r[3 + tpl_idx] for r in sel]
    tpl = SC if tpl_idx == 0 else LD
    p = np.array([pos(x, tpl) for x in o])
    anytp = np.mean([x[1] >= 1 for x in o]) * 100
    tp3 = np.mean([x[1] >= 3 for x in o]) * 100
    wk = np.array([r[1] // (7 * 86400) for r in sel])
    ws = np.unique(wk)
    bs = [np.concatenate([p[wk == w] for w in rng.choice(ws, len(ws))]).mean() for _ in range(1000)]
    print(f"  {label}: n={len(sel)} · TP1 قبل الوقف {anytp:.0f}% · TP3 {tp3:.0f}% · متوسط المركز {p.mean():+.2f}% (CI {np.percentile(bs, 2.5):+.2f}..{np.percentile(bs, 97.5):+.2f}) · موجب {np.mean(p > 0)*100:.0f}%")
    return p.mean()


for tag, tiers in (("A+B (حجم ≥4× وعائد ≥2%)", ("A", "B")), ("B فقط (≥10× و≥3%)", ("B",))):
    trig = [r for r in res if r[5] == "T" and r[2] in tiers]
    ctrl = [r for r in res if r[5] == "C"]
    print(f"\n== {tag} · {len(trig)} إشارة (≈{len(trig)/DAYS:.1f}/يوم)")
    for ti, nm in ((0, "سكالب (وقف -4%)"), (1, "سلم (وقف -5%)")):
        print(f" {nm}")
        summary(trig, ti, "الإشارات")
        summary(ctrl, ti, "عشوائي ")
        ts = sorted(set(r[1] for r in trig)); mid = ts[len(ts) // 2]
        summary([r for r in trig if r[1] < mid], ti, "الإشارات النصف الأول")
        summary([r for r in trig if r[1] >= mid], ti, "الإشارات النصف الثاني")
        cnt = {}
        for r in trig:
            cnt[r[0]] = cnt.get(r[0], 0) + 1
        top = {s for s, _ in sorted(cnt.items(), key=lambda x: -x[1])[:5]}
        summary([r for r in trig if r[0] not in top], ti, "بدون أكثر 5 عملات تكراراً")
