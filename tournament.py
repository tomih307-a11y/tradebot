"""Tournament of 100 independent paper 'brains' (strategy genomes), each with its own compounding account.
Data: 180 days of 15m candles (crypto futures), daily/4H/1H PMO states (forming candle), BTC context.
Phase A (first 60% of time): all brains start with 1000$; a brain is eliminated at -30% from its peak (or <50% of capital).
Phase B (last 40%, never seen in selection): every brain restarts with 1000$ -> tells whether surviving phase A means anything.
Costs: fee 0.04% + slippage 0.05% per side (0.18% round trip), compounding (risk% of current equity / stop distance, leverage <= 5x), max open positions. No funding modelled."""
import bisect
import json
import random
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import scanner as S
import scanner_pmo as P

P.DAYS.update({"Min15": 180, "Min60": 130, "Hour4": 300, "Day1": 300})
ETF = P.ETF | {"USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}
A1, A2, AS = 2 / 35, 2 / 20, 2 / 11
COST = 0.18
STOPS = ("a15", "a25", "a35", "p6", "p10", "low24")
EXITS = ("E1", "T24", "TR")
BTC = {}


class FA:
    def __init__(self, d, per):
        c = d["close"]
        self.t, self.c, self.per = d["time"], c, per
        self.close_t = [x + per for x in self.t]
        roc = [0.0] + [(c[i] / c[i - 1] - 1) * 100 for i in range(1, len(c))]
        self.s1 = S.smooth(roc, 35)
        self.s2 = S.smooth([10 * v for v in self.s1], 20)
        self.sg = S.ema(self.s2, 10)

    def state(self, tc, price):
        k = bisect.bisect_right(self.close_t, tc) - 1
        if k < 150:
            return None
        roc = (price / self.c[k] - 1) * 100
        a = A1 * roc + (1 - A1) * self.s1[k]
        b = A2 * 10 * a + (1 - A2) * self.s2[k]
        sg = AS * b + (1 - AS) * self.sg[k]
        return {"pos": b > 0, "above": b > sg, "up": b > self.s2[k]}


def work(sym):
    now = int(time.time())
    k15, k1, k4, kd = P.kl(sym, "Min15", now), P.kl(sym, "Min60", now), P.kl(sym, "Hour4", now), P.kl(sym, "Day1", now)
    if not all((k15, k1, k4, kd)):
        return []
    f1, f4, fd = FA(k1, 3600), FA(k4, 14400), FA(kd, 86400)
    t, hi, lo, c = k15["time"], k15["high"], k15["low"], k15["close"]
    n = len(c)
    p, ps = S.pmo(c)
    m = [a - b for a, b in zip(S.ema(c, 12), S.ema(c, 26))]
    g = S.ema(m, 9)
    tr = [hi[k] - lo[k] if k == 0 else max(hi[k] - lo[k], abs(hi[k] - c[k - 1]), abs(lo[k] - c[k - 1])) for k in range(n)]
    atr = S.ema(tr, 14)
    out, last_t, last_e = [], -99, -99
    for i in range(700, n - 100):
        touch = p[i - 1] <= 0 < p[i]
        early = p[i - 1] <= ps[i - 1] and p[i] > ps[i] and (any(m[k - 1] <= g[k - 1] and m[k] > g[k] for k in range(i - 2, i + 1)))
        sel_t = touch and i - last_t >= 4
        sel_e = early and not touch and i - last_e >= 8
        if touch: last_t = i
        if early: last_e = i
        if not (sel_t or sel_e):
            continue
        ent, tc = c[i], t[i] + 900
        d, h4, h1 = fd.state(tc, ent), f4.state(tc, ent), f1.state(tc, ent)
        bi = BTC["idx"].get(t[i])
        btc24 = (BTC["c"][bi] / BTC["c"][bi - 96] - 1) * 100 if bi and bi > 100 else None
        sig = {"t": t[i], "sym": sym, "kind": "touch" if sel_t else "early", "d1": bool(d and d["pos"]), "h4": bool(h4 and h4["pos"]), "h1": bool(h1 and h1["pos"]),
               "hot": atr[i] / (st.median(atr[max(0, i - 960):i]) or 1e-12), "week": ent / c[i - 672] - 1 > 0, "pull": (ent / max(hi[i - 95:i + 1]) - 1) * 100, "btc24": btc24}
        stops = {"a15": ent - 1.5 * atr[i], "a25": ent - 2.5 * atr[i], "a35": ent - 3.5 * atr[i], "p6": ent * 0.94, "p10": ent * 0.90, "low24": min(lo[i - 95:i + 1]) * 0.998}
        res = {}
        for sn, stp0 in stops.items():
            stp0 = max(min(stp0, ent * 0.995), ent * 0.85)
            dist = (1 - stp0 / ent) * 100
            o = {}
            hh = ent
            for j in range(i + 1, i + 97):
                for ex in EXITS:
                    if ex in o:
                        continue
                    lvl = max(stp0, hh - 2 * atr[i]) if ex == "TR" else stp0
                    if lo[j] <= lvl:
                        o[ex] = ((lvl / ent - 1) * 100 - COST, j - i)
                    elif ex == "E1" and p[j] < ps[j] and p[j - 1] >= ps[j - 1]:
                        o[ex] = ((c[j] / ent - 1) * 100 - COST, j - i)
                hh = max(hh, hi[j])
                if len(o) == 3:
                    break
            for ex in EXITS:
                if ex not in o:
                    o[ex] = ((c[i + 96] / ent - 1) * 100 - COST, 96)
            for ex in EXITS:
                res[f"{sn}|{ex}"] = (o[ex][0], o[ex][1], dist)
        sig["res"] = res
        out.append(sig)
    return out


def make_genomes(seed=11):
    rnd = random.Random(seed)
    G = [{"name": "G0 تطابق PMO (اليومي فوق الصفر)", "kind": "touch", "d1": True, "h4": None, "h1": None, "hot": None, "week": None, "pull": None, "btc": None, "stop": "a15", "exit": "E1", "risk": 1, "maxopen": 5},
         {"name": "G1 W3 (متقلبة+صاعدة أسبوعاً)", "kind": "touch", "d1": True, "h4": None, "h1": None, "hot": 1.2, "week": True, "pull": None, "btc": None, "stop": "a15", "exit": "E1", "risk": 1, "maxopen": 5},
         {"name": "G2 W4 (الموجة الثانية)", "kind": "touch", "d1": True, "h4": None, "h1": None, "hot": 1.2, "week": True, "pull": -6, "btc": None, "stop": "a15", "exit": "E1", "risk": 1, "maxopen": 5}]
    while len(G) < 100:
        G.append({"name": f"G{len(G)}", "kind": rnd.choice(["touch", "touch", "early"]), "d1": rnd.choice([True, True, None]), "h4": rnd.choice([True, None, None]),
                  "h1": rnd.choice([True, None, None]), "hot": rnd.choice([None, 1.0, 1.2, 1.5]), "week": rnd.choice([True, None]), "pull": rnd.choice([None, -3, -6, -10]),
                  "btc": rnd.choice([None, None, "down", "notdown"]), "stop": rnd.choice(STOPS), "exit": rnd.choice(EXITS), "risk": rnd.choice([1, 2, 3, 5]), "maxopen": rnd.choice([3, 5, 8])})
    return G


def desc(g):
    parts = [g["kind"]]
    for k, lab in (("d1", "يومي+"), ("h4", "4س+"), ("h1", "1س+")):
        if g[k]: parts.append(lab)
    if g["hot"]: parts.append(f"ATR≥{g['hot']}")
    if g["week"]: parts.append("أسبوع↑")
    if g["pull"]: parts.append(f"تراجع≤{g['pull']}%")
    if g["btc"]: parts.append("BTC هابط" if g["btc"] == "down" else "BTC غير هابط")
    parts += [f"وقف {g['stop']}", f"خروج {g['exit']}", f"مخاطرة {g['risk']}%", f"≤{g['maxopen']} مفتوحة"]
    return " ".join(parts)


def passes(g, s):
    if s["kind"] != g["kind"]: return False
    if g["d1"] and not s["d1"]: return False
    if g["h4"] and not s["h4"]: return False
    if g["h1"] and not s["h1"]: return False
    if g["hot"] and s["hot"] < g["hot"]: return False
    if g["week"] and not s["week"]: return False
    if g["pull"] and s["pull"] > g["pull"]: return False
    if g["btc"]:
        if s["btc24"] is None: return False
        if (g["btc"] == "down") != (s["btc24"] <= -1): return False
    return True


def run_brain(g, sigs, t0):
    eq, peak, maxdd, alive, n = 1000.0, 1000.0, 0.0, True, 0
    openp = []                                                      # (close_ts, notional, pct, sym)
    taken = set()
    key = f"{g['stop']}|{g['exit']}"
    for s in sigs:
        if not passes(g, s):
            continue
        still = []
        for cts, notional, pct, sym in openp:
            if cts <= s["t"]:
                eq += notional * pct / 100
                peak = max(peak, eq); maxdd = min(maxdd, eq / peak - 1)
                if eq <= 0.7 * peak or eq <= 500:
                    alive = False
            else:
                still.append((cts, notional, pct, sym))
        openp = still
        if not alive:
            break
        if len(openp) >= g["maxopen"] or any(o[3] == s["sym"] for o in openp):
            continue
        pct, hold, dist = s["res"][key]
        notional = min(eq * g["risk"] / 100 / (dist / 100), eq * 5)
        openp.append((s["t"] + hold * 900, notional, pct, s["sym"]))
        n += 1
    if alive:
        for cts, notional, pct, sym in openp:
            eq += notional * pct / 100
            peak = max(peak, eq); maxdd = min(maxdd, eq / peak - 1)
        alive = eq > 0.7 * peak and eq > 500
    return {"eq": eq, "n": n, "maxdd": maxdd, "alive": alive}


def main():
    now = int(time.time())
    detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
    b = P.kl("BTC_USDT", "Min15", now)
    BTC["idx"] = {tt: i for i, tt in enumerate(b["time"])}; BTC["c"] = b["close"]
    with ThreadPoolExecutor(4) as ex:
        sigs = sorted((s for part in ex.map(work, tick) for s in part), key=lambda s: s["t"])
    t_min, t_max = sigs[0]["t"], sigs[-1]["t"]
    cut = t_min + 0.6 * (t_max - t_min)
    A = [s for s in sigs if s["t"] < cut]; B = [s for s in sigs if s["t"] >= cut]
    f = lambda x: datetime.fromtimestamp(x, timezone.utc).strftime("%Y-%m-%d")
    print(f"{len(tick)} عملة · {len(sigs)} إشارة · المرحلة A (اختيار): {f(t_min)} → {f(cut)} ({len(A)} إشارة) · المرحلة B (اختبار لم يره أحد): {f(cut)} → {f(t_max)} ({len(B)} إشارة)")
    G = make_genomes()
    ra = [run_brain(g, A, t_min) for g in G]
    rb = [run_brain(g, B, cut) for g in G]
    alive = [i for i, r in enumerate(ra) if r["alive"] and r["n"] >= 20]
    dead = [i for i, r in enumerate(ra) if not r["alive"]]
    inactive = [i for i, r in enumerate(ra) if r["alive"] and r["n"] < 20]
    print(f"\\nالمرحلة A: ناجون (≥20 صفقة) {len(alive)} · أُقصوا {len(dead)} · غير نشطين (<20 صفقة) {len(inactive)}")
    top = sorted(alive, key=lambda i: -ra[i]["eq"])[:10]
    print("أفضل 10 في A: الرصيد النهائي (من 1000$) · صفقات · أقصى هبوط · نتيجتهم في المرحلة B (رصيد جديد 1000$)")
    for i in top:
        print(f"  {G[i]['name']}: A {ra[i]['eq']:.0f}$ ({ra[i]['n']} صفقة، هبوط {ra[i]['maxdd']*100:.0f}%) → B {rb[i]['eq']:.0f}$ ({rb[i]['n']} صفقة، {'أُقصي' if not rb[i]['alive'] else 'حي'}) | {desc(G[i])}")
    def grp(idx, nm):
        if not idx:
            print(f"  {nm}: لا أحد"); return
        v = np.array([rb[i]["eq"] for i in idx if rb[i]["n"] >= 10])
        if not len(v):
            print(f"  {nm}: لا أحد نشط في B"); return
        print(f"  {nm}: {len(v)} نشط في B · وسيط الرصيد {np.median(v):.0f}$ · متوسط {v.mean():.0f}$ · رابحون {np.mean(v>1000)*100:.0f}% · أُقصوا {np.mean([not rb[i]['alive'] for i in idx if rb[i]['n']>=10])*100:.0f}% · أفضل {v.max():.0f}$ · أسوأ {v.min():.0f}$")
    print("\\nهل الاختيار في A أفاد في B؟ (كل العقول بدأت B برأس مال جديد 1000$)")
    grp(alive, "ناجون في A")
    grp(dead, "مُقصَون في A")
    grp(top, "أفضل 10 في A")
    grp(list(range(100)), "كل العقول")
    hof = [i for i in alive if rb[i]["alive"] and rb[i]["n"] >= 20 and rb[i]["eq"] > 1000 and ra[i]["eq"] > 1000]
    print(f"\\nقاعة المشاهير (نجت في A وربحت في B بـ20+ صفقة لكل مرحلة): {len(hof)} من 100")
    for i in sorted(hof, key=lambda i: -rb[i]["eq"])[:8]:
        print(f"  {G[i]['name']}: A {ra[i]['eq']:.0f}$ · B {rb[i]['eq']:.0f}$ ({rb[i]['n']} صفقة، هبوط {rb[i]['maxdd']*100:.0f}%) | {desc(G[i])}")
    best = max(max(ra[i]["eq"] for i in range(100)), max(rb[i]["eq"] for i in range(100)))
    print(f"\\nأعلى رصيد وصل إليه أي عقل في أي مرحلة: {best:.0f}$ (الهدف 50000$)")
    json.dump({"G": G, "A": ra, "B": rb}, open("tournament_result.json", "w"))


if __name__ == "__main__":
    main()
