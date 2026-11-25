"""Search for the 'missing piece' for the channel's template. Samples: every 6th... one 1H close per coin per 6 hours (90 crypto futures, Aug 20 - now). Template outcome: stop -5%, TP ladder +3..+20%, 48h, 15m candles.
Targets: y1 = TP1 before the stop; y3 = TP3 before the stop; pos2 = per-position % (6 equal tranches, stop moves to entry after TP1, cost 0.2%).
Features (~45) are known at the 1H close. Time split: first 65% train, last 35% test. Reports single-feature lifts that hold in BOTH periods, a gradient-boosting model with a shuffled control, and a simple rule if one stands out."""
import json
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import scanner as S

now = int(time.time())
ETF = {"SOXL", "SOXS", "MUU", "MVLL", "DRAM", "SKHY", "QQQ", "SPY", "NVIDIA", "US30", "TESLA", "BITO", "SPCX", "USDC", "USDT", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USD1", "USDD", "BUSD", "EURC", "XUSD", "USDG"}
TPS = [3.0, 6.4, 9.8, 13.2, 16.6, 20.0]
STOP, COST = 5.0, 0.2
BTC = {}


def fetch(sym, itv, step, a, b_end):
    out = {k: [] for k in ("time", "open", "high", "low", "close", "vol")}
    while a < b_end:
        b = min(a + step * 1000, b_end)
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
    seen, keep = set(), []
    for i, t in enumerate(out["time"]):
        if t not in seen:
            seen.add(t); keep.append(i)
    return {k: np.array([out[k][i] for i in keep], float) for k in out}


def sma(x, n):
    c = np.cumsum(np.insert(x, 0, 0))
    r = np.full(len(x), np.nan)
    r[n - 1:] = (c[n:] - c[:-n]) / n
    return r


def pm(c):
    p, s = S.pmo(list(c))
    return np.array(p), np.array(s)


def outcome(price, j, hi, lo, cl, policy):
    stop_lvl, nxt, exits, hit1 = -STOP, 0, [None] * 6, False
    for k in range(j + 1, j + 193):
        if lo[k] <= price * (1 + stop_lvl / 100):
            for n in range(nxt, 6):
                exits[n] = stop_lvl
            return exits, hit1
        while nxt < 6 and hi[k] >= price * (1 + TPS[nxt] / 100):
            exits[nxt] = TPS[nxt]
            if nxt == 0:
                hit1 = True
                if policy == 2:
                    stop_lvl = 0.0
            nxt += 1
    for n in range(nxt, 6):
        exits[n] = (cl[j + 192] / price - 1) * 100
    return exits, hit1


def work(sym):
    start = int(datetime(2026, 8, 20, tzinfo=timezone.utc).timestamp())
    h1, m15 = fetch(sym, "Min60", 3600, start, now), fetch(sym, "Min15", 900, start, now)
    h4, d1 = fetch(sym, "Hour4", 14400, start - 40 * 86400, now), fetch(sym, "Day1", 86400, start - 200 * 86400, now)
    if h1 is None or m15 is None or h4 is None or d1 is None or len(h1["time"]) < 800 or len(m15["time"]) < 3000:
        return []
    t, o, hi, lo, c, v = (h1[k] for k in ("time", "open", "high", "low", "close", "vol"))
    n = len(c)
    P1, PS1 = pm(c)
    P4, PS4 = pm(h4["close"]); P1d, PS1d = pm(d1["close"])
    m = S.ema(list(c), 12); m2 = S.ema(list(c), 26); mm = np.array([a - b for a, b in zip(m, m2)]); gg = np.array(S.ema(list(mm), 9)); hist = mm - gg
    te = np.array(S.tema(list(c), 50))
    sm1 = {k: sma(c, k) for k in (5, 10, 30, 60)}
    c15, h15, l15 = m15["close"], m15["high"], m15["low"]
    s15 = {k: sma(c15, k) for k in (5, 10, 30, 60)}
    t15 = {int(x): i for i, x in enumerate(m15["time"])}
    rng_h = (hi / lo - 1) * 100
    rows, last = [], -99
    for i in range(700, n - 49):
        if i - last < 6:
            continue
        tc = int(t[i] + 3600)
        if tc + 48 * 3600 > now:
            break
        j = t15.get(tc - 900)
        if j is None or j + 192 >= len(c15):
            continue
        last = i
        price = c[i]
        k4 = np.searchsorted(h4["time"] + 14400, tc, side="right") - 1
        kd = np.searchsorted(d1["time"] + 86400, tc, side="right") - 1
        bi = BTC["idx"].get(int(t[i]))
        vm = np.median(v[i - 168:i]) or 1e-12
        rng = (hi[i] - lo[i]) or 1e-12
        f = {"ret1h": (c[i] / c[i - 1] - 1) * 100, "ret4h": (c[i] / c[i - 4] - 1) * 100, "ret12h": (c[i] / c[i - 12] - 1) * 100, "ret24h": (c[i] / c[i - 24] - 1) * 100, "ret7d": (c[i] / c[i - 168] - 1) * 100,
             "from7dhigh": (c[i] / hi[i - 168:i + 1].max() - 1) * 100, "from7dlow": (c[i] / lo[i - 168:i + 1].min() - 1) * 100,
             "pos24": (c[i] - lo[i - 24:i + 1].min()) / ((hi[i - 24:i + 1].max() - lo[i - 24:i + 1].min()) or 1e-12),
             "hr_range": rng_h[i], "atr_h": rng_h[i - 14:i + 1].mean(), "stop_over_atr": STOP / (rng_h[i - 14:i + 1].mean() or 1e-9),
             "vol1": v[i] / vm, "vol4": v[i - 3:i + 1].sum() / (4 * vm), "body": (c[i] / o[i] - 1) * 100, "upper_wick": (hi[i] - max(o[i], c[i])) / rng, "lower_wick": (min(o[i], c[i]) - lo[i]) / rng,
             "close_pos": (c[i] - lo[i]) / rng, "brk24": float(c[i] > hi[i - 24:i].max()), "brk48": float(c[i] > hi[i - 48:i].max()),
             "p_sma5": (price / sm1[5][i] - 1) * 100, "p_sma10": (price / sm1[10][i] - 1) * 100, "p_sma30": (price / sm1[30][i] - 1) * 100, "p_sma60": (price / sm1[60][i] - 1) * 100,
             "sma5_10": (sm1[5][i] / sm1[10][i] - 1) * 100, "sma10_30": (sm1[10][i] / sm1[30][i] - 1) * 100, "sma30_60": (sm1[30][i] / sm1[60][i] - 1) * 100,
             "m15_p_sma5": (c15[j] / s15[5][j] - 1) * 100, "m15_p_sma30": (c15[j] / s15[30][j] - 1) * 100, "m15_p_sma60": (c15[j] / s15[60][j] - 1) * 100, "m15_sma5_10": (s15[5][j] / s15[10][j] - 1) * 100,
             "pmo1": P1[i], "pmo1_gap": P1[i] - PS1[i], "pmo1_slope": P1[i] - P1[i - 1], "macd_hist": hist[i] / price * 100, "macd_slope": (hist[i] - hist[i - 1]) / price * 100, "tema_dist": (price / te[i] - 1) * 100,
             "pmo4": P4[k4], "pmo4_gap": P4[k4] - PS4[k4], "pmo4_slope": P4[k4] - P4[k4 - 1], "pmod": P1d[kd], "pmod_gap": P1d[kd] - PS1d[kd],
             "hour": float((datetime.fromtimestamp(tc, timezone.utc) + timedelta(hours=4)).hour), "wday": float((datetime.fromtimestamp(tc, timezone.utc) + timedelta(hours=4)).weekday())}
        if bi and bi > 30:
            f["btc4"] = (BTC["c"][bi] / BTC["c"][bi - 4] - 1) * 100; f["btc24"] = (BTC["c"][bi] / BTC["c"][bi - 24] - 1) * 100
        else:
            f["btc4"] = f["btc24"] = np.nan
        r1, h1f = outcome(price, j, h15, l15, c15, 1)
        r2, _ = outcome(price, j, h15, l15, c15, 2)
        f.update({"sym": sym, "t": tc, "y1": float(r1[0] == TPS[0]), "y3": float(r1[2] == TPS[2]), "y6": float(r1[5] == TPS[5]), "pos1": float(np.mean(r1) - COST), "pos2": float(np.mean(r2) - COST)})
        rows.append(f)
    return rows


detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
        and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in ETF]
b = fetch("BTC_USDT", "Min60", 3600, int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp()), now)
BTC["idx"] = {int(x): i for i, x in enumerate(b["time"])}; BTC["c"] = b["close"]
with ThreadPoolExecutor(4) as ex:
    rows = [r for part in ex.map(work, tick) for r in part]
df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
df.to_json("missing_piece_rows.json")
cut = df.t.quantile(0.65)
tr, te = df[df.t < cut - 48 * 3600].copy(), df[df.t > cut].copy()
feats = [c for c in df.columns if c not in ("sym", "t", "y1", "y3", "y6", "pos1", "pos2")]
print(f"{len(tick)} عملة · عينات: {len(df)} (تدريب {len(tr)} · اختبار {len(te)}) · مؤشرات: {len(feats)}")
for nm, d in (("تدريب", tr), ("اختبار", te)):
    print(f"الأساس ({nm}): TP1 قبل الوقف {d.y1.mean()*100:.0f}% · TP3 {d.y3.mean()*100:.0f}% · TP6 {d.y6.mean()*100:.1f}% · متوسط المركز (وقف ثابت) {d.pos1.mean():+.2f}% · (وقف للدخول بعد TP1) {d.pos2.mean():+.2f}%")
print("\n== مؤشرات واحدة: أعلى 20% وأدنى 20% (حدود من التدريب) — ما يصمد في الفترتين")
res = []
for f in feats:
    s = tr[f].dropna()
    if s.nunique() < 6:
        continue
    lo_q, hi_q = s.quantile([0.2, 0.8]).values
    for side, mtr, mte in (("أدنى 20%", tr[f] <= lo_q, te[f] <= lo_q), ("أعلى 20%", tr[f] >= hi_q, te[f] >= hi_q)):
        a, b2 = tr[mtr], te[mte]
        if len(a) < 300 or len(b2) < 150:
            continue
        res.append((f, side, len(a), a.pos2.mean() - tr.pos2.mean(), len(b2), b2.pos2.mean() - te.pos2.mean(), a.y1.mean() * 100, b2.y1.mean() * 100, a.pos2.mean(), b2.pos2.mean()))
r = pd.DataFrame(res, columns=["feat", "side", "ntr", "lift_tr", "nte", "lift_te", "tp1_tr", "tp1_te", "pos_tr", "pos_te"])
r["both"] = np.minimum(r.lift_tr, r.lift_te)
print(f"{'المؤشر':16s} {'الجهة':8s} {'عدد تدريب':>8s} {'عدد اختبار':>9s} {'متوسط المركز تدريب':>16s} {'اختبار':>8s} {'TP1 تدريب':>9s} {'اختبار':>7s}")
for _, x in r.sort_values("both", ascending=False).head(10).iterrows():
    print(f"{x.feat:16s} {x.side:8s} {int(x.ntr):8d} {int(x.nte):9d} {x.pos_tr:+15.2f}% {x.pos_te:+7.2f}% {x.tp1_tr:8.0f}% {x.tp1_te:6.0f}%")
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
print("\n== نموذج تعلّم آلي (كل المؤشرات) مقابل نموذج مخلوط عشوائياً")
for target in ("y1", "y3"):
    mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100, l2_regularization=5.0, random_state=0).fit(tr[feats], tr[target])
    p = mdl.predict_proba(te[feats])[:, 1]
    sel = te[p >= np.quantile(p, 0.8)]
    shuf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100, l2_regularization=5.0, random_state=0).fit(tr[feats], rng.permutation(tr[target].values))
    ps = shuf.predict_proba(te[feats])[:, 1]
    sels = te[ps >= np.quantile(ps, 0.8)]
    print(f"  {target}: AUC اختبار {roc_auc_score(te[target], p):.3f} (مخلوط {roc_auc_score(te[target], ps):.3f}) · أفضل 20% مرتبة: TP1 {sel.y1.mean()*100:.0f}% TP3 {sel.y3.mean()*100:.0f}% TP6 {sel.y6.mean()*100:.1f}% متوسط المركز {sel.pos2.mean():+.2f}% (الكل {te.pos2.mean():+.2f}%) · المخلوط: {sels.pos2.mean():+.2f}%")
reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100, l2_regularization=5.0, random_state=0).fit(tr[feats], tr.pos2.clip(-5, 20))
pr = reg.predict(te[feats])
for q in (0.8, 0.9):
    sel = te[pr >= np.quantile(pr, q)]
    print(f"  انحدار على متوسط المركز: أفضل {int((1-q)*100)}%: n={len(sel)} · TP1 {sel.y1.mean()*100:.0f}% · TP6 {sel.y6.mean()*100:.1f}% · متوسط المركز {sel.pos2.mean():+.2f}% ({sel.pos2.mean()*20:+.0f}% هامش) مقابل الكل {te.pos2.mean():+.2f}%")
from sklearn.inspection import permutation_importance
pi = permutation_importance(reg, te[feats], te.pos2.clip(-5, 20), n_repeats=3, random_state=0)
o = np.argsort(-pi.importances_mean)[:8]
print("  أهم المؤشرات (على الاختبار): " + ", ".join(f"{feats[k]} ({pi.importances_mean[k]:.3f})" for k in o))
