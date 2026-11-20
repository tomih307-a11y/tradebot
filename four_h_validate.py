"""Validation of the 4H-touch signal (S3) with X2 exit: vs random-entry baseline with the same exit, by month, week-block bootstrap, outlier trimming, coin concentration."""
import numpy as np
import four_h_test as T
from concurrent.futures import ThreadPoolExecutor
import scanner as S
import pmo_only as P
from datetime import datetime, timezone

detail = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
        and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and not x["symbol"][:-5].endswith("STOCK") and x["symbol"][:-5] not in P.ETF]
with ThreadPoolExecutor(4) as ex:
    rows = [r for part in ex.map(T.work, tick) for r in part]
rng = np.random.default_rng(0)


def stats(g, key="X2R", pk="X2"):
    xr = np.array([r[key] for r in g]); xp = np.array([r[pk] for r in g])
    wk = {}
    for r in g:
        wk.setdefault(r["t"] // (7 * 86400), []).append(r[key])
    ks = list(wk)
    sm = np.array([sum(wk[x]) for x in ks]); cn = np.array([len(wk[x]) for x in ks])
    bs = [sm[ix].sum() / cn[ix].sum() for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(500))]
    return f"n={len(g):5d} متوسط {xp.mean():+.3f}% / {xr.mean():+.3f}R | حدّ الثقة (كتل أسبوعية) [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}] | رابحة {np.mean(xp>0)*100:.0f}% | وسيط {np.median(xp):+.2f}%"


S3 = [r for r in rows if r["sig"]["S3"]]
BASE = [r for r in rows if r["sig"]["BASE"]]
print("== X2: إشارة لمس 4س (S3) مقابل دخول عشوائي كل 4 ساعات بنفس الخروج")
print("لمس 4س :", stats(S3))
print("عشوائي :", stats(BASE))
print("\n== لمس 4س حسب الشهر")
bym = {}
for r in S3:
    bym.setdefault(datetime.fromtimestamp(r["t"], timezone.utc).strftime("%Y-%m"), []).append(r)
for k in sorted(bym):
    print(f"{k}: {stats(bym[k])}")
print("\n== حساسية للقيم الشاذة")
xp = np.sort(np.array([r["X2"] for r in S3]))
print(f"بعد حذف أعلى 1% صفقة: متوسط {xp[: int(len(xp)*0.99)].mean():+.3f}% | بعد حذف أعلى 5%: {xp[: int(len(xp)*0.95)].mean():+.3f}% | أعلى 5 صفقات: {', '.join(f'{v:+.1f}%' for v in xp[-5:])}")
print("\n== تركّز العملات (مساهمة كل عملة في مجموع %، أعلى 8)")
bc = {}
for r in S3:
    bc[r["sym"]] = bc.get(r["sym"], 0) + r["X2"]
tot = sum(bc.values())
for k, v in sorted(bc.items(), key=lambda kv: -kv[1])[:8]:
    print(f"{k[:-5]:10s} {v:+8.1f}% ({v/tot*100:.0f}% من المجموع)")
half = np.median([r["t"] for r in S3])
print("\nالنصف الأول :", stats([r for r in S3 if r["t"] < half]))
print("النصف الثاني:", stats([r for r in S3 if r["t"] >= half]))
