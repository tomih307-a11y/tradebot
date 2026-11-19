"""Pick 6 winners (+3% before -3% within 12h) and 6 losers (-3% before +3%) among 15m PMO zero-touches that passed ALL agreed conditions
(strict larger frames + 4H PMO rising + TEMA50 + MACD15 + near/inside bullish OB), crypto only, max one case per coin, random with fixed seed.
Output lines: SYMBOL:MM-DD:HH:MM (UTC+4) | outcome."""
import random
from concurrent.futures import ThreadPoolExecutor

import liquidity_test2 as T
import scanner as S

EXCL = {"NVIDIA", "US30", "TESLA", "BTC", "ETH"}


def main():
    detail = {d["symbol"]: d for d in S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]}
    tick = [x["symbol"] for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"] if x["symbol"].endswith("_USDT") and x["symbol"] in detail
            and S.is_crypto(detail[x["symbol"]]) and float(x.get("amount24") or 0) >= 1_000_000 and T.ok_name(x["symbol"]) and x["symbol"][:-5] not in EXCL]
    with ThreadPoolExecutor(6) as ex:
        rows = [r for part in ex.map(T.work, tick) for r in part]
    ok = [r for r in rows if r["fwd"] and r["pass2"] and r["x"]["tema"] and r["x"]["m15"] and r["x"]["ob_near"]]
    rnd = random.Random(7)
    out = []
    for kind, name in ((1, "رابحة"), (-1, "خاسرة")):
        pool = [r for r in ok if r["first"] == kind]
        rnd.shuffle(pool)
        seen, pick = set(), []
        for r in pool:
            if r["sym"] not in seen:
                seen.add(r["sym"]); pick.append(r)
            if len(pick) == 6:
                break
        out += [(r, name) for r in pick]
    rnd.shuffle(out)
    print(f"مجموعة مرشحة: {len(ok)} لمسة")
    for n, (r, name) in enumerate(out, 1):
        print(f"{n}|{r['sym']}:{T.dt(r['t']):%m-%d}:{T.dt(r['t']):%H:%M}|{name}|صعود {r['up']:+.2f}% هبوط {r['dn']:+.2f}%")


main()
