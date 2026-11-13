"""One-off probe of MEXC futures PUBLIC endpoints/websocket: which data exist, their fields, depth limits, history, and the rate limit.
Run on GitHub Actions (the sandbox cannot reach MEXC):  python tools/mexc_probe.py"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "https://contract.mexc.com"
SYM = "ICP_USDT"


def get(path: str, timeout: int = 15):
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE + path, headers={"User-Agent": "probe"}), timeout=timeout) as r:
            body = r.read().decode()
            return r.status, json.loads(body), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, None, time.time() - t0
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}", None, time.time() - t0


def short(x, n=260):
    s = json.dumps(x, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def section(title):
    print(f"\n=== {title} ===")


def main():
    section("ticker (all contracts)")
    st, d, dt = get("/api/v1/contract/ticker")
    rows = d.get("data") if isinstance(d, dict) else None
    print(st, f"{dt:.2f}s", "contracts:", len(rows or []))
    if rows:
        print("fields:", sorted(rows[0].keys()))
        print("sample:", short(next(r for r in rows if r.get("symbol") == SYM)))

    section("contract detail")
    st, d, dt = get(f"/api/v1/contract/detail?symbol={SYM}")
    print(st, f"{dt:.2f}s")
    if isinstance(d, dict) and d.get("data"):
        x = d["data"]
        print("fields:", sorted(x.keys()))
        for k in ("contractSize", "createTime", "openingTime", "limitMaxVol", "maxLeverage", "takerFeeRate", "makerFeeRate", "priceUnit", "volUnit", "minVol", "riskIncrVol", "indexOrigin", "futureType", "state"):
            print(f"  {k} = {x.get(k)}")

    section("order book (depth) — limits")
    for lim in (20, 100, 500, 1000, 2000, 5000):
        st, d, dt = get(f"/api/v1/contract/depth/{SYM}?limit={lim}")
        n = (len(d["data"]["bids"]), len(d["data"]["asks"])) if isinstance(d, dict) and d.get("data") else None
        print(f"limit={lim}: status={st} levels(bids,asks)={n} {dt:.2f}s")
    st, d, _ = get(f"/api/v1/contract/depth/{SYM}?limit=5")
    if isinstance(d, dict) and d.get("data"):
        print("level format sample:", short(d["data"]["bids"][:2]), "other keys:", [k for k in d["data"] if k not in ("bids", "asks")])

    section("recent trades (deals) — taker side")
    st, d, dt = get(f"/api/v1/contract/deals/{SYM}?limit=100")
    print(st, f"{dt:.2f}s")
    if isinstance(d, dict) and d.get("data"):
        print("count:", len(d["data"]), "sample:", short(d["data"][:2]))

    section("funding rate (current)")
    st, d, dt = get(f"/api/v1/contract/funding_rate/{SYM}")
    print(st, f"{dt:.2f}s", short(d.get("data") if isinstance(d, dict) else d))

    section("funding rate HISTORY")
    st, d, dt = get(f"/api/v1/contract/funding_rate/history?symbol={SYM}&page_num=1&page_size=100")
    print(st, f"{dt:.2f}s")
    if isinstance(d, dict) and d.get("data"):
        x = d["data"]
        rl = x.get("resultList") or []
        print("keys:", list(x.keys()), "· rows in page:", len(rl), "· totalCount:", x.get("totalCount"), "· sample:", short(rl[:1]))
        if rl:
            print("oldest/newest in page:", rl[-1].get("settleTime"), rl[0].get("settleTime"))

    section("klines (history depth and fields)")
    now = int(time.time())
    st, d, dt = get(f"/api/v1/contract/kline/{SYM}?interval=Min5&start={now - 3 * 86400}&end={now}")
    if isinstance(d, dict) and d.get("data"):
        x = d["data"]
        print(st, f"{dt:.2f}s", "fields:", sorted(x.keys()), "bars:", len(x.get("time", [])))
    st, d, dt = get(f"/api/v1/contract/kline/{SYM}?interval=Min15&start={now - 400 * 86400}&end={now}")
    if isinstance(d, dict) and d.get("data"):
        t = d["data"].get("time", [])
        print("Min15 requested 400 days -> bars:", len(t), "oldest:", time.strftime("%Y-%m-%d", time.gmtime(t[0])) if t else None)

    section("index / fair price / index kline")
    for name, p in (("index_price", f"/api/v1/contract/index_price/{SYM}"), ("fair_price", f"/api/v1/contract/fair_price/{SYM}"),
                    ("kline index_price", f"/api/v1/contract/kline/index_price/{SYM}?interval=Min15&start={now - 86400}&end={now}"),
                    ("kline fair_price", f"/api/v1/contract/kline/fair_price/{SYM}?interval=Min15&start={now - 86400}&end={now}")):
        st, d, dt = get(p)
        print(f"{name}: status={st} {dt:.2f}s", short(d.get("data") if isinstance(d, dict) else d, 150))

    section("guesses for dedicated open-interest / long-short / liquidation endpoints")
    for p in ("/api/v1/contract/open_interest", f"/api/v1/contract/open_interest/{SYM}", f"/api/v1/contract/openInterest/{SYM}",
              f"/api/v1/contract/long_short_ratio/{SYM}", f"/api/v1/contract/liquidation/{SYM}", f"/api/v1/contract/risk_reverse/{SYM}",
              f"/api/v1/contract/risk_reverse/history?symbol={SYM}&page_num=1&page_size=5", "/api/v1/contract/support_currencies",
              "/api/v1/contract/ping"):
        st, d, dt = get(p)
        print(f"{p}: status={st} {short(d, 140) if d else ''}")

    section("rate limit (30 quick ticker-by-symbol calls)")
    t0, ok, codes = time.time(), 0, {}
    for _ in range(30):
        st, d, _ = get(f"/api/v1/contract/ticker?symbol={SYM}")
        codes[st] = codes.get(st, 0) + 1
    print(f"30 calls in {time.time() - t0:.1f}s, status counts: {codes}")

    section("websocket")
    try:
        import websocket
    except ImportError:
        print("websocket-client not installed")
        return
    try:
        ws = websocket.create_connection("wss://contract.mexc.com/edge", timeout=10)
        subs = [{"method": "sub.ticker", "param": {"symbol": SYM}}, {"method": "sub.deal", "param": {"symbol": SYM}},
                {"method": "sub.depth", "param": {"symbol": SYM}}]
        for s in subs:
            ws.send(json.dumps(s))
        t0, seen = time.time(), {}
        while time.time() - t0 < 12:
            try:
                msg = json.loads(ws.recv())
            except Exception:
                break
            ch = msg.get("channel")
            if ch not in seen:
                seen[ch] = short(msg, 300)
        for k, v in seen.items():
            print(f"{k}: {v}")
        ws.close()
    except Exception as e:
        print("websocket failed:", type(e).__name__, e)


if __name__ == "__main__":
    main()
