import scanner as S
detail = S.get("https://contract.mexc.com/api/v1/contract/detail")["data"]
tk = {x["symbol"]: x for x in S.get("https://contract.mexc.com/api/v1/contract/ticker")["data"]}
keys = ("SPY", "SP500", "S&P", "US500", "SPX", "VOO", "IVV", "ES_", "NAS", "QQQ", "US30", "DOW", "NDX", "INDEX")
for x in detail:
    s = x["symbol"]
    if any(k in s.upper() for k in keys) or any(k in (x.get("baseCoinName") or "").upper() for k in ("SPY", "S&P", "SP500", "500")):
        t = tk.get(s, {})
        print(s, "|", x.get("baseCoinName"), "|", x.get("indexOrigin"), "| last", t.get("lastPrice"), "| 24h", round(float(t.get("riseFallRate") or 0) * 100, 2), "% | turnover", round(float(t.get("amount24") or 0) / 1e6, 2), "M")
