import json
import urllib.request
d = json.load(urllib.request.urlopen("https://contract.mexc.com/api/v1/contract/detail", timeout=60))["data"]
by = {x["symbol"]: x for x in d}
print("total contracts:", len(d))
print("keys:", sorted(d[0].keys()))
for s in ("BTC_USDT", "LUNC_USDT", "PHAROS_USDT", "SQQQ_USDT", "UVXY_USDT", "SILVER_USDT", "LEAD_USDT", "XPT_USDT", "EWY_USDT", "OPENAI_USDT", "PAXG_USDT", "TQQQ_USDT", "ANTHROPIC_USDT"):
    x = by.get(s)
    if x:
        print(s, {k: x.get(k) for k in ("type", "futureType", "baseCoinName", "indexOrigin", "category", "contractSize", "maxLeverage", "apiAllowed", "isNew", "isHot", "openingTime")})
