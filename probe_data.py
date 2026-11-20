"""Probe which public derivatives-flow data can be reached from the runner, and how much history it gives."""
import json
import time
import urllib.request

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=25)
        return r.status, r.read().decode()[:3000]
    except Exception as e:
        code = getattr(e, "code", None)
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:
            pass
        return code or "ERR", f"{type(e).__name__} {e} {body}"

now = int(time.time() * 1000)
tests = {
 "Binance futures: taker buy/sell ratio 5m": "https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=5m&limit=500",
 "Binance futures: open interest history 5m": "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=500",
 "Binance futures: funding history": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=5",
 "Binance futures: klines (has taker buy volume)": "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=3",
 "Binance futures: forced liquidations": "https://fapi.binance.com/fapi/v1/allForceOrders?symbol=BTCUSDT&limit=5",
 "Bybit: open interest 15min": "https://api.bybit.com/v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=15min&limit=200",
 "Bybit: long/short ratio 15min": "https://api.bybit.com/v5/market/account-ratio?category=linear&symbol=BTCUSDT&period=15min&limit=200",
 "Bybit: klines": "https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=15&limit=3",
 "OKX: open interest history": "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy=BTC&period=5m",
 "OKX: taker volume": "https://www.okx.com/api/v5/rubik/stat/taker-volume?ccy=BTC&instType=CONTRACTS&period=5m",
 "MEXC futures: recent deals (has side)": "https://contract.mexc.com/api/v1/contract/deals/BTC_USDT?limit=5",
 "MEXC futures: funding history": "https://contract.mexc.com/api/v1/contract/funding_rate/history?symbol=BTC_USDT&page_num=1&page_size=3",
 "MEXC futures: ticker (holdVol)": "https://contract.mexc.com/api/v1/contract/ticker?symbol=BTC_USDT",
}
for name, url in tests.items():
    code, body = get(url)
    print(f"\n## {name}\n   HTTP {code}")
    if code == 200:
        try:
            j = json.loads(body if len(body) < 2900 else body[:2900])
        except Exception:
            j = None
        print("   ", body[:350].replace("\n", " "))
    else:
        print("   ", body[:250].replace("\n", " "))

# history depth for the Binance 5m ratio endpoints
for name, url in (("takerlongshortRatio start 31d ago", f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=5m&limit=500&startTime={now-31*86400000}&endTime={now-30*86400000}"),
                  ("takerlongshortRatio start 20d ago", f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=5m&limit=500&startTime={now-20*86400000}&endTime={now-19*86400000}"),
                  ("klines taker volume 100d ago", f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=3&startTime={now-100*86400000}")):
    code, body = get(url)
    print(f"\n## depth: {name}: HTTP {code} -> {body[:200]}")
