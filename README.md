# tradebot — candidate scanner (MEXC USDT futures, crypto, long only)

`scanner2.py [hours]` lists recent 15m signals with all context for a human to judge. It does **not** recommend and does not claim profit:
research (15m/1H/4H, up to 2 years, 20k signals) found no rule that beats random entry after costs out of sample.

Signals: PMO(35,20,10) zero-touch up, or PMO bullish cross + MACD(12,26,9) liquidity entry. Context: TEMA50, 1H/4H/daily PMO+MACD state with bearish-cross warnings,
LuxAlgo-style order blocks (CC BY-NC-SA 4.0, © LuxAlgo; personal use), suggested stop (order-block low or 16-bar swing low), volume multiple, BTC state.

Run on GitHub Actions: push a new timestamp to `.github/run-scan.txt`; results are published to branch `scan-results` (`scan.txt`).
Research scripts: `build_dataset*.py`, `liquidity_test*.py`, `swing4h.py`, `analyze_rise.py`, `gainers_study.py`.
