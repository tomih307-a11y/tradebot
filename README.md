# tradebot — candidate scanner (MEXC USDT futures, crypto, long only)

`scanner2.py [hours]` lists recent 15m signals with all context for a human to judge. It does **not** recommend and does not claim profit:
research (15m/1H/4H, up to 2 years, 20k signals) found no rule that beats random entry after costs out of sample.

Signals: PMO(35,20,10) zero-touch up, or PMO bullish cross + MACD(12,26,9) liquidity entry. Context: TEMA50, 1H/4H/daily PMO+MACD state with bearish-cross warnings,
LuxAlgo-style order blocks (CC BY-NC-SA 4.0, © LuxAlgo; personal use), suggested stop (order-block low or 16-bar swing low), volume multiple, BTC state.

Run on GitHub Actions: push a new timestamp to `.github/run-scan.txt`; results are published to branch `scan-results` (`scan.txt`).
Research scripts: `build_dataset*.py`, `liquidity_test*.py`, `swing4h.py`, `analyze_rise.py`, `gainers_study.py`.

## PMO-only scanner
`scanner_pmo.py [hours]` — PMO(35,20,10) only: 15m zero-touch up, grouped by daily PMO above zero (matches the rule) or not, plus an 'about to touch' list, exit status (blue crossed below orange) and a 1.5xATR stop for display. Not a recommendation: in the 70-day test the best variant (zero-touch + daily PMO above zero) was ~break-even (+0.017R, CI contains zero).
