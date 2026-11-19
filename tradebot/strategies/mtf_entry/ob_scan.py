"""Live read-only scan: bullish 1H order blocks that price is in/at NOW (user rule: always enter from an order block).
NOT a validated edge (see docs/OB_ENTRY.md and the SMC D1/H1 result). Usage: python -m tradebot.strategies.mtf_entry.ob_scan [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from tradebot.strategies.multi_agent.smc import order_block, structure

from .channel_logic import atr14

N, MAX_AGE, NEAR_ATR, BUF_ATR, MIN_TURN = 3, 48, 0.5, 0.25, 5_000_000


def scan_symbol(o, h, l, c, px, o4, h4, l4, c4):
    """Return dict for a bullish 1H OB that price is in/near now, else None. Uses closed candles + current price px."""
    _, ev = structure(c, h, l, N)
    cand = [e for e in ev if e["dir"] == 1 and len(c) - 1 - e["i"] <= MAX_AGE]
    if not cand:
        return None
    e = cand[-1]
    ob = order_block(e, o, h, l, c)
    if ob is None:
        return None
    lo, hi = ob[0], ob[1]
    after = slice(e["i"] + 1, len(c))
    if (c[after] < lo).any() or px < lo:                               # mitigated/broken
        return None
    atr = atr14(h, l, c)[-1]
    if not np.isfinite(atr) or not (lo <= px <= hi + NEAR_ATR * atr):
        return None
    first = bool(not (l[after] <= hi).any() and px <= hi)
    t4, _ = structure(c4, h4, l4, N)
    stop = lo - BUF_ATR * atr
    return dict(lo=lo, hi=hi, px=px, stop=stop, risk_pct=(px - stop) / px * 100, atr_pct=atr / px * 100, first=first,
                htf_up=t4 == 1, age=len(c) - 1 - e["i"], kind=e.get("kind", ""))


def main():
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import TF_SECONDS, MexcClient

    from .paper import NON_CRYPTO

    out = sys.argv[1] if len(sys.argv) > 1 else None
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= MIN_TURN]
    res = []
    for tk in tick:
        sym = tk["symbol"]
        try:
            def get(tf, days):
                sec = TF_SECONDS[tf]
                k = drop_unclosed(fetch_range(cl, sym, tf, int(now // sec * sec) - days * 86400, now), tf, now)
                return tuple(np.asarray(k[x], float) for x in ("open", "high", "low", "close"))
            o, h, l, c = get("Min60", 40)
            o4, h4, l4, c4 = get("Hour4", 80)
            if len(c) < 200 or len(c4) < 60:
                continue
            r = scan_symbol(o, h, l, c, float(tk["lastPrice"]), o4, h4, l4, c4)
            if r:
                r.update(sym=sym, turn=float(tk["amount24"]), chg=float(tk.get("riseFallRate") or 0) * 100)
                res.append(r)
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
    lines = [f"مسح الأوردر بلوك الصاعد (ساعة) · {len(tick)} عقداً بسيولة ≥ 5M · {time.strftime('%Y-%m-%d %H:%M', time.gmtime(now))} UTC", "",
             "مرتبة: أول لمسة، ثم اتجاه 4س صاعد، ثم أصغر مخاطرة. وقف = أدنى المنطقة − 0.25×ATR الساعة. وقف متحرك بالمسافة نفسها."]
    for r in sorted(res, key=lambda r: (not r["first"], not r["htf_up"], r["risk_pct"])):
        lines.append(f"{r['sym']:<14} سعر {r['px']:.6g} · منطقة [{r['lo']:.6g}, {r['hi']:.6g}] · {'أول لمسة' if r['first'] else 'لمسة سابقة'} · اتجاه 4س {'صاعد' if r['htf_up'] else 'غير صاعد'} · "
                     f"عمر الكسر {r['age']}س ({r['kind']}) · وقف {r['stop']:.6g} (−{r['risk_pct']:.1f}%) · ATR {r['atr_pct']:.1f}% · سيولة {r['turn'] / 1e6:.1f}M · 24س {r['chg']:+.1f}%")
    if not res:
        lines.append("لا شيء")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
