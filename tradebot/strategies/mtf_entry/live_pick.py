"""Live scan with FIXED rules, read-only: (A) hunter tiers (docs/HUNTER_TIERS.md), (B) MA5/10/30/60 pullback S (docs/CHANNEL_LOGIC.md).
Usage: python -m tradebot.strategies.mtf_entry.live_pick [out_file]"""
from __future__ import annotations

import sys
import time

import numpy as np

from tradebot.config import load_config
from tradebot.data.klines import drop_unclosed, fetch_range
from tradebot.mexc.client import TF_SECONDS, MexcClient

from . import channel_logic as CL
from .early_signs import WARM, features
from .hunter_tiers import NAMES, TIERS, alerts, tier_of
from .paper import NON_CRYPTO


def kl(cl, sym, tf, days, now):
    sec = TF_SECONDS[tf]
    k = drop_unclosed(fetch_range(cl, sym, tf, int(now // sec * sec) - days * 86400, now), tf, now)
    return tuple(np.asarray(k[f], float) for f in ("open", "high", "low", "close", "vol", "time"))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    tick = [t for t in cl.tickers() if str(t.get("symbol", "")).endswith("_USDT") and not t["symbol"].endswith("STOCK_USDT")
            and t["symbol"] not in NON_CRYPTO and float(t.get("amount24") or 0) >= 300_000]
    chg = {t["symbol"]: float(t.get("riseFallRate") or 0) * 100 for t in tick}
    amt = {t["symbol"]: float(t.get("amount24") or 0) for t in tick}
    A, B = [], []
    for tk in tick:
        sym = tk["symbol"]
        try:
            o, h, l, c, v, t = kl(cl, sym, "Min15", 12, now)
            if len(c) < WARM + 20:
                continue
            f, fmax, valid = features(o, h, l, c, v, t, np.array([0.0]), np.array([0.0]), np.zeros(len(c)))
            tier = tier_of({x: f[x] for x in ("F1", "F2", "F3")})
            tier[:WARM] = -1
            al = alerts(tier)
            recent = [(i, k) for i, k in al if i >= len(c) - 8]
            if recent:
                i, k = min(recent, key=lambda x: x[1])          # best (lowest index) tier among recent alerts
                a1 = CL.atr14(*kl(cl, sym, "Min60", 14, now)[1:4])[-1]
                A.append(dict(sym=sym, tier=k, age=(len(c) - 1 - i) * 15, px=c[-1], chg24=chg[sym], amt=amt[sym], atr1h=a1,
                              ret_since=(c[-1] / c[i] - 1) * 100, f1=bool(f["F1"][i]), f2=bool(f["F2"][i]), f3=bool(f["F3"][i])))
            for tf, days in (("Min60", 14), ("Hour4", 40)):
                o2, h2, l2, c2, _, t2 = kl(cl, sym, tf, days, now)
                if len(c2) < 120:
                    continue
                s, b1 = CL.setups(c2)
                if s[-3:].any():
                    a2 = CL.atr14(h2, l2, c2)[-1]
                    B.append(dict(sym=sym, tf=tf, bars=int(len(s) - 1 - np.where(s)[0][-1]), px=c2[-1], atr=a2, chg24=chg[sym], amt=amt[sym]))
        except Exception as e:
            print("skip", sym, e, file=sys.stderr)
    lines = [f"فحص حي · {len(tick)} عقداً · {time.strftime('%Y-%m-%d %H:%M', time.gmtime(now))} UTC · قواعد ثابتة (docs/HUNTER_TIERS.md، docs/CHANNEL_LOGIC.md)", ""]
    lines.append("(أ) تنبيهات طبقات الصائد خلال آخر ساعتين:")
    for r in sorted(A, key=lambda r: (r["tier"], r["age"])):
        risk = CL.ATR_MULT * r["atr1h"]
        lines.append(f"  {TIERS[r['tier']]} {NAMES[TIERS[r['tier']]]:<24}{r['sym']:<14} منذ {r['age']}د · سعر {r['px']:.6g} · 24س {r['chg24']:+.1f}% · منذ التنبيه {r['ret_since']:+.1f}% · "
                     f"سيولة {r['amt'] / 1e6:.1f}M · ATR ساعة {r['atr1h'] / r['px'] * 100:.2f}% · وقف 1.5ATR = {r['px'] - risk:.6g} · هدف 2R = {r['px'] + 2 * risk:.6g}")
    if not A:
        lines.append("  لا شيء")
    lines += ["", "(ب) إشارة المتوسطات S (رجوع بعد كسر، شراء) في آخر 3 شموع:"]
    for r in sorted(B, key=lambda r: (r["tf"], r["bars"])):
        risk = CL.ATR_MULT * r["atr"]
        lines.append(f"  {r['tf']:<6}{r['sym']:<14} منذ {r['bars']} شمعة · سعر {r['px']:.6g} · 24س {r['chg24']:+.1f}% · سيولة {r['amt'] / 1e6:.1f}M · وقف {r['px'] - risk:.6g} · هدف 2R {r['px'] + 2 * risk:.6g}")
    if not B:
        lines.append("  لا شيء")
    both = sorted({r["sym"] for r in A} & {r["sym"] for r in B})
    lines += ["", f"في القائمتين معاً: {', '.join(both) if both else 'لا شيء'}"]
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
