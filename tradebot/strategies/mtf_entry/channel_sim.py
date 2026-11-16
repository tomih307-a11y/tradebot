"""Portfolio simulation on the dated channel signals (assumptions: docs/CHANNEL_SIM.md)."""
from __future__ import annotations

import sys
import time

import numpy as np

from tradebot.config import load_config
from tradebot.data.klines import drop_unclosed, fetch_range
from tradebot.mexc.client import MexcClient

from .channel_check2 import DAYS, SIGNALS, locate

START, FRAC, MM = 250.0, 0.10, 0.005


def simulate(sig, h, l, c, t, j, lev, fee):
    """Return (list of (time, pnl as fraction of margin) events, status text) for entry at bar j."""
    e, stop, tps = sig["entry"], sig["stop"], sorted(sig["tps"])
    n = len(tps)
    liq = e * (1 - 1 / lev + MM)
    ev, rem, be, k = [], 1.0, False, 0
    ev.append((t[j], -fee * lev))                                   # entry fee
    for i in range(j + 1, len(c)):
        if not be:
            if l[i] <= liq:
                ev.append((t[i], -1.0 * 1.0)); return ev, "تصفية قبل TP1"
            if l[i] <= stop and stop > liq:
                ev.append((t[i], rem * lev * (stop / e - 1) - fee * lev * rem)); return ev, "وقف قبل TP1"
        else:
            if l[i] <= e:
                ev.append((t[i], -fee * lev * rem)); return ev, f"وقف عند الدخول بعد {k}/{n} أهداف"
        while k < n and h[i] >= tps[k]:
            f = 1.0 / n
            ev.append((t[i], f * lev * (tps[k] / e - 1) - fee * lev * f))
            rem -= f; k += 1; be = True
        if k == n:
            return ev, f"كل الأهداف {n}/{n}"
    ev.append((t[-1], rem * lev * (c[-1] / e - 1) - fee * lev * rem))
    return ev, f"مفتوحة عند آخر سعر بعد {k}/{n} أهداف"


def portfolio(trades, lev, fee):
    """trades: list of (name, j_time, events_fn(lev,fee)->(events,status)). Margin = 10% of realized equity at entry."""
    plan = []
    for name, tj, fn in trades:
        ev, status = fn(lev, fee)
        plan.append([name, tj, ev, status])
    plan.sort(key=lambda x: x[1])
    done, rows = [], []
    for name, tj, ev, status in plan:
        eq = START + sum(m * p for m, tt, p in done if tt <= tj)
        eq = max(eq, 0.0)
        m = eq * FRAC
        for tt, p in ev:
            done.append((m, tt, p))
        net = sum(p for _, p in ev) * m
        rows.append((name, tj, m, net, status))
    final = START + sum(m * p for m, _, p in done)
    return rows, max(final, 0.0)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    trades = []
    for sig in SIGNALS:
        try:
            k = drop_unclosed(fetch_range(cl, sig["sym"], "Min1", now // 60 * 60 - DAYS * 86400, now), "Min1", now)
            h, l, c, t = (np.asarray(k[x], float) for x in ("high", "low", "close", "time"))
            m, nm, nc = locate(h, t, sig)
            if m is None or (nm < 2 and len(sig["hits"]) > 1):
                continue
            i1 = min(m.values())
            lo = max(i1 - 1440, 0)
            reach = np.where(l[lo:i1 + 1] <= sig["entry"] * 1.002)[0]
            if len(reach) == 0:
                continue
            j = lo + int(reach[-1])
        except Exception as ex:
            print("skip", sig["sym"], ex, file=sys.stderr); continue
        trades.append((sig["sym"].replace("_USDT", ""), t[j], (lambda lev, fee, s=sig, h=h, l=l, c=c, t=t, j=j: simulate(s, h, l, c, t, j, lev, fee))))
    lines = [f"محاكاة محفظة {START:.0f}$ · هامش {FRAC * 100:.0f}% من المحفظة لكل صفقة · {len(trades)} توصية مؤرخة · الافتراضات: docs/CHANNEL_SIM.md", ""]
    for fee in (0.0, 0.0004):
        lines.append(f"##### {'بلا رسوم' if fee == 0 else 'مع رسوم 0.04% لكل تنفيذ'}")
        for lev in (20, 50, 100):
            rows, final = portfolio(trades, lev, fee)
            lines.append(f"-- الرافعة {lev}x → المحفظة بعد الصفقة الأخيرة: {final:,.2f}$ (الربح {final - START:+,.2f}$ = {(final / START - 1) * 100:+.0f}%)")
            if fee == 0.0:
                for name, tj, m, net, status in rows:
                    lines.append(f"   {name:<6}{time.strftime('%m-%d %H:%M', time.gmtime(tj))} UTC · هامش {m:8.2f}$ · النتيجة {net:+10.2f}$ · {status}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if out:
        open(out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
