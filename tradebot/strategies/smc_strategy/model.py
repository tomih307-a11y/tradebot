"""SMC strategy model. The rules are registered in docs/SMC_STRATEGY.md; nothing here is tuned.

Everything is computed on CLOSED candles and is causal: a swing is used only n bars after it forms, a 4H state is used only from the close
of its 4H candle. Longs are implemented once; shorts run the same code on mirrored prices (o,h,l,c) -> (-o,-l,-h,-c).
"""
from __future__ import annotations

import numpy as np

from tradebot.strategies.multi_agent.smc import pivots, structure

N_SW = 3              # fractal length on both timeframes
K_CHOCH = 16          # 15m bars allowed between the sweep and the CHoCH
BUF_ATR = 0.1         # stop buffer in ATR(15m)
RR_MIN = 2.0          # minimum target distance / stop distance
MAX_BARS = 288        # 72h time exit (15m bars)
LAST_HIGHS = 10       # 4H swing highs considered as target liquidity
CHASE_ATR = 1.0       # CHoCH close may be at most this far above the zone top


def mirror(o, h, l, c):
    return -o, -l, -h, -c


def ob_extreme(ev: dict, h: np.ndarray, l: np.ndarray):
    """Order block of a structure break: the candle with the lowest low (dir=+1) / highest high (dir=-1) in [origin, break)."""
    a, b = ev["origin"], ev["i"]
    if b <= a:
        return None
    j = a + int(np.argmin(l[a:b])) if ev["dir"] == 1 else a + int(np.argmax(h[a:b]))
    return float(l[j]), float(h[j])


def _last_confirmed(piv, n, size):
    """Array: price of the most recent swing confirmed as of each bar (NaN before the first)."""
    out = np.full(size, np.nan)
    conf = sorted((idx + n, price) for idx, price in piv)
    p, cur = 0, np.nan
    for i in range(size):
        while p < len(conf) and conf[p][0] <= i:
            cur = conf[p][1]
            p += 1
        out[i] = cur
    return out


def htf_state(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = N_SW):
    """Per 4H bar (as of its close): bias (+1/-1/0), the latest still-valid OB in the bias direction, dealing range, and swing-high targets."""
    N = len(c)
    _, events = structure(c, h, l, n)
    ev_at: dict[int, list] = {}
    for e in events:
        ev_at.setdefault(e["i"], []).append(e)
    bias = np.zeros(N, int)
    zlo, zhi = np.full(N, np.nan), np.full(N, np.nan)
    active, cur = [], 0
    for i in range(N):
        for e in ev_at.get(i, []):
            cur = e["dir"]
            ob = ob_extreme(e, h, l)
            if ob:
                active.append({"dir": e["dir"], "lo": ob[0], "hi": ob[1], "i": i})
        keep = []
        for z in active:
            if z["i"] < i and ((z["dir"] == 1 and l[i] < z["lo"]) or (z["dir"] == -1 and h[i] > z["hi"])):
                continue                                   # a wick beyond the far edge mitigates the block
            keep.append(z)
        active = keep[-8:]
        bias[i] = cur
        same = [z for z in active if z["dir"] == cur]
        if same:
            zlo[i], zhi[i] = same[-1]["lo"], same[-1]["hi"]
    hs, ls = pivots(h, l, n)
    sh_conf = np.array(sorted(idx + n for idx, _ in hs), dtype=int)
    sh_price = np.array([p for _, p in sorted(hs)], dtype=float)
    return {"bias": bias, "zlo": zlo, "zhi": zhi, "rng_hi": _last_confirmed(hs, n, N), "rng_lo": _last_confirmed(ls, n, N),
            "sh_conf": sh_conf, "sh_price": sh_price}


def ltf_setups(o, h, l, c, atr, close4_t, t15, st, n: int = N_SW, ltf_sec: int = 900):
    """Long setups on the 15m chart. -> list of dict(entry_i, stop, target, sweep_i, choch_i, rr, zlo, zhi).
    close4_t: close time of each 4H bar; t15: OPEN time of each 15m bar; st: htf_state output (same price space)."""
    N = len(c)
    hs15, ls15 = pivots(h, l, n)
    last_sl, last_sh = _last_confirmed(ls15, n, N), _last_confirmed(hs15, n, N)
    i4 = np.searchsorted(close4_t, np.asarray(t15) + ltf_sec, side="right") - 1          # last 4H candle already closed at each 15m close
    ok4 = i4 >= 0
    j4 = np.clip(i4, 0, None)
    zlo, zhi, bias = st["zlo"][j4], st["zhi"][j4], st["bias"][j4]
    rhi, rlo = st["rng_hi"][j4], st["rng_lo"][j4]
    P = np.roll(last_sl, 1)
    P[0] = np.nan
    mid_ok = (zlo + zhi) / 2 <= (rlo + rhi) / 2                                        # the block sits in the discount of the 4H range
    cond = (ok4 & (bias == 1) & np.isfinite(zlo) & np.isfinite(rlo) & (rhi > rlo) & mid_ok & np.isfinite(P) & np.isfinite(atr)
            & (l < P) & (c > P) & (l <= zhi) & (c >= zlo))
    out = []
    for b in np.where(cond)[0]:
        if b < 1 or b + 2 >= N:
            continue
        Hs = last_sh[b - 1]
        if not np.isfinite(Hs) or Hs <= c[b]:
            continue
        ch = None
        for x in range(b + 1, min(b + K_CHOCH, N - 2) + 1):
            if l[x] < l[b]:                                                            # a lower low before the CHoCH invalidates the sweep
                break
            if c[x] > Hs:
                ch = x
                break
        if ch is None:
            continue
        entry = float(o[ch + 1])
        if c[ch] > zhi[b] + CHASE_ATR * atr[ch]:
            continue
        stop = min(float(l[b]), float(zlo[b])) - BUF_ATR * float(atr[ch])
        risk = entry - stop
        if risk <= 0:
            continue
        k = int(np.searchsorted(st["sh_conf"], i4[ch], side="right"))
        cand = st["sh_price"][max(0, k - LAST_HIGHS):k]
        above = cand[cand > entry * 1.001]
        if len(above) == 0:
            continue
        target = float(above.min())
        rr = (target - entry) / risk
        if rr < RR_MIN:
            continue
        out.append({"entry_i": ch + 1, "stop": stop, "target": target, "sweep_i": int(b), "choch_i": int(ch), "rr": float(rr),
                    "zlo": float(zlo[b]), "zhi": float(zhi[b])})
    return out


def sim_fixed(s, i0, stop, tgt, o, h, l, c, fee_pct, slip_pct, max_bars=MAX_BARS):
    """Fixed stop/target trade entered at the open of bar i0 (real price space, s = +1 long / -1 short). Adverse first inside a candle.
    -> dict(R_net, R_gross, outcome 0 time / 1 stop / 2 target, exit_i, risk_pct) or None when the stop is not beyond the entry."""
    n = len(c)
    px = float(o[i0]) * (1 + s * slip_pct / 100.0)
    risk = s * (px - stop)
    if risk <= 0 or i0 >= n:
        return None
    last = min(n - 1, i0 + max_bars - 1)
    ex, outcome, k_exit = None, 0, last
    for k in range(i0, last + 1):
        adverse, favour = (l[k], h[k]) if s == 1 else (h[k], l[k])
        if s * (adverse - stop) <= 0:
            ex = (float(o[k]) if (k > i0 and s * (o[k] - stop) <= 0) else stop) * (1 - s * slip_pct / 100.0)
            outcome, k_exit = 1, k
            break
        if s * (favour - tgt) >= 0:
            ex, outcome, k_exit = tgt, 2, k
            break
    if ex is None:
        ex = float(c[last]) * (1 - s * slip_pct / 100.0)
    gross = s * (ex - px) / risk
    return {"R_gross": gross, "R_net": gross - fee_pct / 100.0 * (px + ex) / risk, "outcome": outcome, "exit_i": k_exit, "risk_pct": risk / px * 100.0}


def find_trades(o, h, l, c, atr, t15, k4, side, fee_pct, slip_pct, start_ts=0, ltf_sec=900, htf_sec=14400):
    """All sequential (non-overlapping) trades of one side. k4: dict of 4H arrays (time, open, high, low, close). -> list of trade dicts."""
    s = 1 if side == "long" else -1
    if s == 1:
        o_, h_, l_, c_ = o, h, l, c
        o4, h4, l4, c4 = k4["open"], k4["high"], k4["low"], k4["close"]
    else:
        o_, h_, l_, c_ = mirror(o, h, l, c)
        o4, h4, l4, c4 = mirror(k4["open"], k4["high"], k4["low"], k4["close"])
    st = htf_state(h4, l4, c4)
    setups = ltf_setups(o_, h_, l_, c_, atr, k4["time"] + htf_sec, t15, st, ltf_sec=ltf_sec)
    trades, busy = [], -1
    for su in sorted(setups, key=lambda x: x["entry_i"]):
        if su["entry_i"] <= busy or t15[su["entry_i"]] < start_ts:
            continue
        stop, tgt = (su["stop"], su["target"]) if s == 1 else (-su["stop"], -su["target"])
        r = sim_fixed(s, su["entry_i"], stop, tgt, o, h, l, c, fee_pct, slip_pct)
        if r is None:
            continue
        busy = r["exit_i"]
        trades.append({**r, "T": int(t15[su["entry_i"]]), "side": side, "rr_plan": su["rr"], "entry_i": su["entry_i"], "stop": stop, "target": tgt})
    return trades


def live_setups(o, h, l, c, atr, t15, k4, side, max_age=1):
    """Setups whose CHoCH closed on the last `max_age`+1 closed 15m bars, i.e. entries that are still actionable at the next open.
    A pseudo bar (open = last close) stands in for the unknown next candle. -> list of dicts in REAL prices with `age` (0 = the last closed bar)."""
    s = 1 if side == "long" else -1
    N = len(c)
    o2, h2, l2, c2 = (np.append(a, c[-1]) for a in (o, h, l, c))
    a2 = np.append(atr, atr[-1])
    t2 = np.append(t15, t15[-1] + 900)
    if s == 1:
        mo, mh, ml, mc = o2, h2, l2, c2
        o4, h4, l4, c4 = k4["open"], k4["high"], k4["low"], k4["close"]
    else:
        mo, mh, ml, mc = mirror(o2, h2, l2, c2)
        o4, h4, l4, c4 = mirror(k4["open"], k4["high"], k4["low"], k4["close"])
    st = htf_state(h4, l4, c4)
    out = []
    for su in ltf_setups(mo, mh, ml, mc, a2, k4["time"] + 4 * 3600, t2, st):
        age = (N - 1) - su["choch_i"]
        if 0 <= age <= max_age:
            stop, tgt = (su["stop"], su["target"]) if s == 1 else (-su["stop"], -su["target"])
            out.append({"side": side, "entry": float(c[-1]), "stop": stop, "target": tgt, "rr": su["rr"], "age": age,
                        "choch_i": su["choch_i"], "sweep_i": su["sweep_i"]})
    return out
