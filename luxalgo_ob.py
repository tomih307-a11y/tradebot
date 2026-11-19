"""Internal order blocks re-implemented from LuxAlgo 'Smart Money Concepts' (CC BY-NC-SA 4.0, (c) LuxAlgo) for personal, non-commercial analysis.
Defaults as in the Pine script: swing length 50, internal structure size 5, ATR(200) volatility filter, High/Low mitigation, 5 internal order blocks shown."""
import math


def rma(vals, n):
    """Pine ta.rma: first value at index n-1 is the SMA, then alpha = 1/n."""
    out, e = [math.nan] * len(vals), None
    for i, v in enumerate(vals):
        if i == n - 1:
            e = sum(vals[:n]) / n; out[i] = e
        elif i >= n:
            e = (e * (n - 1) + v) / n; out[i] = e
    return out


def run(o, h, l, c, swing_len=50, internal_len=5, atr_len=200, max_show=5):
    n = len(c)
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, n)]
    atr = rma(tr, atr_len)
    parsed_h, parsed_l = [], []
    legs = {"swing": 0, "internal": 0}
    piv = {k: {"cur": math.nan, "crossed": False, "idx": 0} for k in ("sh", "sl", "ih", "il")}
    trend = {"swing": 0, "internal": 0}
    obs, alive, snaps = [], [], []                       # alive: list of ob dicts, newest first
    prev_level = {"ih": math.nan, "il": math.nan}
    for i in range(n):
        hv = (h[i] - l[i]) >= 2 * atr[i] if not math.isnan(atr[i]) else False
        parsed_h.append(l[i] if hv else h[i]); parsed_l.append(h[i] if hv else l[i])

        def structure(size, kind):
            if i < size:
                return
            window = h[i - size + 1:i + 1]
            lwin = l[i - size + 1:i + 1]
            leg = legs[kind]
            new_high = h[i - size] > max(window)
            new_low = l[i - size] < min(lwin)
            new = leg
            if new_high:
                new = 0
            elif new_low:
                new = 1
            if new != leg:
                legs[kind] = new
                key = ("sl" if new == 1 else "sh") if kind == "swing" else ("il" if new == 1 else "ih")
                p = piv[key]
                p["cur"] = l[i - size] if new == 1 else h[i - size]
                p["crossed"] = False; p["idx"] = i - size

        structure(swing_len, "swing")
        structure(internal_len, "internal")
        # internal structure breaks -> order blocks
        ih, il = piv["ih"], piv["il"]
        if not math.isnan(ih["cur"]) and not ih["crossed"] and c[i] > ih["cur"] and (i == 0 or c[i - 1] <= prev_level["ih"]) and ih["cur"] != piv["sh"]["cur"]:
            ih["crossed"] = True; trend["internal"] = 1
            seg = parsed_l[ih["idx"]:i]
            if seg:
                j = ih["idx"] + seg.index(min(seg))
                ob = {"bias": 1, "hi": parsed_h[j], "lo": parsed_l[j], "t": j, "created": i, "removed": None}
                obs.append(ob); alive.insert(0, ob)
        if not math.isnan(il["cur"]) and not il["crossed"] and c[i] < il["cur"] and (i == 0 or c[i - 1] >= prev_level["il"]) and il["cur"] != piv["sl"]["cur"]:
            il["crossed"] = True; trend["internal"] = -1
            seg = parsed_h[il["idx"]:i]
            if seg:
                j = il["idx"] + seg.index(max(seg))
                ob = {"bias": -1, "hi": parsed_h[j], "lo": parsed_l[j], "t": j, "created": i, "removed": None}
                obs.append(ob); alive.insert(0, ob)
        for ob in list(alive):                           # mitigation by High/Low
            if (ob["bias"] == -1 and h[i] > ob["hi"]) or (ob["bias"] == 1 and l[i] < ob["lo"]):
                ob["removed"] = i; alive.remove(ob)
        prev_level = {"ih": ih["cur"], "il": il["cur"]}
        snaps.append([ob for ob in alive[:max_show]])
    return obs, snaps
