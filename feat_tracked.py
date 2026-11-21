import json, os, sys
import build_features as B
state = json.load(open("tracker.json"))
syms = sorted({x["sym"] for x in state} | {"BLESS_USDT"})
now = int(__import__("time").time())
B.btc_prepare(now)
want = {(x["sym"], x["signal_t"]): x for x in state}
rows = []
for sym in syms:
    for r in B.work(sym):
        key = (sym, r["t"])
        if key in want or sym == "BLESS_USDT":
            x = want.get(key)
            r["tracked"] = bool(x)
            if x:
                r["group"], r["status"], r["R_live"] = x["group"], x["status"], x.get("R", x.get("open_R"))
                r["peak_pct"] = (x["peak_since"] / x["entry"] - 1) * 100 if x.get("peak_since") else None
                r["peak_during_pct"] = (x["peak_during"] / x["entry"] - 1) * 100 if x.get("peak_during") else None
            rows.append(r)
json.dump(rows, open("feat_tracked.json", "w"))
print(len(rows), "rows", sum(1 for r in rows if r.get("tracked")), "tracked")
