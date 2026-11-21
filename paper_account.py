"""Paper account with compounding. Replays tracked virtual trades chronologically.
 - size = risk% of CURRENT equity / stop distance (profits are reinvested), capped by max leverage
 - costs: fee 0.04%/side + slippage 0.05%/side on notional + funding (rate captured at detection, charged per 8h held; unknown = 0)
 - limits: max concurrent positions, one position per coin, drawdown brake (risk halves below -10% from peak, no new trades below -25%)
 - equity = realised; marked = realised + unrealised PnL of open trades (after exit costs)
CLI: python3 paper_account.py tracker.json [risk_pct=1.0] [max_open=5] [group=all|match|w3|w4]"""
import json
import sys

FEE, SLIP, MAXLEV = 0.04 / 100, 0.05 / 100, 5.0


def select(state, grp):
    if grp == "match":
        return [x for x in state if x["group"] == "match"]
    if grp == "w3":
        return [x for x in state if x.get("w3")]
    if grp == "w4":
        return [x for x in state if x.get("w4")]
    return list(state)


def simulate(state, risk_pct=1.0, max_open=5, grp="all", start=1000.0, now=None):
    trades = select(state, grp)
    events = []
    for x in trades:
        events.append((x["detect_t"], 1, x))
        if x["status"] == "closed":
            events.append((x["closed_t"], 0, x))
    events.sort(key=lambda e: (e[0], e[1]))
    equity, peak, maxdd = start, start, 0.0
    open_pos, log, skipped = {}, [], 0
    for ts, kind, x in events:
        if kind == 1:
            dd = equity / peak - 1
            if dd <= -0.25 or len(open_pos) >= max_open or x["sym"] in open_pos:
                skipped += 1
                continue
            r = risk_pct * (0.5 if dd <= -0.10 else 1.0) / 100
            stop_dist = max(1 - x["stop"] / x["entry"], 0.002)
            open_pos[x["sym"]] = (min(equity * r / stop_dist, equity * MAXLEV), x)
        else:
            if x["sym"] not in open_pos:
                continue
            notional, _ = open_pos.pop(x["sym"])
            ret = x["exit_px"] / x["entry"] - 1
            hours = (x["closed_t"] - x["detect_t"]) / 3600
            funding = notional * (x.get("funding") or 0) / 100 * hours / 8
            pnl = notional * (ret - 2 * (FEE + SLIP)) - funding
            equity += pnl
            peak = max(peak, equity)
            maxdd = min(maxdd, equity / peak - 1)
            log.append({"t": ts, "sym": x["sym"], "ret": ret * 100, "pnl": pnl, "equity": equity})
    unreal = 0.0
    for sym, (notional, x) in open_pos.items():
        ret = x.get("now", x["entry"]) / x["entry"] - 1
        unreal += notional * (ret - 2 * (FEE + SLIP))
    return {"equity": equity, "marked": equity + unreal, "maxdd": maxdd, "n": len(log), "skipped": skipped, "open": len(open_pos), "log": log,
            "exposure": sum(n for n, _ in open_pos.values()), "start": start}


if __name__ == "__main__":
    path = sys.argv[1]
    risk = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    mo = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    g = sys.argv[4] if len(sys.argv) > 4 else "all"
    r = simulate(json.load(open(path)), risk, mo, g)
    print(f"{g}: {r['n']} صفقة · الرصيد {r['equity']:.2f}$ ({(r['equity']/r['start']-1)*100:+.2f}%) · مع المفتوحة {r['marked']:.2f}$ · أقصى هبوط {r['maxdd']*100:.1f}% · مفتوحة {r['open']} · تخطّي {r['skipped']}")
