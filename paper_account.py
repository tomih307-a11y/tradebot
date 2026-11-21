"""Paper account with compounding. Replays tracked virtual trades chronologically on a virtual account.
 - position size = risk% of CURRENT equity / stop distance  (compounding: profits are reinvested, losses shrink size), capped by max leverage
 - costs: fee 0.04%/side + slippage 0.05%/side on the notional (funding not modelled)
 - limits: max concurrent positions, one position per coin, drawdown brake (risk halves below -10% from peak, stops new trades below -25%)
Usage: python3 paper_account.py tracker.json [risk_pct=1.0] [max_open=5] [group=all|match|w3]"""
import json
import sys

path = sys.argv[1]
risk_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
max_open = int(sys.argv[3]) if len(sys.argv) > 3 else 5
grp = sys.argv[4] if len(sys.argv) > 4 else "all"
FEE, SLIP, MAXLEV = 0.04 / 100, 0.05 / 100, 5.0
trades = [x for x in json.load(open(path)) if x["status"] == "closed"]
if grp == "match":
    trades = [x for x in trades if x["group"] == "match"]
elif grp == "w3":
    trades = [x for x in trades if x.get("w3")]
events = []
for x in trades:
    events.append((x["detect_t"], 1, x)); events.append((x["closed_t"], 0, x))            # closes processed before opens at the same time
events.sort(key=lambda e: (e[0], e[1]))
equity, peak, start = 1000.0, 1000.0, 1000.0
open_pos, log, skipped, maxdd = {}, [], 0, 0.0
for ts, kind, x in events:
    if kind == 1:
        dd = equity / peak - 1
        if dd <= -0.25 or len(open_pos) >= max_open or x["sym"] in open_pos:
            skipped += 1; continue
        r = risk_pct * (0.5 if dd <= -0.10 else 1.0) / 100
        stop_dist = 1 - x["stop"] / x["entry"]
        notional = min(equity * r / stop_dist, equity * MAXLEV)
        open_pos[x["sym"]] = (notional, x)
    else:
        if x["sym"] not in open_pos:
            continue
        notional, _ = open_pos.pop(x["sym"])
        ret = x["exit_px"] / x["entry"] - 1
        pnl = notional * (ret - 2 * (FEE + SLIP))
        equity += pnl
        peak = max(peak, equity)
        maxdd = min(maxdd, equity / peak - 1)
        log.append((ts, x["sym"], x["group"], ret * 100, pnl, equity))
print(f"حساب ورقي · رأس المال {start:.0f}$ · مخاطرة {risk_pct}% من الرصيد الحالي لكل صفقة · أقصى مفتوحة {max_open} · رافعة قصوى {MAXLEV:.0f}× · رسوم+انزلاق مضمّنة · المجموعة: {grp}")
print(f"صفقات منفّذة {len(log)} · تخطّي {skipped} · الرصيد النهائي {equity:.2f}$ ({(equity/start-1)*100:+.2f}%) · أقصى هبوط {maxdd*100:.1f}%")
for ts, sym, g, ret, pnl, eq in log:
    print(f"  {sym[:-5]:10s} {g:6s} {ret:+6.2f}% · ربح/خسارة {pnl:+8.2f}$ · الرصيد {eq:8.2f}$")
