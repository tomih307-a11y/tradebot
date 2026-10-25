"""Walk-forward test of the SMC analyst on MEXC history (no news, no 1m entry-confirmation: see report notes).

At every closed 1H bar the agent sees only candles up to that bar (1H window + 4H bars already closed) and may publish a setup.
Execution is replayed on 15m candles: limit entry at the zone's first-touch edge (valid `expire` 1H bars; cancelled if never touched),
stop beyond the zone, targets at R multiples (equal shares), optional breakeven after TP1, time stop, adverse-first inside a candle.
Costs: taker fee on both sides + slippage on market-type exits. Baselines with the SAME stop distance / targets:
  market  = enter at the setup's 1H close (does the zone wait matter?)   reverse = opposite side of the market baseline.
Usage: python tools/smc_study.py --out reports/smc --days 150 --top 25
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.config import load_config                                  # noqa: E402
from tradebot.data.klines import get_klines                               # noqa: E402
from tradebot.mexc.client import TF_SECONDS, MexcClient                   # noqa: E402
from tradebot.strategies.multi_agent.smc import SMCAgent                  # noqa: E402
from tradebot.strategies.multi_agent.types import Candidate               # noqa: E402

H1, M15 = 3600, 900


def simulate_trade(s, entry, stop, tps, c15, j0, fill_mode, expire_n, max_hold_n, be, fee_pct, slip_pct, zone_edge=None):
    """s=+1/-1. c15: dict of arrays; j0 = first 15m candle that may be used.
    fill_mode 'limit' (touch of `entry`, within expire_n candles) or 'market' (at open of candle j0).
    -> dict(filled, R_net, R_gross, reason, tp_hit, exit_j, fill_j) ; R in units of the initial risk."""
    o, h, l, c = c15["open"], c15["high"], c15["low"], c15["close"]
    n = len(c)
    risk = s * (entry - stop)
    if risk <= 0 or j0 >= n:
        return {"filled": False}
    j = j0
    if fill_mode == "market":
        px_in = float(o[j0])
        if s * (px_in - stop) <= 0:
            return {"filled": False}
        fill_j, first_fill_candle = j0, False
        px_in = float(o[j0]) * (1 + s * slip_pct / 100.0)
        risk = s * (px_in - stop)
        if risk <= 0:
            return {"filled": False}
    else:
        fill_j = None
        for j in range(j0, min(n, j0 + expire_n)):
            if s * (l[j] - entry) <= 0 if s == 1 else s * (h[j] - entry) <= 0:     # long: low<=entry ; short: high>=entry
                fill_j = j
                break
        if fill_j is None:
            return {"filled": False}
        px_in = entry if s * (o[fill_j] - entry) >= 0 else float(o[fill_j])           # a better open fills at the open
        first_fill_candle = True
    cur_stop, rem, tp_hit, share = stop, 1.0, 0, 1.0 / len(tps)
    gross = 0.0
    fee_in = fee_pct / 100.0 * px_in / risk                                          # per unit position, in R
    fee_out = 0.0
    reason = "time"
    exit_j = min(n - 1, fill_j + max_hold_n)
    start = fill_j if first_fill_candle else fill_j
    for j in range(start, min(n, fill_j + max_hold_n + 1)):
        adverse = l[j] if s == 1 else h[j]
        favourable = h[j] if s == 1 else l[j]
        if s * (adverse - cur_stop) <= 0:                                            # stop (adverse first; also in the fill candle)
            px = float(o[j]) if (j > fill_j and s * (o[j] - cur_stop) <= 0) else cur_stop
            label = "breakeven" if (tp_hit and cur_stop == px_in) else "stop"
            px *= (1 - s * slip_pct / 100.0)
            gross += rem * s * (px - px_in) / risk
            fee_out += rem * fee_pct / 100.0 * px / risk
            rem, reason, exit_j = 0.0, label, j
            break
        if j == fill_j and first_fill_candle:
            continue                                                                 # favourable moves in the fill candle are not trusted
        while tp_hit < len(tps) and s * (favourable - tps[tp_hit]) >= 0:
            q = rem if tp_hit == len(tps) - 1 else min(share, rem)
            gross += q * s * (tps[tp_hit] - px_in) / risk
            fee_out += q * fee_pct / 100.0 * tps[tp_hit] / risk
            rem = round(rem - q, 12)
            tp_hit += 1
            if tp_hit == 1 and be and s * (px_in - cur_stop) > 0:
                cur_stop = px_in
            if rem <= 1e-9:
                break
        if rem <= 1e-9:
            reason, exit_j = "all_tp", j
            break
    if rem > 1e-9:                                                                   # time stop at the last allowed candle's close
        jj = min(n - 1, fill_j + max_hold_n)
        px = float(c[jj]) * (1 - s * slip_pct / 100.0)
        gross += rem * s * (px - px_in) / risk
        fee_out += rem * fee_pct / 100.0 * px / risk
        reason, exit_j = "time", jj
    return {"filled": True, "R_gross": gross, "R_net": gross - fee_in - fee_out, "reason": reason, "tp_hit": tp_hit,
            "exit_j": exit_j, "fill_j": fill_j, "risk_pct": risk / px_in * 100.0}


def summarize(rs):
    x = np.asarray(rs, float)
    if len(x) == 0:
        return None
    wins, losses = x[x > 0].sum(), -x[x < 0].sum()
    eq = np.cumsum(x)
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)), "win": float((x > 0).mean() * 100),
            "pf": float(wins / losses) if losses > 0 else float("inf"), "total": float(x.sum()),
            "t": float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0,
            "mdd": float((np.maximum.accumulate(eq) - eq).max())}


def generate(cfg, client, symbols, eval_start, now, args):
    agent = SMCAgent(cfg)
    agent.c = cfg.agents.smc.override(min_score=0.0)
    out, diag = {}, {"bars": 0, "setups": 0, "symbols": 0}
    data_dir = cfg.pmo_zero_cross.backtest.data_dir
    for n_, sym in enumerate(symbols, 1):
        try:
            k1 = get_klines(client, data_dir, sym, "Min60", eval_start - 320 * H1, now)
            k4 = get_klines(client, data_dir, sym, "Hour4", eval_start - 320 * 4 * H1, now)
            k15 = get_klines(client, data_dir, sym, "Min15", eval_start - 3600, now)
        except Exception as e:
            print(f"  {sym}: data failed {e}", flush=True)
            continue
        if len(k1["time"]) < 350 or len(k15["time"]) < 500:
            continue
        diag["symbols"] += 1
        close4 = k4["time"] + 4 * H1
        recs = []
        for i in range(299, len(k1["time"])):
            T = int(k1["time"][i]) + H1
            if T < eval_start:
                continue
            j4 = int(np.searchsorted(close4, T, side="right"))
            if j4 < 120:
                continue
            w1 = {f: k1[f][i - 299:i + 1] for f in ("time", "open", "high", "low", "close")}
            w4 = {f: k4[f][max(0, j4 - 300):j4] for f in ("time", "open", "high", "low", "close")}
            diag["bars"] += 1
            r = agent.analyze(Candidate(sym, float(k1["close"][i]), 0, 0, 0), w1, w4, T)
            if r:
                diag["setups"] += 1
                recs.append((T, r))
        out[sym] = (k15, recs)
        print(f"  [{n_}/{len(symbols)}] {sym}: {len(recs)} setups", flush=True)
    return out, diag


def run_variant(data, cfg, min_score, be, mode, args):
    """Sequential, non-overlapping per symbol. Returns list of trade dicts."""
    fee, slip = cfg.trading.taker_fee_pct, args.slip
    trades = []
    for sym, (k15, recs) in data.items():
        t15 = k15["time"]
        busy_until, used = 0, set()
        for T, r in recs:
            if r.score < min_score or T < busy_until:
                continue
            key = (round(r.zone[0], 8), round(r.zone[1], 8), r.side)
            if key in used:
                continue
            s = 1 if r.side == "long" else -1
            j0 = int(np.searchsorted(t15, T, side="left"))
            near = r.zone[1] if s == 1 else r.zone[0]
            risk = s * (near - r.stop)
            tps = [near + s * m * risk for m in cfg.agents.smc.tp_r]
            if mode == "zone":
                res = simulate_trade(s, near, r.stop, tps, k15, j0, "limit", args.expire * 4, args.hold * 4, be, fee, slip)
            else:                                   # market baselines: same distances around the 1H close
                ms = s if mode == "market" else -s
                px = r.price
                stop = px - ms * risk
                t_ = [px + ms * m * risk for m in cfg.agents.smc.tp_r]
                res = simulate_trade(ms, px, stop, t_, k15, j0, "market", 0, args.hold * 4, be, fee, slip)
            used.add(key)
            if not res["filled"]:
                busy_until = T + args.expire * H1 if mode == "zone" else T
                continue
            busy_until = int(t15[res["exit_j"]]) + M15
            trades.append({"symbol": sym, "T": T, "side": r.side if mode != "reverse" else ("short" if s == 1 else "long"),
                           "score": r.score, "mode": mode, **res})
    return sorted(trades, key=lambda t: t["T"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/smc")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--expire", type=int, default=8, help="setup validity, 1H bars")
    ap.add_argument("--hold", type=int, default=72, help="max holding, 1H bars")
    ap.add_argument("--slip", type=float, default=0.05, help="% slippage on market-type fills/exits")
    a = ap.parse_args()
    cfg = load_config(a.config)
    client = MexcClient(cfg.mexc)
    now = int(time.time())
    eval_start = now - a.days * 86400
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    symbols = [t["symbol"] for t in tick[: a.top]]
    print(f"symbols={len(symbols)} days={a.days}", flush=True)
    data, diag = generate(cfg, client, symbols, eval_start, now, a)
    print(diag, flush=True)

    L = ["# اختبار تحليل SMC على تاريخ MEXC (walk-forward)", ""]
    L.append(f"- العقود: {diag['symbols']} من أعلى {a.top} سيولة (اختيار بحسب سيولة اليوم → انحياز البقاء ممكن) · الفترة: آخر {a.days} يوماً")
    L.append(f"- شموع الساعة المحلَّلة: {diag['bars']} · تنبيهات SMC (بلا حد درجة): {diag['setups']}")
    L.append(f"- الدخول: أمر حد عند أول حافة للمنطقة، صلاحية {a.expire} ساعات؛ الأهداف {list(cfg.agents.smc.tp_r)}R بحصص متساوية؛ أقصى احتفاظ {a.hold} ساعة")
    L.append(f"- التكلفة: رسوم Taker {cfg.trading.taker_fee_pct}% للجهتين + انزلاق {a.slip}% على الخروج السوقي/دخول السوق. R = المخاطرة الأولية (المسافة للوقف).")
    L.append("- **غير مُحاكى:** تأكيد شموع الدقيقة لوكيل الدخول (الدقيقة غير متاحة تاريخياً)، الأخبار، تصويت الوكلاء الآخرين، التمويل.\n")
    L.append("| الدرجة ≥ | BE بعد TP1 | النمط | الصفقات | % تعبئة | ربح % | متوسط R | وسيط R | PF | مجموع R | t | أقصى تراجع R |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    all_rows = []
    for ms in (0.5, 0.67, 0.83):
        for be in (False, True):
            n_recs = None
            for mode, label in (("zone", "SMC (حد عند المنطقة)"), ("market", "أساس: سوق بنفس الوقف"), ("reverse", "أساس: عكس الاتجاه")):
                tr = run_variant(data, cfg, ms, be, mode, a)
                sm = summarize([t["R_net"] for t in tr])
                cand = sum(1 for _, (_, recs) in data.items() for _, r in recs if r.score >= ms)
                fill = f"{len(tr) / cand * 100:.0f}%" if cand else "-"
                if sm:
                    L.append(f"| {ms} | {'نعم' if be else 'لا'} | {label} | {sm['n']} | {fill} | {sm['win']:.0f}% | {sm['mean']:+.3f} | "
                             f"{sm['median']:+.3f} | {sm['pf']:.2f} | {sm['total']:+.1f} | {sm['t']:+.2f} | {sm['mdd']:.1f} |")
                else:
                    L.append(f"| {ms} | {'نعم' if be else 'لا'} | {label} | 0 | - | - | - | - | - | - | - | - |")
                if mode == "zone" and ms == 0.5 and not be:
                    all_rows = tr
    # stability: first vs second half of the period for the main variant
    if all_rows:
        mid = eval_start + (now - eval_start) // 2
        h1 = summarize([t["R_net"] for t in all_rows if t["T"] < mid])
        h2 = summarize([t["R_net"] for t in all_rows if t["T"] >= mid])
        L.append("\n## الثبات عبر الزمن (SMC، درجة ≥ 0.5، بدون BE)\n")
        L.append("| النصف | الصفقات | متوسط R | ربح % |\n|---|---|---|---|")
        for nm, sm in (("الأول", h1), ("الثاني", h2)):
            L.append(f"| {nm} | {sm['n']} | {sm['mean']:+.3f} | {sm['win']:.0f}% |" if sm else f"| {nm} | 0 | - | - |")
        risk = np.mean([t["risk_pct"] for t in all_rows])
        mean_r = np.mean([t["R_net"] for t in all_rows])
        L.append(f"\n- متوسط مسافة الوقف: {risk:.2f}% من سعر الدخول. بهامش 10% ورافعة 5: متوسط العائد على الرصيد للصفقة ≈ "
                 f"{mean_r * risk / 100 * 5 * 10:+.3f}% (R × مسافة الوقف × رافعة × 10%).")
        by_reason = {}
        for t in all_rows:
            by_reason.setdefault(t["reason"], []).append(t["R_net"])
        L.append("\n| سبب الخروج | عدد | متوسط R |\n|---|---|---|")
        for k, v in sorted(by_reason.items()):
            L.append(f"| {k} | {len(v)} | {np.mean(v):+.3f} |")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.md").write_text("\n".join(L), encoding="utf-8")
    if all_rows:
        with gzip.open(Path(a.out) / "trades.csv.gz", "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "T", "side", "score", "R_net", "R_gross", "reason", "tp_hit", "risk_pct"])
            for t in all_rows:
                w.writerow([t["symbol"], t["T"], t["side"], f"{t['score']:.2f}", f"{t['R_net']:.4f}", f"{t['R_gross']:.4f}",
                            t["reason"], t["tp_hit"], f"{t['risk_pct']:.3f}"])
    print("\n".join(L))


if __name__ == "__main__":
    main()
