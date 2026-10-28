"""Walk-forward test of the user's rules (4H order block + 15m PMO signal-cross/first-zero-touch) with an open target and trailing stop.

Fidelity: triggers come from the LIVE code path (RulesAgent.analyze on the same windows the bot would load: 15m x700, 4H x300, closed
candles only). A cheap pre-filter on the full-history PMO only decides where to call it. Entry = next 15m open (+slippage), stop beyond
the OB, no targets; the stop trails by `open_trail_atr` x ATR(1H) once `open_trail_activation_r` x risk in profit, updated on each 15m close
and effective from the next candle; adverse-first inside a candle; unprofitable trades are closed after max_hold_sec (as the exit agent does).
NOT simulated: news, the other agents' votes, the 2-concurrent-trades cap (one position per symbol), funding.
Config is FIXED (config.toml defaults). Primary result = last 40% of the period; first 60% shown as the design window. A small sensitivity
grid is reported for transparency only.
Usage: python tools/rules_study.py --out reports/rules --days 150 --top 40
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smc_study import summarize                                        # noqa: E402
from smc_study2 import boot_ci                                          # noqa: E402
from tradebot.config import load_config                                 # noqa: E402
from tradebot.data.klines import get_klines                              # noqa: E402
from tradebot.indicators import atr_rma, pmo                             # noqa: E402
from tradebot.mexc.client import MexcClient                              # noqa: E402
from tradebot.strategies.multi_agent.rules import RulesAgent, cross_above_zero, signal_cross_then_zero_touch   # noqa: E402
from tradebot.strategies.multi_agent.types import Candidate              # noqa: E402

M15, H1, H4 = 900, 3600, 14400


def simulate_open(s, stop0, o, h, l, c, atr, j0, trail_atr, act_r, max_hold, loser_bars, fee_pct, slip_pct):
    """Open-target trade entered at the open of candle j0. -> dict or {'filled': False}."""
    n = len(c)
    if j0 >= n:
        return {"filled": False}
    px_in = float(o[j0]) * (1 + s * slip_pct / 100.0)
    risk = s * (px_in - stop0)
    if risk <= 0:
        return {"filled": False}
    stop, last = stop0, min(n - 1, j0 + max_hold)
    exit_px, reason, k_exit = None, "max_hold", last
    for k in range(j0, last + 1):
        adverse = l[k] if s == 1 else h[k]
        if s * (adverse - stop) <= 0:
            exit_px = float(o[k]) if (k > j0 and s * (o[k] - stop) <= 0) else stop
            exit_px *= (1 - s * slip_pct / 100.0)
            reason, k_exit = ("trail_stop" if s * (stop - stop0) > 0 else "stop"), k
            break
        price = float(c[k])
        upnl = s * (price - px_in)
        if k - j0 >= loser_bars and upnl <= 0:
            exit_px, reason, k_exit = price * (1 - s * slip_pct / 100.0), "time_loser", k
            break
        if upnl >= act_r * risk and np.isfinite(atr[k]):
            t = price - s * trail_atr * atr[k]
            if s * (t - stop) > 0:
                stop = t
    if exit_px is None:
        exit_px = float(c[last]) * (1 - s * slip_pct / 100.0)
    gross = s * (exit_px - px_in) / risk
    fees = fee_pct / 100.0 * (px_in + exit_px) / risk
    return {"filled": True, "R_gross": gross, "R_net": gross - fees, "reason": reason, "exit_j": k_exit,
            "bars": k_exit - j0, "risk_pct": risk / px_in * 100.0}


def find_triggers(cfg, sym, k15, k4, k1, eval_start, mitigation, conditions, htf):
    """Calls the live RulesAgent on the windows the bot would see. -> list of dict(j, T, side, stop, zone)."""
    ind, rc = cfg.pmo_zero_cross.indicator, cfg.agents.rules
    p, sg, _ = pmo(k15["close"], ind.roc_smooth_len, ind.roc_scale, ind.pmo_smooth_len, ind.signal_len)
    p, sg = p.copy(), sg.copy()
    p[: int(ind.warmup_bars)] = np.nan
    sg[: int(ind.warmup_bars)] = np.nan
    aligned = "pmo_cross_zero_aligned" in conditions
    agent = RulesAgent(cfg)
    agent.c = rc.override(mitigation=mitigation, conditions=list(conditions), htf=htf)
    t15, close4, close1 = k15["time"], k4["time"] + H4, k1["time"] + H1
    phtf = None
    if aligned:                                           # full-history HTF PMO mapped to each 15m close: a cheap pre-filter only
        hk, hstep = (k1, H1) if htf == "Min60" else (k4, H4)
        ph, _, _ = pmo(hk["close"], ind.roc_smooth_len, ind.roc_scale, ind.pmo_smooth_len, ind.signal_len)
        ph = ph.copy()
        ph[: int(ind.warmup_bars)] = np.nan
        ix = np.searchsorted(hk["time"] + hstep, t15 + M15, side="right") - 1
        phtf = np.where(ix >= 0, ph[np.clip(ix, 0, None)], np.nan)
    out = []
    bars15, bars4 = int(rc.bars_15m), (int(rc.htf_bars) if aligned else 300)
    for j in range(int(ind.warmup_bars) + 1, len(t15)):
        T = int(t15[j]) + M15
        if T < eval_start:
            continue
        if aligned:
            ok = any(cross_above_zero(p[: j + 1], sg[: j + 1], s_, int(rc.pmo_cross_window_bars)) is not None and phtf[j] * s_ > 0 for s_ in (1, -1))
        else:
            ok = any(signal_cross_then_zero_touch(p[: j + 1], sg[: j + 1], s_, int(rc.pmo_cross_max_age_bars), int(rc.pmo_touch_window_bars)) for s_ in (1, -1))
        if not ok:
            continue
        i4 = int(np.searchsorted(close4, T, side="right"))
        if i4 < 60 or j + 1 < bars15:
            continue
        w4 = {f: k4[f][max(0, i4 - bars4):i4] for f in ("time", "open", "high", "low", "close")}
        w15 = {f: k15[f][j + 1 - bars15:j + 1] for f in ("time", "open", "high", "low", "close")}
        w1 = None
        if aligned:
            i1 = int(np.searchsorted(close1, T, side="right"))
            w1 = {f: k1[f][max(0, i1 - int(rc.htf_bars)):i1] for f in ("time", "open", "high", "low", "close")}
        rec = agent.analyze(Candidate(sym, float(k15["close"][j]), 0, 0, 0), w1, w4, T, w15)
        if rec:
            out.append({"j": j, "T": T, "side": rec.side, "stop": rec.stop, "zone": rec.zone})
    return out


def replay(data, trig, params, cfg, reverse=False, t_lo=0, t_hi=10 ** 12):
    trades = []
    for sym, d in data.items():
        busy = -1
        for tr in trig.get(sym, []):
            if tr["j"] + 1 <= busy or not (t_lo <= tr["T"] < t_hi):
                continue
            s = 1 if tr["side"] == "long" else -1
            stop0, ref = tr["stop"], float(d["o"][tr["j"] + 1]) if tr["j"] + 1 < len(d["o"]) else None
            if ref is None:
                continue
            if reverse:
                dist = s * (ref - stop0)
                s, stop0 = -s, ref + s * dist
            r = simulate_open(s, stop0, d["o"], d["h"], d["l"], d["c"], d["atr"], tr["j"] + 1, params["trail_atr"], params["act_r"],
                              params["max_hold"], params["loser_bars"], cfg.trading.taker_fee_pct, params["slip"])
            if not r["filled"]:
                continue
            busy = r["exit_j"]
            trades.append({"symbol": sym, "T": tr["T"], "side": "long" if s == 1 else "short", **r})
    return sorted(trades, key=lambda t: t["T"])


def stats(trades):
    sm = summarize([t["R_net"] for t in trades])
    if not sm:
        return None
    sm["gross"] = float(np.mean([t["R_gross"] for t in trades]))
    sm["bars"] = float(np.mean([t["bars"] for t in trades])) * 15 / 60
    return sm


def row(label, sm):
    if not sm:
        return f"| {label} | 0 | - | - | - | - | - | - | - |"
    return (f"| {label} | {sm['n']} | {sm['mean']:+.3f} | {sm['gross']:+.3f} | {sm['win']:.0f}% | {sm['pf']:.2f} | {sm['t']:+.2f} | "
            f"{sm['mdd']:.1f} | {sm['bars']:.1f} |")


HEAD = "| | الصفقات | صافي R | إجمالي R | ربح % | PF | t | أقصى تراجع R | متوسط الاحتفاظ (ساعة) |\n|---|---|---|---|---|---|---|---|---|"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/rules")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--max-hold-days", type=int, default=10)
    ap.add_argument("--val-pct", type=int, default=40)
    ap.add_argument("--conditions", default="price_in_ob_4h,pmo15_zero_touch")
    ap.add_argument("--htf", default="Min60", help="Min60 | Hour4 (used by pmo_cross_zero_aligned)")
    ap.add_argument("--mitigations", default="wick,close")
    a = ap.parse_args()
    cfg = load_config(a.config)
    client = MexcClient(cfg.mexc)
    now = int(time.time())
    eval_start = now - a.days * 86400
    split = eval_start + (now - eval_start) * (100 - a.val_pct) // 100
    ex, rc = cfg.agents.exit, cfg.agents.rules
    conditions, mits = [c.strip() for c in a.conditions.split(",")], [m.strip() for m in a.mitigations.split(",")]
    base = {"trail_atr": float(ex.open_trail_atr), "act_r": float(ex.open_trail_activation_r), "slip": a.slip,
            "max_hold": a.max_hold_days * 96, "loser_bars": int(ex.max_hold_sec) // M15}
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    symbols = [t["symbol"] for t in tick[: a.top]]
    dd = cfg.pmo_zero_cross.backtest.data_dir
    data, trig = {}, {m: {} for m in mits}
    n_ev = 0
    for n_, sym in enumerate(symbols, 1):
        try:
            k15 = get_klines(client, dd, sym, "Min15", eval_start - 900 * M15, now)
            k4 = get_klines(client, dd, sym, "Hour4", eval_start - 135 * 86400, now)
            k1 = get_klines(client, dd, sym, "Min60", eval_start - 45 * 86400, now)
        except Exception as e:
            print(f"  {sym}: data failed {e}", flush=True)
            continue
        if len(k15["time"]) < 1200 or len(k4["time"]) < 100 or len(k1["time"]) < 100:
            continue
        a1 = atr_rma(k1["high"], k1["low"], k1["close"], int(cfg.agents.technical.atr_len))
        idx = np.searchsorted(k1["time"] + H1, k15["time"] + M15, side="right") - 1
        atr15 = np.where(idx >= 0, a1[np.clip(idx, 0, None)], np.nan)
        data[sym] = {"o": k15["open"], "h": k15["high"], "l": k15["low"], "c": k15["close"], "atr": atr15}
        for m in mits:
            trig[m][sym] = find_triggers(cfg, sym, k15, k4, k1, eval_start, m, conditions, a.htf)
        n_ev += len(trig[mits[0]][sym])
        print(f"  [{n_}/{len(symbols)}] {sym}: {len(trig[mits[0]][sym])} triggers", flush=True)

    L = [f"# اختبار شروطك: {' + '.join(conditions)}" + (f" (الفريم الأكبر: {a.htf})" if "pmo_cross_zero_aligned" in conditions else "") + " — هدف مفتوح ووقف متحرك", ""]
    scope = f"كل عقود USDT المتاحة ({len(tick)})" if a.top >= len(tick) else f"أعلى {a.top} سيولة"
    L.append(f"- العقود: {len(data)} مُحلَّلة من {scope} (استُبعد ما ليس له تاريخ 15د كافٍ) · عقود موجودة اليوم فقط ⇒ انحياز بقاء ممكن · آخر {a.days} يوماً · إجمالي الإشارات ({mits[0]}): {n_ev}")
    L.append(f"- **الإعدادات ثابتة** من config.toml (لم تُضبط على النتائج): mitigation={mits[0]}، ob_method={rc.ob_method}، وقف خلف الـOB بـ{rc.stop_buffer_atr}×ATR(4س)، "
             f"وقف متحرك {base['trail_atr']}×ATR(ساعة) بعد ربح {base['act_r']}R، إغلاق الخاسرة بعد {int(ex.max_hold_sec) // 3600} ساعة، أقصى احتفاظ {a.max_hold_days} أيام.")
    L.append(f"- دخول عند فتح الشمعة التالية (+{a.slip}% انزلاق)، رسوم Taker {cfg.trading.taker_fee_pct}% للجهتين، صفقة واحدة لكل عملة. R = المخاطرة الأولية.")
    L.append("- **غير مُحاكى:** الأخبار، تصويت الوكلاء الآخرين، حد صفقتين متزامنتين، التمويل. «إجمالي» = قبل الرسوم والانزلاق.\n")
    main_trades = replay(data, trig[mits[0]], base, cfg)
    sel = [t for t in main_trades if t["T"] < split]
    val = [t for t in main_trades if t["T"] >= split]
    rev_val = [t for t in replay(data, trig[mits[0]], base, cfg, reverse=True) if t["T"] >= split]
    L.append("## النتيجة الأساسية (نافذة التحقق = آخر %d%% من الفترة)\n" % a.val_pct)
    L.append(HEAD)
    L.append(row("شروطك (التحقق)", stats(val)))
    L.append(row("أساس: عكس الاتجاه نفس المسافة (التحقق)", stats(rev_val)))
    L.append(row("شروطك (نافذة التصميم، أول %d%%)" % (100 - a.val_pct), stats(sel)))
    lo, hi = boot_ci(val)
    z = stats(val)
    L.append(f"\n- فترة ثقة 95% لمتوسط صافي R في التحقق (إعادة عيّنة بالأيام): [{lo:+.3f} ، {hi:+.3f}]")
    verdict = "عدد الصفقات غير كافٍ للحكم"
    if z and z["n"] >= 100:
        r_ = stats(rev_val)
        ok = z["mean"] > 0 and lo > 0 and (not r_ or z["mean"] > r_["mean"])
        verdict = "اجتاز (متوسط موجب وحد ثقة أدنى موجب ويتفوق على العكس)" if ok else "لم يجتز: " + (
            "متوسط صافي R غير موجب" if z["mean"] <= 0 else "حد الثقة الأدنى ≤ 0 أو لا يتفوق على عكس الاتجاه")
    if val:
        risk = float(np.mean([t["risk_pct"] for t in val]))
        L.append(f"- مسافة الوقف المتوسطة {risk:.2f}% ⇒ بهامش 10% ورافعة 5: ≈ {z['mean'] * risk / 100 * 5 * 10:+.3f}% من الرصيد للصفقة")
        by_sym = {}
        for t in val:
            by_sym[t["symbol"]] = by_sym.get(t["symbol"], 0.0) + t["R_net"]
        tot = sum(by_sym.values())
        top3 = sorted(by_sym.values(), reverse=True)[:3]
        L.append(f"- التركّز: أفضل 3 عملات تمثل {sum(top3):+.1f}R من مجموع {tot:+.1f}R في التحقق (عملات: {len(by_sym)})")
        L.append("\n| سبب الخروج | عدد | متوسط R |\n|---|---|---|")
        for k, v in sorted({r_["reason"]: [] for r_ in val}.items()):
            xs = [t["R_net"] for t in val if t["reason"] == k]
            L.append(f"| {k} | {len(xs)} | {np.mean(xs):+.3f} |")
        rank = {sym: i + 1 for i, sym in enumerate(symbols)}
        L.append("\n| شريحة السيولة (ترتيب حجم تداول 24س اليوم) | العقود | الصفقات | صافي R | إجمالي R | ربح % |\n|---|---|---|---|---|---|")
        for nm, lo_, hi_ in (("1–40", 1, 40), ("41–150", 41, 150), ("151–400", 151, 400), ("401+", 401, 10 ** 6)):
            xs = [t for t in val if lo_ <= rank[t["symbol"]] <= hi_]
            if xs:
                L.append(f"| {nm} | {len({t['symbol'] for t in xs})} | {len(xs)} | {np.mean([t['R_net'] for t in xs]):+.3f} | "
                         f"{np.mean([t['R_gross'] for t in xs]):+.3f} | {np.mean([t['R_net'] > 0 for t in xs]) * 100:.0f}% |")
            else:
                L.append(f"| {nm} | 0 | 0 | - | - | - |")
        L.append("\n| الاتجاه | عدد | متوسط R |\n|---|---|---|")
        for sd in ("long", "short"):
            xs = [t["R_net"] for t in val if t["side"] == sd]
            L.append(f"| {sd} | {len(xs)} | {np.mean(xs):+.3f} |" if xs else f"| {sd} | 0 | - |")
    L.append("\n## حساسية الإعدادات (للشفافية فقط — لا يُبنى عليها اختيار؛ كل صف على نافذتي التصميم والتحقق)\n")
    L.append("| mitigation | وقف متحرك ATR | تفعيل R | تصميم: n / صافي R | تحقق: n / صافي R |\n|---|---|---|---|---|")
    for mit in mits:
        for ta in (1.5, 2.0, 3.0):
            for ar in (0.5, 1.0, 1.5):
                tr_ = replay(data, trig[mit], {**base, "trail_atr": ta, "act_r": ar}, cfg)
                s1, s2 = stats([t for t in tr_ if t["T"] < split]), stats([t for t in tr_ if t["T"] >= split])
                f = lambda x: f"{x['n']} / {x['mean']:+.3f}" if x else "0 / -"
                L.append(f"| {mit} | {ta} | {ar} | {f(s1)} | {f(s2)} |")
    L.append(f"\n## الحكم الآلي\n\n**{verdict}**  (المعيار: متوسط صافي R > 0 في التحقق، والحد الأدنى لفترة الثقة > 0، ويتفوق على عكس الاتجاه)")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.md").write_text("\n".join(L), encoding="utf-8")
    if main_trades:
        with gzip.open(Path(a.out) / "trades.csv.gz", "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "T", "side", "R_net", "R_gross", "reason", "bars15", "risk_pct"])
            for t in main_trades:
                w.writerow([t["symbol"], t["T"], t["side"], f"{t['R_net']:.4f}", f"{t['R_gross']:.4f}", t["reason"], t["bars"], f"{t['risk_pct']:.3f}"])
    print("\n".join(L))


if __name__ == "__main__":
    main()
