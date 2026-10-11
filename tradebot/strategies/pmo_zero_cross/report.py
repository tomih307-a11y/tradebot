"""Backtest metrics, tables, sweeps, filter impact, CSV export, markdown report."""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

from . import model as M
from .model import Trade

FILTERS = ["filter_1h_slope", "filter_1h_signal", "filter_eta_max", "filter_4h_slope", "filter_btc"]


def iso(ts) -> str:
    return "" if ts is None else datetime.fromtimestamp(int(ts), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def valid(trades):
    return [t for t in trades if t.exit_reason != M.END_OF_DATA]


def split_trades(trades, split_ts):
    return ([t for t in trades if t.entry_ts < split_ts], [t for t in trades if t.entry_ts >= split_ts])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def _retained(trades):
    """Share of the peak profit kept at exit: sum(gross return) / sum(MFE) over trades that had MFE > 0.
    (An average of per-trade ratios explodes when a trade's MFE is tiny; per-trade values are in the CSV.)"""
    xs = [t for t in trades if t.mfe_pct and t.mfe_pct > 0]
    return sum(t.gross_ret_pct for t in xs) / sum(t.mfe_pct for t in xs) if xs else None


def stats(trades, sl_pct, min_sample) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "low_sample": True}
    rets = [t.net_ret_pct for t in trades]
    wins = [r for r in rets if r > 0]
    avg = sum(rets) / n
    return {
        "n": n,
        "h1_rate": sum(t.h1_cross_bars is not None for t in trades) / n,
        "h4_rate": sum(t.h4_cross_bars is not None for t in trades) / n,
        "win_rate": len(wins) / n,
        "avg_ret": avg,
        "expectancy_r": avg / sl_pct,                       # average net return in units of initial risk
        "avg_mfe": _mean([t.mfe_pct for t in trades]),
        "avg_mae": _mean([t.mae_pct for t in trades]),
        "avg_hours": _mean([t.bars_held for t in trades]) * 0.25,
        "mfe_retained": _retained(trades),
        "sum_pnl": sum(t.pnl_usdt for t in trades),
        "low_sample": n < min_sample,
    }


def _f(x, d=2, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x * 100:.1f}%" if pct else f"{x:.{d}f}"


def _row(label, s):
    label = label.replace("|", " / ")        # a pipe would split the markdown table cell
    if s["n"] == 0:
        return f"| {label} | 0 | - | - | - | - | - | - | - | - | - | |"
    flag = "عينة غير كافية" if s["low_sample"] else ""
    return (f"| {label} | {s['n']} | {_f(s['h1_rate'], pct=True)} | {_f(s['h4_rate'], pct=True)} | "
            f"{_f(s['win_rate'], pct=True)} | {_f(s['avg_ret'], 3)} | {_f(s['expectancy_r'], 3)} | "
            f"{_f(s['avg_mfe'])} | {_f(s['avg_mae'])} | {_f(s['avg_hours'], 1)} | {_f(s['mfe_retained'], 2)} | {flag} |")


TABLE_HEAD = ("| الخلية | الصفقات | عبور الساعة | عبور الـ4س | نسبة الربح | متوسط العائد % | القيمة المتوقعة (R) | "
              "MFE % | MAE % | المدة (ساعة) | MFE المحتفظ | ملاحظة |\n|---|---|---|---|---|---|---|---|---|---|---|---|")


# ───────────── grouping keys (slope = level 1, position = level 2) ─────────────
SLOPE_CLASSES = ["with|with", "with|against", "against|with", "against|against"]
POS_PAIRS = [(a, b) for a in ("above", "below") for b in ("above", "below")]


def key_main(t):
    return t.snap["slope_class"]


def key_sub(t):
    return f"{t.snap['slope_class']}|{t.snap['h1']['pos']}|{t.snap['h4']['pos']}"


def eta_label(eta, edges):
    if math.isinf(eta):
        return "inf"
    for a, b in zip(edges, edges[1:]):
        if a <= eta < b:
            return f"{a}-{b}"
    return f"{edges[-1]}+"


def eta_labels(edges):
    return [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f"{edges[-1]}+", "inf"]


def cell_tables(trades, cfg):
    bt = cfg.pmo_zero_cross.backtest
    sl, ms, edges = cfg.trading.stop_loss_pct, bt.min_sample, list(bt.eta_buckets)
    main = {c: stats([t for t in trades if key_main(t) == c], sl, ms) for c in SLOPE_CLASSES}
    sub = {f"{c}|{p1}|{p4}": stats([t for t in trades if key_sub(t) == f"{c}|{p1}|{p4}"], sl, ms)
           for c in SLOPE_CLASSES for p1, p4 in POS_PAIRS}
    eta = {lab: stats([t for t in trades if eta_label(t.snap["h1"]["eta"], edges) == lab], sl, ms)
           for lab in eta_labels(edges)}
    return {"main": main, "sub": sub, "eta": eta}


def eta_accuracy(trades):
    """Expected ETA_1H at entry vs actual 1H-crossing time (both in 1H candles). Only trades that had a
    finite ETA at entry AND whose 1H actually crossed can be scored."""
    rows = [(t.snap["h1"]["eta"], t.h1_cross_bars / 4.0) for t in trades
            if t.initial_stage == M.WAIT_1H and t.h1_cross_bars is not None and not math.isinf(t.snap["h1"]["eta"])]
    n_fin = sum(1 for t in trades if t.initial_stage == M.WAIT_1H and not math.isinf(t.snap["h1"]["eta"]))
    if not rows:
        return {"n": 0, "n_finite": n_fin}
    err = [actual - eta for eta, actual in rows]               # + = crossed later than predicted
    return {"n": len(rows), "n_finite": n_fin, "mean_err": sum(err) / len(err), "median_err": st.median(err),
            "mean_abs_err": sum(abs(e) for e in err) / len(err), "median_abs_err": st.median([abs(e) for e in err])}


def phase_distribution(trades):
    names = ["WAIT_1H (الدخول→عبور الساعة)", "WAIT_4H (عبور الساعة→عبور الـ4س)", "RIDE (عبور الـ4س→الخروج)"]
    out = []
    tot = sum(sum(t.phase_moves) for t in trades) if trades else 0.0
    for k in range(3):
        mv = sum(t.phase_moves[k] for t in trades)
        mfes = [t.phase_mfe[k] for t in trades if t.phase_mfe[k] is not None]
        out.append({"name": names[k], "sum": mv, "share": (mv / tot) if tot else None,
                    "avg_mfe": _mean(mfes), "n_mfe": len(mfes)})
    return out, tot


# ───────────── CSV ─────────────
def trade_row(t: Trade) -> dict:
    r = {
        "id": t.id, "symbol": t.symbol, "side": t.side,
        "entry_time": iso(t.entry_ts + 900), "entry_price": t.entry_price,
        "exit_time": iso(t.exit_ts + 900) if t.exit_ts is not None else "", "exit_price": t.exit_price,
        "exit_reason": t.exit_reason, "exit_reason_ar": M.REASON_AR.get(t.exit_reason, ""),
        "fail_detail": t.fail_detail or "", "initial_stage": t.initial_stage, "max_stage": t.max_stage,
        "h1_cross_bars": "" if t.h1_cross_bars is None else t.h1_cross_bars,
        "h4_cross_bars": "" if t.h4_cross_bars is None else t.h4_cross_bars,
        "transitions": "|".join(f"{a}:{b}@{p}" for a, b, p in t.transitions),
        "h1_cross_price": "" if t.h1_cross_price is None else t.h1_cross_price,
        "h4_cross_price": "" if t.h4_cross_price is None else t.h4_cross_price,
        "wait_deadline_final": "" if t.wait_deadline is None else t.wait_deadline,
        "wait_deadline_log": "|".join(f"{b}->{d}" for b, d in t.deadline_log),
        "slope_class": t.snap["slope_class"], "n_slopes_with": t.snap["n_slopes_with"],
        "phase1_move_pct": t.phase_moves[0], "phase2_move_pct": t.phase_moves[1], "phase3_move_pct": t.phase_moves[2],
        "phase1_mfe_pct": "" if t.phase_mfe[0] is None else t.phase_mfe[0],
        "phase2_mfe_pct": "" if t.phase_mfe[1] is None else t.phase_mfe[1],
        "phase3_mfe_pct": "" if t.phase_mfe[2] is None else t.phase_mfe[2],
        "bars_held": t.bars_held, "initial_stop": t.initial_stop, "final_stop": t.stop,
        "stop_stage": t.stop_stage, "n_stop_moves": t.n_stop_moves,
        "mfe_pct": t.mfe_pct, "mae_pct": t.mae_pct, "mfe_retained": "" if t.mfe_retained is None else t.mfe_retained,
        "gross_ret_pct": t.gross_ret_pct, "fees_pct": t.fees_pct, "funding_pct": t.funding_pct,
        "net_ret_pct": t.net_ret_pct, "pnl_usdt": t.pnl_usdt,
    }
    for fr in ("h1", "h4"):
        for k, v in t.snap[fr].items():
            r[f"{fr}_{k}"] = v
    for fr in ("btc15", "btc1h"):
        for k in ("pmo", "signal", "slope"):
            r[f"{fr}_{k}"] = t.snap.get(fr, {}).get(k, "")
    return r


def write_csv(trades, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [trade_row(t) for t in trades]
    with open(path, "w", newline="") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ───────────── variants (sweeps / filters) ─────────────
def _variant_line(label, sel, val, base_sel, min_sample):
    ok = (val["n"] >= min_sample and val["avg_ret"] > 0 and sel["n"] > 0 and sel["avg_ret"] >= base_sel)
    if val["n"] < min_sample:
        verdict = "عينة غير كافية"
    else:
        verdict = "مقبول" if ok else "مرفوض"
    return (f"| {label} | {sel['n']} | {_f(sel.get('avg_ret'), 3)} | {val['n']} | {_f(val.get('avg_ret'), 3)} | {verdict} |")


VARIANT_HEAD = ("| القيمة | صفقات (اختيار) | متوسط العائد % (اختيار) | صفقات (تحقق) | متوسط العائد % (تحقق) | القرار |\n"
                "|---|---|---|---|---|---|")


def write_report(cfg, run_fn, base_engine, start_ts, end_ts, out_dir, symbols=None) -> dict:
    """run_fn(cfg_override) -> PMOEngine. base_engine is the run with the config as written."""
    s = cfg.pmo_zero_cross
    bt = s.backtest
    sl, ms = cfg.trading.stop_loss_pct, bt.min_sample
    split_ts = start_ts + int((end_ts - start_ts) * bt.split_selection_pct / 100.0)
    trades = valid(base_engine.closed)
    eod = [t for t in base_engine.closed if t.exit_reason == M.END_OF_DATA]
    sel, val = split_trades(trades, split_ts)
    L = []
    L.append("# تقرير الباكتست — تسلسل عبور الصفر لـ PMO\n")
    L.append(f"- الفترة: {iso(start_ts)} → {iso(end_ts)}")
    L.append(f"- حد التقسيم ({bt.split_selection_pct}% اختيار / {100 - bt.split_selection_pct}% تحقق): {iso(split_ts)} (حسب وقت الدخول)")
    L.append(f"- العملات: {', '.join(sorted(symbols if symbols is not None else {t.symbol for t in base_engine.closed}))}")
    L.append(f"- الرسوم/جهة: {cfg.trading.taker_fee_pct}% · الانزلاق: {bt.slippage_pct}% · وقف ابتدائي: {sl}%")
    L.append(f"- المهلة: min={s.exits.min_wait_1h} شمعة · ETA×{s.exits.wait_eta_mult} · سقف={s.exits.max_wait_1h_cap} شمعة · مفعّلة={s.exits.fail_1h_timeout}")
    L.append(f"- صفقات مغلقة قسراً بنهاية البيانات (مستبعدة من الإحصاءات): {len(eod)}")
    rej = {}
    for r in base_engine.rejected:
        rej[r["reason"]] = rej.get(r["reason"], 0) + 1
    L.append("- إشارات مرفوضة: " + (", ".join(f"{('الحد الأقصى ممتلئ' if k == 'max_open_full' else k)}={v}" for k, v in sorted(rej.items())) or "لا شيء"))
    L.append(f"- معيار القرار: يُقبل المتغير فقط إذا كان متوسط عائد التحقق > 0 مع ≥ {ms} صفقة، ومتوسط الاختيار ≥ الأساس.\n")

    L.append("## ملخص عام: الاختيار | التحقق | الكل\n")
    L.append(TABLE_HEAD)
    for lab, grp in (("اختيار", sel), ("تحقق", val), ("الكل", trades)):
        L.append(_row(lab, stats(grp, sl, ms)))

    L.append("\n## مقاييس التسلسل\n")
    for lab, grp in (("الكل", trades), ("اختيار", sel), ("تحقق", val)):
        a = [t for t in grp if t.initial_stage == M.WAIT_1H]
        b = [t for t in grp if t.initial_stage != M.RIDE]
        c1 = [t.h1_cross_bars for t in a if t.h1_cross_bars is not None]
        c4 = [t.h4_cross_bars for t in b if t.h4_cross_bars is not None]
        L.append(f"**{lab}** — كل الصفقات: عبرت الساعة {_f(stats(grp, sl, ms).get('h1_rate'), pct=True)}، عبرت الـ4س {_f(stats(grp, sl, ms).get('h4_rate'), pct=True)}.")
        L.append(f"  - الصفقات التي بدأت WAIT_1H ({len(a)}): عبرت الساعة {_f(len(c1) / len(a) if a else None, pct=True)}، "
                 f"عدد شموع 15د حتى العبور: متوسط {_f(_mean(c1), 1)} · وسيط {_f(_median(c1), 1)}")
        L.append(f"  - الصفقات التي بدأت قبل RIDE ({len(b)}): عبرت الـ4س {_f(len(c4) / len(b) if b else None, pct=True)}، "
                 f"شموع 15د من الدخول حتى العبور: متوسط {_f(_mean(c4), 1)} · وسيط {_f(_median(c4), 1)}")

    L.append("\n## دقة ETA_1H (المتوقع لحظة الدخول مقابل الفعلي، بشموع الساعة؛ + = عبرت متأخرة عن التوقع)\n")
    L.append("| المجموعة | صفقات ETA منتهٍ | عبرت فعلاً | متوسط الخطأ | وسيط الخطأ | متوسط |الخطأ| | وسيط |الخطأ| |\n|---|---|---|---|---|---|---|")
    for lab, grp in (("الكل", trades), ("اختيار", sel), ("تحقق", val)):
        e = eta_accuracy(grp)
        L.append(f"| {lab} | {e['n_finite']} | {e['n']} | {_f(e.get('mean_err'))} | {_f(e.get('median_err'))} | "
                 f"{_f(e.get('mean_abs_err'))} | {_f(e.get('median_abs_err'))} |")
    L.append("\n(الصفقات التي خرجت قبل العبور لا يمكن تقييم ETA لها، فالمقياس متحيز نحو العبور السريع.)")

    L.append("\n## توزيع الربح على المراحل (حركة السعر لصالح الصفقة، % من سعر الدخول)\n")
    L.append("| المجموعة | المرحلة | مجموع الحركة % | نسبتها من الإجمالي | متوسط MFE في المرحلة % | صفقات دخلت المرحلة |\n|---|---|---|---|---|---|")
    for lab, grp in (("الكل", trades), ("اختيار", sel), ("تحقق", val)):
        dist, tot = phase_distribution(grp)
        for d in dist:
            L.append(f"| {lab} | {d['name']} | {_f(d['sum'])} | {_f(d['share'], pct=True)} | {_f(d['avg_mfe'], 3)} | {d['n_mfe']} |")
        L.append(f"| {lab} | الإجمالي | {_f(tot)} | | | |")

    L.append("\n## العائد حسب أعلى مرحلة وصلتها الصفقة\n")
    L.append(TABLE_HEAD)
    for lab, st_ in (("فشل عند WAIT_1H", M.WAIT_1H), ("فشل عند WAIT_4H", M.WAIT_4H), ("وصل RIDE", M.RIDE)):
        L.append(_row(lab, stats([t for t in trades if t.max_stage == st_], sl, ms)))

    L.append("\n## الجداول حسب الحالة الابتدائية (المستوى الأول: ميل الساعة | ميل الـ4س)\n")
    for lab, grp in (("الاختيار", sel), ("التحقق", val)):
        tabs = cell_tables(grp, cfg)
        L.append(f"\n### الجدول الرئيسي: ميل الساعة × ميل الـ4س — {lab}\n")
        L.append(TABLE_HEAD)
        for cell in SLOPE_CLASSES:
            L.append(_row(f"**{cell}**", tabs["main"][cell]))
            for p1, p4 in POS_PAIRS:                                  # level 2: position of 1H / 4H vs zero
                L.append(_row(f"&nbsp;&nbsp;↳ موقع الساعة {p1} · الـ4س {p4}", tabs["sub"][f"{cell}|{p1}|{p4}"]))
        L.append(f"\n### شرائح ETA_1H (شموع ساعة) — {lab}\n")
        L.append(TABLE_HEAD)
        for cell, sv in tabs["eta"].items():
            L.append(_row(cell, sv))
    L.append("\n(with = الميل مع اتجاه الصفقة، against = ضده؛ above = في جهة الصفقة من الصفر)")

    L.append("\n## سبب الخروج\n")
    L.append(TABLE_HEAD)
    for r in (M.STOP_INITIAL, M.STOP_TRAIL_1, M.STOP_TRAIL_2, M.STOP_TRAIL_3):
        L.append(_row(f"{M.REASON_AR[r]} ({r})", stats([t for t in trades if t.exit_reason == r], sl, ms)))
    for det, ar in (("1h_slope", "ميل الساعة"), ("pmo15_back", "عودة PMO_15"), ("timeout", "مهلة")):
        L.append(_row(f"فشل الساعة — {ar}", stats([t for t in trades if t.exit_reason == M.FAIL_1H and t.fail_detail == det], sl, ms)))
    L.append(_row("فشل الـ4 ساعات", stats([t for t in trades if t.exit_reason == M.FAIL_4H], sl, ms)))
    L.append("\n(عمود «MFE المحتفظ» = مجموع العائد الإجمالي عند الخروج ÷ مجموع أقصى ربح وصلته الصفقات؛ القيمة لكل صفقة في CSV)")

    # ───────── sweeps ─────────
    base_sel = stats(sel, sl, ms).get("avg_ret", -1e9) if sel else -1e9

    def variant(label, **over):
        eng = run_fn(_override(cfg, **over))
        a, b = split_trades(valid(eng.closed), split_ts)
        return _variant_line(label, stats(a, sl, ms), stats(b, sl, ms), base_sel, ms)

    L.append("\n## مقارنة مضاعفات الوقف المتحرك (متغير واحد في كل مرة)\n")
    for key, vals in (("mult_1", bt.sweep_mult_1), ("mult_2", bt.sweep_mult_2), ("mult_3", bt.sweep_mult_3)):
        L.append(f"\n### TRAIL_{key.upper()}\n")
        L.append(VARIANT_HEAD)
        for v in vals:
            L.append(variant(f"{v}" + (" (الحالي)" if v == s.trail.get(key) else ""),
                             **{f"pmo_zero_cross.trail.{key}": v}))

    L.append("\n## مقارنة المهلة (WAIT_ETA_MULT)\n")
    L.append(VARIANT_HEAD)
    for v in bt.sweep_wait_eta_mult:
        L.append(variant(f"WAIT_ETA_MULT={v}" + (" (الحالي)" if v == s.exits.wait_eta_mult and s.exits.fail_1h_timeout else ""),
                         **{"pmo_zero_cross.exits.wait_eta_mult": v, "pmo_zero_cross.exits.fail_1h_timeout": True}))
    L.append(variant("بدون مهلة (الخروج بالميل فقط)", **{"pmo_zero_cross.exits.fail_1h_timeout": False}))

    L.append("\n## أثر الفلاتر (مع الفلتر مقابل بدونه)\n")
    L.append(VARIANT_HEAD)
    L.append(_variant_line("بدون فلاتر (الأساس)", stats(sel, sl, ms), stats(val, sl, ms), base_sel, ms))
    for fl in FILTERS:
        L.append(variant(fl.upper(), **{f"pmo_zero_cross.filters.{fl}": True}))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pmo_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    write_csv(base_engine.closed, out / "pmo_trades.csv")
    with open(out / "pmo_rejected.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "symbol", "side", "reason"])
        for r in base_engine.rejected:
            w.writerow([iso(r["ts"] + 900), r["symbol"], r["side"], r["reason"]])
    (out / "last_backtest_cells.json").write_text(json.dumps(cell_tables(trades, cfg), indent=1, sort_keys=True), encoding="utf-8")
    return {"report": str(out / "pmo_report.md"), "trades": len(trades)}


def _override(cfg, **kw):
    """_override(cfg, **{"pmo_zero_cross.trail.mult_1": 2.0})"""
    from tradebot.config import Section
    d = cfg.to_dict()
    for path, value in kw.items():
        cur = d
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value
    return Section(d, "root")
