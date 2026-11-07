"""Forward (paper) test of the cross-sectional momentum hypotheses registered in docs/PREREG_FORWARD.md. Runs once a day.

Each run: (1) settles every open tranche that is 7 days old, using MEXC last prices; (2) opens today's tranche for each signal
(mom30 = H1, mom7 = H2, random = control); (3) writes state.json, results.csv and summary.md. No orders are ever sent: this is bookkeeping.
Usage: python tools/forward_xsec.py --dir reports/forward
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xsec import block, block_boot_ci, hac_t                     # noqa: E402
from tradebot.config import load_config                          # noqa: E402
from tradebot.data.klines import drop_unclosed, fetch_range      # noqa: E402
from tradebot.mexc.client import MexcClient                      # noqa: E402

DAY = 86400
SIGNALS = ("mom30", "mom7", "random")
COST_LEG = 2 * (0.04 + 0.05) / 100.0           # fixed by the pre-registration: 0.18% per leg per tranche
MIN_ELIGIBLE = 25
QUANTILE = 0.2
MIN_SETTLED = 90
MAX_TRANCHES = 150


# ───────────────────────────── pure functions (tested) ─────────────────────────────
def select_portfolio(scores: dict, q: float = QUANTILE):
    """scores: symbol -> score. Top q by score = long, bottom q = short (deterministic tie-break by symbol)."""
    items = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
    k = max(1, int(np.ceil(q * len(items))))
    return [s for s, _ in items[-k:]], [s for s, _ in items[:k]]


def settle(tr: dict, prices: dict, cost_leg: float = COST_LEG, min_present: float = 0.8):
    """Fill exit data and the net return of a tranche. Returns the updated tranche (status settled | failed)."""
    lr = [prices[s] / e - 1 for s, e in tr["longs"].items() if s in prices and prices[s] > 0]
    sr = [prices[s] / e - 1 for s, e in tr["shorts"].items() if s in prices and prices[s] > 0]
    n_all = len(tr["longs"]) + len(tr["shorts"])
    if len(lr) + len(sr) < min_present * n_all or not lr or not sr:
        tr.update(status="failed", note="missing prices at exit")
        return tr
    gross = (np.mean(lr) - np.mean(sr)) / 2.0
    tr.update(status="settled", gross=float(gross), net=float(gross - cost_leg), n_long=len(lr), n_short=len(sr))
    return tr


def summarize_signal(nets: np.ndarray, control: np.ndarray | None):
    """Pre-registered statistics for one signal."""
    b = block(nets)
    if not b:
        return None
    b["t_hac"] = hac_t(nets, 14)
    lo, hi = block_boot_ci(nets, 14)
    b["ci_lo"], b["ci_hi"] = lo * 100, hi * 100
    b["control_mean_pct"] = float(control.mean() * 100) if control is not None and len(control) else None
    return b


def verdict(sig: str, st):
    if not st or st["n"] < MIN_SETTLED:
        return f"لم يبلغ {MIN_SETTLED} شريحة مُسوّاة بعد ({0 if not st else st['n']})"
    need = 1.96 if sig == "mom30" else 2.24
    ok = (st["exp_pct"] > 0 and st["t_hac"] >= need and st["ci_lo"] > 0
          and (st["control_mean_pct"] is None or st["exp_pct"] > st["control_mean_pct"]))
    return "نجحت (حسب المعيار المسجّل)" if ok else ("مرفوضة (بلغت الحد الأقصى دون تحقق المعيار)" if st["n"] >= MAX_TRANCHES else "لم تتحقق شروط النجاح بعد")


# ───────────────────────────── IO ─────────────────────────────
def load_state(d: Path):
    p = d / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"start": None, "tranches": []}


def open_tranches(client, state, now, top, top_m, seed_date):
    """Compute signals from the last CLOSED daily candles and open one tranche per signal for today (idempotent)."""
    today = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
    if any(t["date"] == today for t in state["tranches"]):
        return 0, "already opened today"
    tick = [t for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")]
    tick.sort(key=lambda t: -float(t.get("amount24") or 0))
    tick = tick[:top]
    prices = {t["symbol"]: float(t["lastPrice"]) for t in tick}
    feats = {}
    for t in tick:
        sym = t["symbol"]
        try:
            k = drop_unclosed(fetch_range(client, sym, "Day1", int(now // DAY * DAY) - 45 * DAY, now), "Day1", now)
        except Exception:
            continue
        c, a = k["close"], k["amount"]
        if len(c) < 31 or not np.isfinite(c[-31:]).all():
            continue
        feats[sym] = {"mom30": float(c[-1] / c[-31] - 1), "mom7": float(c[-1] / c[-8] - 1), "turn30": float(np.nanmean(a[-30:]))}
    elig = sorted(feats, key=lambda s: -feats[s]["turn30"])[:top_m]
    if len(elig) < MIN_ELIGIBLE:
        return 0, f"only {len(elig)} eligible contracts"
    rng = np.random.default_rng(int(seed_date.replace("-", "")))
    rand = {s: float(rng.random()) for s in elig}
    n = 0
    for sig in SIGNALS:
        sc = rand if sig == "random" else {s: feats[s][sig] for s in elig}
        longs, shorts = select_portfolio(sc)
        state["tranches"].append({"date": today, "signal": sig, "entry_ts": now, "status": "open",
                                  "longs": {s: prices[s] for s in longs if s in prices}, "shorts": {s: prices[s] for s in shorts if s in prices}})
        n += 1
    return n, "ok"


def write_outputs(d: Path, state):
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(state, indent=1))
    rows = [t for t in state["tranches"] if t["status"] == "settled"]
    with (d / "results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "signal", "entry_ts", "exit_ts", "gross", "net", "n_long", "n_short"])
        for t in rows:
            w.writerow([t["date"], t["signal"], t["entry_ts"], t.get("exit_ts"), f"{t['gross']:.6f}", f"{t['net']:.6f}", t["n_long"], t["n_short"]])
    nets = {s: np.array([t["net"] for t in sorted(rows, key=lambda t: t["entry_ts"]) if t["signal"] == s]) for s in SIGNALS}
    L = [f"# اختبار أمامي: الزخم المقطعي ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})", "",
         "التسجيل المسبق: `docs/PREREG_FORWARD.md`. ورقي فقط: لا أوامر. التكلفة الثابتة 0.18% لكل ساق.", "",
         f"- شرائح مفتوحة: {sum(1 for t in state['tranches'] if t['status'] == 'open')} · مُسوّاة: {len(rows)} · فاشلة: {sum(1 for t in state['tranches'] if t['status'] == 'failed')}",
         "", "| الإشارة | مُسوّاة | فوز % (لازم) | payoff | متوسط صافٍ % | t (Newey-West) | فترة 95% | متوسط الشاهد % | الحالة |", "|---|---|---|---|---|---|---|---|---|"]
    for s, hyp in (("mom30", "H1"), ("mom7", "H2"), ("random", "شاهد")):
        st = summarize_signal(nets[s], nets["random"] if s != "random" else None) if len(nets[s]) >= 10 else None
        if st:
            L.append(f"| {s} ({hyp}) | {st['n']} | {st['win'] * 100:.0f} ({st['required_win'] * 100:.0f}) | {st['payoff']:.2f} | {st['exp_pct']:+.3f} | {st['t_hac']:+.2f} | "
                     f"[{st['ci_lo']:+.2f}, {st['ci_hi']:+.2f}] | {'–' if st['control_mean_pct'] is None else format(st['control_mean_pct'], '+.3f')} | {verdict(s, st) if s != 'random' else '–'} |")
        else:
            L.append(f"| {s} ({hyp}) | {len(nets[s])} | – | – | – | – | – | – | {verdict(s, None) if s != 'random' else '–'} |")
    (d / "summary.md").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/forward")
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--top-m", type=int, default=150)
    ap.add_argument("--hold", type=int, default=7)
    ap.add_argument("--max-days", type=int, default=165)
    ap.add_argument("--config", default="config.toml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    client = MexcClient(cfg.mexc)
    d = Path(a.dir)
    state = load_state(d)
    now = int(time.time())
    if state["start"] is None:
        state["start"] = now
    prices = {t["symbol"]: float(t["lastPrice"]) for t in client.tickers() if str(t.get("symbol", "")).endswith("_USDT")}
    settled = 0
    for tr in state["tranches"]:
        if tr["status"] == "open" and now >= tr["entry_ts"] + a.hold * DAY - 1800:
            settle(tr, prices)
            tr["exit_ts"] = now
            settled += tr["status"] == "settled"
    opened, why = 0, "past the registered horizon"
    n_by_sig = {s: sum(1 for t in state["tranches"] if t["signal"] == s) for s in SIGNALS}
    if now - state["start"] <= a.max_days * DAY and max(n_by_sig.values()) < MAX_TRANCHES:
        opened, why = open_tranches(client, state, now, a.top, a.top_m, datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d"))
    print(f"settled {settled} · opened {opened} ({why})")
    print(write_outputs(d, state))


if __name__ == "__main__":
    main()
