"""Independent audit of the study code: (A) synthetic random-walk baseline, (B) look-ahead check by truncation, (C) pure-python re-implementation cross-check.
Usage: python tools/audit_tests.py   (needs MEXC public data for B and C)"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tradebot.strategies.mtf_entry import channel_logic as CL  # noqa: E402
from tradebot.strategies.mtf_entry import quality_scan as QS  # noqa: E402
from tradebot.strategies.mtf_entry import immediate_move as IM  # noqa: E402


def audit_a():
    rng = np.random.default_rng(5)
    out = []
    for drift in (0.0, -0.0005):
        res = {}
        for cost in (0.0, QS.COST):
            QS.COST = cost
            rs = []
            for _ in range(60):
                n = 3000
                ret = rng.normal(drift, 0.012, n)
                c = 100 * np.exp(np.cumsum(ret)); o = np.concatenate(([100.0], c[:-1]))
                h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
                atr = CL.atr14(h, l, c)
                for i in range(100, n - 60, 7):
                    r = QS.trade(o, h, l, c, i, atr[i])
                    if r is not None:
                        rs.append(r)
            res[cost] = np.mean(rs)
        QS.COST = 0.002
        out.append(f"A. سير عشوائي بانحراف {drift:+.4f}/ساعة: متوسط R بلا تكلفة {res[0.0]:+.3f} · مع التكلفة {res[0.002]:+.3f}")
    return out


def audit_bc():
    from tradebot.config import load_config
    from tradebot.data.klines import drop_unclosed, fetch_range
    from tradebot.mexc.client import MexcClient
    from tradebot.strategies.mtf_entry.early_signs import features
    from tradebot.strategies.mtf_entry.ob_entry import hourly

    cl, now = MexcClient(load_config("config.toml").mexc), int(time.time())
    out, bad_comp, bad_feat, n_comp, n_feat, bad_tr, n_tr = [], 0, 0, 0, 0, 0, 0
    rng = np.random.default_rng(3)
    for sym in ("BTC_USDT", "ETH_USDT", "SOL_USDT", "DOGE_USDT", "TAO_USDT", "XRP_USDT"):
        k = drop_unclosed(fetch_range(cl, sym, "Min15", int(now // 900 * 900) - 40 * 96 * 900, now), "Min15", now)
        t, o15, h15, l15, c15, v15 = (np.asarray(k[x], float) for x in ("time", "open", "high", "low", "close", "vol"))
        th, o, h, l, c = hourly(t, o15, h15, l15, c15)
        comp, ob_lo, atr = QS.components(th, o, h, l, c)
        for i in rng.integers(300, len(c) - 60, 60):                       # (B1) components on truncated data == on full data
            cm, ol, at = QS.components(th[:i + 1], o[:i + 1], h[:i + 1], l[:i + 1], c[:i + 1])
            n_comp += 1
            if not (np.array_equal(cm[i], comp[i]) and (np.isnan(ol[i]) and np.isnan(ob_lo[i]) or abs(ol[i] - ob_lo[i]) < 1e-12)):
                bad_comp += 1
        n = len(c15)
        z = np.zeros(n)
        f, fmax, valid = features(o15, h15, l15, c15, v15, t, np.array([0.0]), np.array([0.0]), z)
        for i in rng.integers(800, n - 20, 25):                            # (B2) hunter flags on truncated data == on full data
            m = i + 1
            f2, _, _ = features(o15[:m], h15[:m], l15[:m], c15[:m], v15[:m], t[:m], np.array([0.0]), np.array([0.0]), np.zeros(m))
            n_feat += 1
            if any(bool(f[x][i]) != bool(f2[x][i]) for x in ("F1", "F2", "F3")):
                bad_feat += 1
        for i in rng.integers(120, len(c) - 60, 40):                       # (C) pure-python re-implementation of trade()
            dist = atr[i]
            if not np.isfinite(dist) or dist <= 0:
                continue
            e, hh_, stop, px = c[i], c[i], c[i] - dist, None
            for j in range(i + 1, i + 1 + QS.HOLD):
                if l[j] <= stop:
                    px = min(stop, o[j]); break
                hh_ = max(hh_, h[j]); stop = max(stop, hh_ - dist)
            if px is None:
                px = c[i + QS.HOLD]
            ref = (px - e) / dist - QS.COST * e / dist
            got = QS.trade(o, h, l, c, i, dist)
            n_tr += 1
            if got is None or abs(got - ref) > 1e-9:
                bad_tr += 1
    out.append(f"B1. نظر للمستقبل في المكونات (SMC/المتوسطات/RSI/MACD): {bad_comp} اختلافاً من {n_comp} فحصاً")
    out.append(f"B2. نظر للمستقبل في علامات الصائد (F1-F3): {bad_feat} اختلافاً من {n_feat} فحصاً")
    out.append(f"C.  إعادة تنفيذ مستقلة لحساب الصفقة: {bad_tr} اختلافاً من {n_tr} صفقة")
    return out


if __name__ == "__main__":
    lines = ["تدقيق مستقل لكود الاختبارات", ""] + audit_a() + audit_bc()
    text = "\n".join(lines)
    print(text)
    Path("reports/mtf_scan").mkdir(parents=True, exist_ok=True)
    open("reports/mtf_scan/audit.txt", "w", encoding="utf-8").write(text)
