"""Volume profile around each channel signal entry (data cut at the signal). Profile = candle volume spread evenly over its high-low range, 100 price bins, windows 7d and 30d (1H candles).
Reports POC, value area (70%) low/high, where the entry sits (percent of range, inside VA?, distance to POC), nearest high-volume node above/below, and whether the stop / TPs sit near nodes."""
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import scanner as S

U = lambda y, mo, d, h: int((datetime(y, mo, d, h, tzinfo=timezone.utc) - timedelta(hours=4)).timestamp())
SIG = [("AT_USDT", U(2026, 11, 25, 21), 0.0544, 0.0517, [0.056, 0.0579, 0.0597, 0.0616, 0.0634, 0.0653]),
       ("MOODENG_USDT", U(2026, 11, 25, 21), 0.02792, 0.02652, [0.02876, 0.02971, 0.03066, 0.03161, 0.03255, 0.0335]),
       ("EVAA_USDT", U(2026, 11, 25, 22), 2.785, 2.646, [2.869, 2.963, 3.058, 3.153, 3.247, 3.342]),
       ("ELSA_USDT", U(2026, 11, 26, 13), 0.09577, 0.09098, [0.09864, 0.1019, 0.10516, 0.10841, 0.11167, 0.11492])]
now = int(time.time())


def kl(sym, a, b):
    out = {k: [] for k in ("time", "high", "low", "vol")}
    while a < b:
        e = min(a + 3600 * 1000, b)
        d = None
        for r in range(4):
            try:
                d = S.get(f"https://contract.mexc.com/api/v1/contract/kline/{sym}?interval=Min60&start={a}&end={e}")["data"]; break
            except Exception:
                time.sleep(1.5 + r)
        if d is None:
            raise SystemExit("fetch failed")
        for k in out:
            out[k] += d[k]
        a = e
    return {k: np.array(v, float) for k, v in out.items()}


def profile(h, l, v, nb=100):
    lo, hi = l.min(), h.max()
    edges = np.linspace(lo, hi, nb + 1)
    vol = np.zeros(nb)
    for hh, ll, vv in zip(h, l, v):
        i0 = min(nb - 1, int((ll - lo) / (hi - lo) * nb)); i1 = min(nb - 1, int((hh - lo) / (hi - lo) * nb))
        vol[i0:i1 + 1] += vv / (i1 - i0 + 1)
    return edges, vol


for sym, t0, entry, stop, tps in SIG:
    m = kl(sym, t0 - 31 * 86400, t0)
    print(f"== {sym[:-5]}")
    for days in (7, 30):
        k = m["time"] >= t0 - days * 86400
        h, l, v = m["high"][k], m["low"][k], m["vol"][k]
        ed, vol = profile(h, l, v)
        mid = (ed[:-1] + ed[1:]) / 2
        poc = mid[vol.argmax()]
        order = np.argsort(-vol); cum = np.cumsum(vol[order]) / vol.sum(); va = order[:np.searchsorted(cum, 0.7) + 1]
        val, vah = mid[va].min(), mid[va].max()
        ei = min(len(vol) - 1, max(0, np.searchsorted(ed, entry) - 1))
        rel = vol[ei] / vol.mean()
        thr = np.percentile(vol, 80)
        hvn_up = [x for x in mid if x > entry and vol[min(len(vol) - 1, np.searchsorted(ed, x) - 1)] >= thr]
        hvn_dn = [x for x in mid if x < entry and vol[min(len(vol) - 1, np.searchsorted(ed, x) - 1)] >= thr]
        pos = (entry - ed[0]) / (ed[-1] - ed[0]) * 100
        print(f"  {days}ي: نطاق {ed[0]:.5g}–{ed[-1]:.5g} · موقع الدخول {pos:.0f}% · POC {poc:.5g} ({(entry/poc-1)*100:+.1f}% من الدخول) · منطقة القيمة {val:.5g}–{vah:.5g} ({'داخل' if val <= entry <= vah else 'خارج'}) · حجم عند الدخول ×{rel:.2f} من المتوسط")
        print(f"       أقرب عقدة حجم عالٍ فوق الدخول: {(min(hvn_up)/entry-1)*100:+.1f}%" if hvn_up else "       لا عقدة حجم عالٍ فوق الدخول", end=" · ")
        print(f"تحت الدخول: {(max(hvn_dn)/entry-1)*100:+.1f}%" if hvn_dn else "لا عقدة تحت")
        sl = vol[min(len(vol) - 1, max(0, np.searchsorted(ed, stop) - 1))] / vol.mean()
        print(f"       حجم عند الوقف ×{sl:.2f} · عند TP1 ×{vol[min(len(vol)-1, max(0, np.searchsorted(ed, tps[0]) - 1))]/vol.mean():.2f} · TP3 ×{vol[min(len(vol)-1, max(0, np.searchsorted(ed, tps[2]) - 1))]/vol.mean():.2f}")
