"""Market-data recorder for the future explosion hunter (public MEXC data only, no keys).
One snapshot = every USDT contract's ticker (price, 24h turnover, OPEN INTEREST, funding, bid/ask) + order-book depth (USDT within 1% / 2% of mid)
for the liquid contracts. These fields have NO history on MEXC, so they can only be learned by recording them from now on.
Usage:  python -m tradebot.strategies.mtf_entry.recorder snapshot OUT_DIR      |      ... consolidate DIR"""
from __future__ import annotations

import csv
import gzip
import io
import os
import sys
import time
from datetime import datetime, timezone

FIELDS = ["ts", "symbol", "last", "fair", "index", "bid1", "ask1", "vol24", "amt24", "oi", "funding", "chg24", "hi24", "lo24",
          "bid_d1", "bid_d2", "ask_d1", "ask_d2", "d_cov", "csize"]
DEPTH_MIN_TURNOVER = 300_000          # USDT 24h; depth only for contracts liquid enough to trade
DEPTH_LIMIT, DEPTH_LIMIT_FALLBACK = 500, 100


def side_depth(levels, mid: float, csize: float, pct: float, bids: bool) -> float:
    """USDT value of resting orders within pct of mid. levels = [[price, contracts, order_count], ...] (MEXC: quantity is index 1)."""
    lim = mid * (1 - pct) if bids else mid * (1 + pct)
    tot = 0.0
    for lv in levels or []:
        p, v = float(lv[0]), float(lv[1])
        if (bids and p >= lim) or ((not bids) and p <= lim):
            tot += p * v * csize
    return tot


def coverage(bids, asks, mid: float) -> float:
    """How far from mid (as a fraction) the thinner side of the fetched book reaches: if it is < 0.02 the 2% depth is truncated."""
    def far(levels):
        return max((abs(float(lv[0]) - mid) / mid for lv in levels or []), default=0.0)
    return min(far(bids), far(asks))


def snapshot(client, now: int | None = None) -> list[dict]:
    now = int(time.time()) if now is None else now
    detail = client._request("GET", "/api/v1/contract/detail") or []
    detail = detail if isinstance(detail, list) else [detail]
    csize = {c["symbol"]: float(c.get("contractSize") or 1.0) for c in detail}
    rows = []
    for t in client.tickers():
        s = str(t.get("symbol", ""))
        if not s.endswith("_USDT"):
            continue
        g = lambda k: t.get(k)
        rows.append({"ts": now, "symbol": s, "last": g("lastPrice"), "fair": g("fairPrice"), "index": g("indexPrice"), "bid1": g("bid1"), "ask1": g("ask1"),
                     "vol24": g("volume24"), "amt24": g("amount24"), "oi": g("holdVol"), "funding": g("fundingRate"), "chg24": g("riseFallRate"),
                     "hi24": g("high24Price"), "lo24": g("lower24Price"), "bid_d1": "", "bid_d2": "", "ask_d1": "", "ask_d2": "", "d_cov": "", "csize": csize.get(s, "")})
    for r in rows:
        try:
            if float(r["amt24"] or 0) < DEPTH_MIN_TURNOVER or not r["csize"]:
                continue
            try:
                d = client._request("GET", f"/api/v1/contract/depth/{r['symbol']}", {"limit": DEPTH_LIMIT}) or {}
            except Exception:
                d = client._request("GET", f"/api/v1/contract/depth/{r['symbol']}", {"limit": DEPTH_LIMIT_FALLBACK}) or {}
            mid = float(r["last"])
            cs = float(r["csize"])
            r["bid_d1"], r["bid_d2"] = (round(side_depth(d.get("bids"), mid, cs, p, True), 1) for p in (0.01, 0.02))
            r["ask_d1"], r["ask_d2"] = (round(side_depth(d.get("asks"), mid, cs, p, False), 1) for p in (0.01, 0.02))
            r["d_cov"] = round(coverage(d.get("bids"), d.get("asks"), mid), 5)
        except Exception as e:                                # one failing contract must not lose the whole snapshot
            print("depth skip", r["symbol"], e, file=sys.stderr)
    return rows


def write_snapshot(rows: list[dict], out_dir: str, now: int | None = None) -> str:
    now = int(time.time()) if now is None else now
    os.makedirs(out_dir, exist_ok=True)
    name = datetime.fromtimestamp(now, timezone.utc).strftime("snap_%Y%m%d_%H%M.csv.gz")
    path = os.path.join(out_dir, name)
    buf = io.StringIO()
    w = csv.DictWriter(buf, FIELDS)
    w.writeheader()
    w.writerows(rows)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return path


def consolidate(data_dir: str, today: str | None = None) -> list[str]:
    """Merge the small per-run files of every PAST UTC day into one day_YYYYMMDD.csv.gz (and delete the small ones)."""
    today = today or datetime.now(timezone.utc).strftime("%Y%m%d")
    by_day: dict[str, list[str]] = {}
    for n in sorted(os.listdir(data_dir)):
        if n.startswith("snap_") and n.endswith(".csv.gz"):
            day = n[5:13]
            if day < today:
                by_day.setdefault(day, []).append(n)
    made = []
    for day, names in by_day.items():
        out = os.path.join(data_dir, f"day_{day}.csv.gz")
        with gzip.open(out, "wt", encoding="utf-8") as fo:
            for i, n in enumerate(names):
                with gzip.open(os.path.join(data_dir, n), "rt", encoding="utf-8") as fi:
                    lines = fi.readlines()
                fo.writelines(lines if i == 0 else lines[1:])
        for n in names:
            os.remove(os.path.join(data_dir, n))
        made.append(out)
    return made


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "snapshot":
        from tradebot.config import load_config
        from tradebot.mexc.client import MexcClient
        rows = snapshot(MexcClient(load_config("config.toml").mexc))
        p = write_snapshot(rows, sys.argv[2])
        have_depth = sum(1 for r in rows if r["bid_d1"] != "")
        print(f"{len(rows)} عقداً · عمق لـ{have_depth} · {p}")
    elif len(sys.argv) == 3 and sys.argv[1] == "consolidate":
        print(consolidate(sys.argv[2]))
