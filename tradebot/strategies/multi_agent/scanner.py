"""Agent 1 — market scanner: ranks every USDT contract by liquidity and movement and hands the top N to the analyst."""
from __future__ import annotations

import math

from .types import Candidate


class ScannerAgent:
    def __init__(self, cfg):
        self.c = cfg.agents.scanner

    def rank(self, tickers: list[dict]) -> list[Candidate]:
        out = []
        for t in tickers:
            sym = t.get("symbol", "")
            if not sym.endswith("_USDT"):
                continue
            turnover = float(t.get("amount24") or 0)
            if turnover < float(self.c.min_turnover_usdt):
                continue
            chg = float(t.get("riseFallRate") or 0) * 100.0
            score = float(self.c.w_move) * abs(chg) + float(self.c.w_liquidity) * math.log10(turnover)
            out.append(Candidate(sym, float(t["lastPrice"]), turnover, chg, score))
        out.sort(key=lambda c: (-c.score, c.symbol))
        return out[: int(self.c.top_n)]
