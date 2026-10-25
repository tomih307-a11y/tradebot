"""Messages passed between the agents (each agent reads one, writes one)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:                 # Scanner -> Technical
    symbol: str
    price: float
    turnover: float
    change_pct: float
    score: float


@dataclass
class Recommendation:            # Technical -> News -> Entry
    symbol: str
    side: str                    # "long" | "short"
    score: float                 # 0..1 (share of technical conditions met) + news bias
    price: float
    atr: float
    zone: tuple[float, float]    # preferred entry zone (lo, hi)
    stop: float
    tps: list[float]
    reasons: list[str] = field(default_factory=list)
    created_ts: int = 0
    news_bias: float = 0.0
    signal: object | None = None   # the channel Signal this recommendation came from (None = own analysis)


@dataclass
class NewsVerdict:               # News -> orchestrator
    bias: float                  # -1 (bearish) .. +1 (bullish)
    headlines: list[str] = field(default_factory=list)


@dataclass
class EntryPlan:                 # Entry -> orchestrator
    action: str                  # "market" | "wait" | "skip"
    reason: str


@dataclass
class ExitAction:                # Exit -> orchestrator
    new_stop: float | None = None
    close: bool = False
    reason: str = ""
