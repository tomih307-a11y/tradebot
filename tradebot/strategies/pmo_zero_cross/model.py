"""Trade record and small helpers (side math, ETA, entry snapshot)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

WAIT_1H, WAIT_4H, RIDE = "WAIT_1H", "WAIT_4H", "RIDE"
STAGE_RANK = {WAIT_1H: 0, WAIT_4H: 1, RIDE: 2}
ENTRY_PRIORITY = {RIDE: 0, WAIT_4H: 1, WAIT_1H: 2}          # section 8: RIDE first

STOP_INITIAL, STOP_TRAIL_1, STOP_TRAIL_2, STOP_TRAIL_3 = (
    "STOP_INITIAL", "STOP_TRAIL_1", "STOP_TRAIL_2", "STOP_TRAIL_3")
FAIL_1H, FAIL_4H, END_OF_DATA = "FAIL_1H", "FAIL_4H", "END_OF_DATA"
STOP_REASON_BY_STAGE = {0: STOP_INITIAL, 1: STOP_TRAIL_1, 2: STOP_TRAIL_2, 3: STOP_TRAIL_3}
EXIT_REASONS = [STOP_INITIAL, STOP_TRAIL_1, STOP_TRAIL_2, STOP_TRAIL_3, FAIL_1H, FAIL_4H, END_OF_DATA]

REASON_AR = {
    STOP_INITIAL: "وقف ابتدائي", STOP_TRAIL_1: "وقف متحرك مرحلة 1", STOP_TRAIL_2: "وقف متحرك مرحلة 2",
    STOP_TRAIL_3: "وقف متحرك مرحلة 3", FAIL_1H: "فشل الساعة", FAIL_4H: "فشل الـ4 ساعات",
    END_OF_DATA: "نهاية البيانات",
}
STAGE_AR = {WAIT_1H: "انتظار الساعة", WAIT_4H: "انتظار الـ4 ساعات", RIDE: "ترند كامل"}


def side_sign(side: str) -> int:
    return 1 if side == "long" else -1


def eta_candles(pmo: float, slope: float, s: int) -> float:
    """|PMO|/|SLOPE| in candles of that frame; only if the frame is on the opposite side
    of zero AND its slope points toward zero; otherwise infinity."""
    if math.isnan(pmo) or math.isnan(slope):
        return math.inf
    if s * pmo < 0 and s * slope > 0:
        return abs(pmo) / abs(slope)
    return math.inf


def frame_state(pmo: float, signal: float, slope: float, s: int) -> dict:
    return {
        "pmo": pmo, "signal": signal, "slope": slope,
        "pos": "above" if s * pmo >= 0 else "below",             # relative to trade direction
        "slope_dir": "with" if s * slope >= 0 else "against",
        "pmo_vs_signal": "above" if pmo > signal else "below",   # raw (not side-adjusted)
        "eta": eta_candles(pmo, slope, s),
    }


def slope_class(h1: dict, h4: dict) -> str:
    """Level-1 initial-state class: slope is the primary classifier (1H slope | 4H slope)."""
    return f"{h1['slope_dir']}|{h4['slope_dir']}"


def n_slopes_with(h1: dict, h4: dict) -> int:
    return (h1["slope_dir"] == "with") + (h4["slope_dir"] == "with")


@dataclass
class Trade:
    id: int
    symbol: str
    side: str
    entry_ts: int                    # open time of the 15m cross bar; execution at its close
    entry_price: float
    initial_stage: str
    stage: str
    max_stage: str
    stop: float
    initial_stop: float
    stop_stage: int = 0              # trailing stage of the rule that last raised the stop
    trail_stage: int = 0
    extreme_close: float = 0.0       # highest (long) / lowest (short) close since entry
    best_price: float = 0.0          # best high/low seen after entry (for MFE)
    worst_price: float = 0.0         # worst low/high seen after entry (for MAE)
    h1_cross_bars: int | None = None # 15m bars from entry to the 1H crossing (0 = already across)
    h4_cross_bars: int | None = None
    transitions: list = field(default_factory=list)   # [[stage, bars_from_entry, price], ...]
    h1_cross_price: float | None = None
    h4_cross_price: float | None = None
    wait_deadline: int | None = None                  # absolute 15m bar number (ts // 900)
    deadline_log: list = field(default_factory=list)  # [[bar_no, deadline], ...] every (re)computation
    phase_best: list = field(default_factory=lambda: [None, None, None])  # best favorable price seen per phase
    phase_moves: list | None = None                   # favorable move (% of entry) per phase on exit
    phase_mfe: list | None = None
    n_stop_moves: int = 0
    snap: dict = field(default_factory=dict)
    stop_order_id: int | None = None
    position_id: int | None = None
    vol: int | None = None
    # filled on exit
    exit_ts: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    fail_detail: str | None = None
    bars_held: int | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None
    gross_ret_pct: float | None = None
    mfe_retained: float | None = None
    fees_pct: float | None = None
    funding_pct: float | None = None
    net_ret_pct: float | None = None
    pnl_usdt: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        return cls(**d)

    @property
    def s(self) -> int:
        return side_sign(self.side)
