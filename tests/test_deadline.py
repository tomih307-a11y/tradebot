"""Section 4 dynamic WAIT_DEADLINE (+ mandatory test #4) and the section 8 slope-first priority."""
import math

import pytest

from conftest import FB, make_engine, ovr, step
from tradebot.strategies.pmo_zero_cross import model as M


def deadline_of(cfg, pmo1h, slope1h, s=1):
    """Deadline (bars after entry) computed at entry for the given 1H state (favorable units)."""
    fb = FB(s=s, n=6).cross_at(2).set("pmo1h", slice(None), pmo1h).set("slope1h", slice(None), slope1h)
    eng, f = make_engine(cfg), fb.build()
    for k in range(3):
        step(eng, f, k)
    return eng.open["X"].wait_deadline - eng.open["X"].entry_ts // 900


@pytest.mark.parametrize("s", [1, -1])
def test_deadline_formula(cfg, s):
    assert deadline_of(cfg, -1.0, 1.0, s) == 16                 # ETA 1 -> ceil(6)=6 < MIN 16
    assert deadline_of(cfg, -3.0, 1.0, s) == 18                 # ETA 3 -> ceil(18)
    assert deadline_of(cfg, -3.01, 1.0, s) == math.ceil(3.01 * 6)
    assert deadline_of(cfg, -1000.0, 1.0, s) == 96              # capped at 24h
    assert deadline_of(cfg, -5.0, -1.0, s) == 16                # slope away from zero -> ETA inf -> MIN
    c = ovr(cfg, **{"pmo_zero_cross.exits.wait_eta_mult": 2.0})
    assert deadline_of(c, -3.0, 1.0, s) == 24                   # ceil(3*4*2)


def test_deadline_recomputed_on_each_closed_hour_candle(cfg):
    fb = FB(s=1, n=40).cross_at(2).set("pmo1h", slice(None), -10).set("slope1h", slice(None), 1)
    fb.closed1h[7] = True
    fb.set("pmo1h", slice(7, None), -4)                          # 1H candle closes at bar 7: ETA now 4 -> 24 bars
    eng, f = make_engine(cfg), fb.build()
    for k in range(9):
        step(eng, f, k)
    tr = eng.open["X"]
    assert tr.deadline_log == [[2, 2 + 60], [7, 7 + 24]]         # entry: ceil(10*6)=60 ; recompute at bar 7
    assert tr.wait_deadline == 31


@pytest.mark.parametrize("s", [1, -1])
def test_short_with_pmo1h_17_slope_minus_1_2_is_not_timed_out_before_14h(cfg, s):
    """Mandatory test #4. For s=-1 this is the spec's case: PMO_1H=+17, slope -1.2 per hour (real units).
    The old fixed 4h deadline would have exited at bar 19; the ETA-based one must wait for the crossing."""
    n = 80
    fb = FB(s=s, n=n).cross_at(2)
    fb.set("pmo1h", slice(None), -17.0).set("slope1h", slice(None), 1.2)      # favorable units
    j = 0
    for k in range(3, n, 4):                                                   # a 1H candle closes every 4 bars
        j += 1
        fb.closed1h[k] = True
        fb.set("pmo1h", slice(k, None), -(17.0 - 1.2 * j))
    eng, f = make_engine(cfg), fb.build()
    cross_bar = None
    for k in range(n):
        step(eng, f, k)
        if "X" in eng.open and eng.open["X"].stage != M.WAIT_1H:
            cross_bar = k
            break
        assert not eng.closed, f"exited at bar {k}: {eng.closed[0].exit_reason}/{eng.closed[0].fail_detail}"
    assert cross_bar is not None
    hours_waited = (cross_bar - 2) / 4
    assert hours_waited >= 14, hours_waited                                   # 17/1.2 = 14.17 h until the cross
    tr = eng.open["X"]
    assert tr.stage == M.WAIT_4H and tr.h1_cross_bars == cross_bar - 2
    assert tr.deadline_log[0] == [2, 2 + 85]                                  # ceil(14.1667 * 4 * 1.5) = 85


@pytest.mark.parametrize("s", [1, -1])
def test_slope_reversal_still_exits_before_deadline(cfg, s):
    fb = FB(s=s, n=60).cross_at(2).set("pmo1h", slice(None), -17.0).set("slope1h", slice(None), 1.2)
    fb.closed1h[11] = True
    fb.set("slope1h", slice(11, None), -0.5)
    eng, f = make_engine(cfg), fb.build()
    for k in range(12):
        step(eng, f, k)
    assert eng.closed[0].exit_reason == M.FAIL_1H and eng.closed[0].fail_detail == "1h_slope"


@pytest.mark.parametrize("s", [1, -1])
def test_timeout_switch_off_means_slope_only(cfg, s):
    c = ovr(cfg, **{"pmo_zero_cross.exits.fail_1h_timeout": False})
    fb = FB(s=s, n=200).cross_at(2)
    eng, f = make_engine(c), fb.build()
    for k in range(200):
        step(eng, f, k)
    assert "X" in eng.open


# ───────────── section 8: slope-first priority ─────────────
@pytest.mark.parametrize("s", [1, -1])
def test_priority_slopes_with_beat_stage(cfg, s):
    c = ovr(cfg, **{"trading.max_open_trades": 1})

    def mk(name, p1, p4, sl1, sl4):
        fb = FB(name, s=s, n=8).cross_at(3)
        fb.set("pmo1h", slice(None), p1).set("pmo4h", slice(None), p4)
        fb.set("slope1h", slice(None), sl1).set("slope4h", slice(None), sl4)
        return fb.build()
    feats = {
        "RIDE_BOTH_AGAINST": mk("RIDE_BOTH_AGAINST", 1, 1, -1, -1),     # RIDE but 0 slopes with
        "WAIT1H_BOTH_WITH": mk("WAIT1H_BOTH_WITH", -1, -1, 1, 1),       # WAIT_1H but 2 slopes with
        "WAIT4H_ONE_WITH": mk("WAIT4H_ONE_WITH", 1, -1, 1, -1),         # 1 slope with
    }
    eng = make_engine(c)
    for k in range(4):
        step(eng, feats, k)
    assert list(eng.open) == ["WAIT1H_BOTH_WITH"]
    assert {r["symbol"] for r in eng.rejected} == {"RIDE_BOTH_AGAINST", "WAIT4H_ONE_WITH"}
    assert eng.open["WAIT1H_BOTH_WITH"].snap["n_slopes_with"] == 2
    assert eng.open["WAIT1H_BOTH_WITH"].snap["slope_class"] == "with|with"

    c2 = ovr(cfg, **{"trading.max_open_trades": 2})                      # then 1 slope-with, then stage
    eng2 = make_engine(c2)
    for k in range(4):
        step(eng2, feats, k)
    assert set(eng2.open) == {"WAIT1H_BOTH_WITH", "WAIT4H_ONE_WITH"}
