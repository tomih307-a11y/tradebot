"""Section 4: every stage transition and every failure exit, long and short."""
import pytest

from conftest import FB, make_engine, ovr, step
from tradebot.strategies.pmo_zero_cross import model as M

SIDES = [1, -1]


def entered(cfg, fb, bars=3):
    eng = make_engine(cfg)
    f = fb.build()
    for k in range(bars):
        step(eng, f, k)
    return eng, f


@pytest.mark.parametrize("s", SIDES)
@pytest.mark.parametrize("p1,p4,stage", [(-1, -1, M.WAIT_1H), (1, -1, M.WAIT_4H), (1, 1, M.RIDE),
                                         (-1, 1, M.WAIT_1H)])
def test_initial_stage(cfg, s, p1, p4, stage):
    fb = FB(s=s).cross_at(2).set("pmo1h", slice(None), p1).set("pmo4h", slice(None), p4)
    eng, _ = entered(cfg, fb)
    tr = eng.open["X"]
    assert tr.side == ("long" if s == 1 else "short")
    assert tr.initial_stage == tr.stage == stage


@pytest.mark.parametrize("s", SIDES)
def test_wait1h_to_wait4h_on_hour_close(cfg, s):
    fb = FB(s=s).cross_at(2)
    fb.closed1h[5] = True
    fb.set("pmo1h", slice(5, None), 0.5)
    eng, f = entered(cfg, fb, 5)
    assert eng.open["X"].stage == M.WAIT_1H
    step(eng, f, 5)
    tr = eng.open["X"]
    assert tr.stage == M.WAIT_4H and tr.h1_cross_bars == 3 and tr.h4_cross_bars is None


@pytest.mark.parametrize("s", SIDES)
def test_wait1h_straight_to_ride_when_4h_already_across(cfg, s):
    fb = FB(s=s).cross_at(2).set("pmo4h", slice(None), 1)
    fb.closed1h[5] = True
    fb.set("pmo1h", slice(5, None), 0.5)
    eng, f = entered(cfg, fb, 6)
    tr = eng.open["X"]
    assert tr.stage == M.RIDE and tr.h1_cross_bars == 3 and tr.h4_cross_bars == 3


@pytest.mark.parametrize("s", SIDES)
def test_wait4h_to_ride_on_4h_close(cfg, s):
    fb = FB(s=s).cross_at(2).set("pmo1h", slice(None), 1)
    fb.closed4h[6] = True
    fb.set("pmo4h", slice(6, None), 0.1)
    eng, f = entered(cfg, fb, 6)
    assert eng.open["X"].stage == M.WAIT_4H
    step(eng, f, 6)
    assert eng.open["X"].stage == M.RIDE and eng.open["X"].max_stage == M.RIDE


# ───────────── WAIT_1H failures ─────────────
@pytest.mark.parametrize("s", SIDES)
def test_fail_1h_slope(cfg, s):
    fb = FB(s=s).cross_at(2)
    fb.closed1h[5] = True
    fb.set("slope1h", slice(5, None), -0.1)           # PMO_1H still below zero
    eng, f = entered(cfg, fb, 5)
    step(eng, f, 5)
    assert "X" not in eng.open
    tr = eng.closed[0]
    assert tr.exit_reason == M.FAIL_1H and tr.fail_detail == "1h_slope"


@pytest.mark.parametrize("s", SIDES)
def test_slope_negative_without_candle_close_does_not_fail(cfg, s):
    fb = FB(s=s).cross_at(2).set("slope1h", slice(None), -0.1)   # already negative at entry, no new 1H candle
    eng, f = entered(cfg, fb, 8)
    assert "X" in eng.open


@pytest.mark.parametrize("s", SIDES)
def test_fail_1h_pmo15_back(cfg, s):
    fb = FB(s=s).cross_at(2).set("pmo15", slice(5, None), -0.2)
    eng, f = entered(cfg, fb, 5)
    assert "X" in eng.open
    step(eng, f, 5)
    assert eng.closed[0].exit_reason == M.FAIL_1H and eng.closed[0].fail_detail == "pmo15_back"


@pytest.mark.parametrize("s", SIDES)
def test_fail_1h_timeout(cfg, s):
    fb = FB(s=s, n=60).cross_at(2)                    # ETA_1H=1 -> wait = max(16, 6) = 16 -> deadline bar 18
    eng, f = entered(cfg, fb, 19)                     # processed bars 0..18: still waiting
    assert "X" in eng.open and eng.open["X"].wait_deadline == 18
    step(eng, f, 19)                                  # bar 19 > deadline 18
    assert eng.closed[0].exit_reason == M.FAIL_1H and eng.closed[0].fail_detail == "timeout"


@pytest.mark.parametrize("s", SIDES)
def test_crossing_on_timeout_bar_counts_as_cross(cfg, s):
    fb = FB(s=s, n=60).cross_at(2)
    fb.closed1h[19] = True
    fb.set("pmo1h", slice(19, None), 0.5)
    eng, f = entered(cfg, fb, 20)
    assert "X" in eng.open and eng.open["X"].stage == M.WAIT_4H


@pytest.mark.parametrize("s", SIDES)
@pytest.mark.parametrize("flag,setup", [
    ("fail_1h_slope", lambda fb: (fb.closed1h.__setitem__(5, True), fb.set("slope1h", slice(5, None), -0.1))),
    ("fail_1h_pmo15_back", lambda fb: fb.set("pmo15", slice(5, None), -0.2)),
    ("fail_1h_timeout", lambda fb: None),
])
def test_exit_switches_1h(cfg, s, flag, setup):
    c = ovr(cfg, **{f"pmo_zero_cross.exits.{flag}": False})
    fb = FB(s=s, n=60).cross_at(2)
    setup(fb)
    eng = make_engine(c)
    f = fb.build()
    for k in range(40 if flag == "fail_1h_timeout" else 12):
        step(eng, f, k)
    assert "X" in eng.open


# ───────────── WAIT_4H failures ─────────────
@pytest.mark.parametrize("s", SIDES)
def test_fail_4h_slope(cfg, s):
    fb = FB(s=s).cross_at(2).set("pmo1h", slice(None), 1)
    fb.closed4h[6] = True
    fb.set("slope4h", slice(6, None), -0.1)
    eng, f = entered(cfg, fb, 6)
    step(eng, f, 6)
    tr = eng.closed[0]
    assert tr.exit_reason == M.FAIL_4H and tr.fail_detail == "4h_slope" and tr.max_stage == M.WAIT_4H


@pytest.mark.parametrize("s", SIDES)
def test_fail_4h_pmo1h_back(cfg, s):
    fb = FB(s=s).cross_at(2).set("pmo1h", slice(None), 1).set("pmo1h", slice(6, None), -0.3)
    fb.closed1h[6] = True
    eng, f = entered(cfg, fb, 6)
    step(eng, f, 6)
    assert eng.closed[0].exit_reason == M.FAIL_4H and eng.closed[0].fail_detail == "pmo1h_back"


@pytest.mark.parametrize("s", SIDES)
@pytest.mark.parametrize("flag", ["fail_4h_slope", "fail_4h_pmo1h_back"])
def test_exit_switches_4h(cfg, s, flag):
    c = ovr(cfg, **{f"pmo_zero_cross.exits.{flag}": False})
    fb = FB(s=s).cross_at(2).set("pmo1h", slice(None), 1)
    if flag == "fail_4h_slope":
        fb.closed4h[6] = True
        fb.set("slope4h", slice(6, None), -0.1)
    else:
        fb.set("pmo1h", slice(6, None), -0.3)
        fb.closed1h[6] = True
    eng = make_engine(c)
    f = fb.build()
    for k in range(10):
        step(eng, f, k)
    assert "X" in eng.open


@pytest.mark.parametrize("s", SIDES)
def test_wait4h_pmo15_reversal_does_not_exit(cfg, s):
    fb = FB(s=s).cross_at(2).set("pmo1h", slice(None), 1).set("pmo15", slice(5, None), -1)
    eng, f = entered(cfg, fb, 12)
    assert "X" in eng.open and eng.open["X"].stage == M.WAIT_4H


@pytest.mark.parametrize("s", SIDES)
def test_ride_has_no_indicator_exit(cfg, s):
    fb = (FB(s=s).cross_at(2).set("pmo1h", slice(None), 1).set("pmo4h", slice(None), 1)
          .set("pmo15", slice(5, None), -1).set("slope4h", slice(None), -1).set("slope1h", slice(None), -1))
    fb.closed1h[:] = True
    fb.closed4h[:] = True
    fb.set("pmo1h", slice(8, None), -1)
    eng, f = entered(cfg, fb, 20)
    assert "X" in eng.open and eng.open["X"].stage == M.RIDE


@pytest.mark.parametrize("s", SIDES)
def test_transitions_recorded_with_candle_counts(cfg, s):
    fb = FB(s=s, n=60).cross_at(2)
    fb.closed1h[6] = True
    fb.set("pmo1h", slice(6, None), 1)
    fb.closed4h[9] = True
    fb.set("pmo4h", slice(9, None), 1)
    eng, f = entered(cfg, fb, 10)
    assert [t[:2] for t in eng.open["X"].transitions] == [[M.WAIT_1H, 0], [M.WAIT_4H, 4], [M.RIDE, 7]]
    assert all(len(t) == 3 for t in eng.open["X"].transitions)          # each transition also records its price
    assert (eng.open["X"].h1_cross_bars, eng.open["X"].h4_cross_bars) == (4, 7)
