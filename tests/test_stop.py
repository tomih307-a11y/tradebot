"""Section 5: trailing stop stages, never-backwards rule, only closed bars."""
import numpy as np
import pytest

from conftest import FB, fav, make_engine, ovr, step
from tradebot.strategies.pmo_zero_cross import model as M

SIDES = [1, -1]


def ladder(s):
    """entry bar 2 @100; 1H crosses bar 4 (px 103); 4H crosses bar 7; 1H slope turns down bar 9."""
    fb = FB(s=s, n=30).cross_at(2)
    fb.price(slice(3, 4), 101).price(slice(4, 5), 103).price(slice(5, None), 110)
    fb.closed1h[4] = True
    fb.set("pmo1h", slice(4, None), 1)
    fb.closed4h[7] = True
    fb.set("pmo4h", slice(7, None), 1)
    fb.price(slice(8, None), 120)
    fb.closed1h[9] = True
    fb.set("slope1h", slice(9, None), -0.1)
    fb.price(slice(9, None), 121).price(slice(10, None), 125)
    return fb


@pytest.mark.parametrize("s", SIDES)
def test_stage_ladder(cfg, s):
    fb = ladder(s)
    eng, f = make_engine(cfg), fb.build()
    tr = None
    def stop(): return eng.open["X"].stop
    for k in range(3):
        step(eng, f, k)
    tr = eng.open["X"]
    assert tr.stop == pytest.approx(fav(s, 100 * (1 - 0.03)))             # stage 0: project stop
    assert tr.entry_price == pytest.approx(100)
    step(eng, f, 3)
    assert stop() == pytest.approx(fav(s, 97))                            # still WAIT_1H: untouched
    step(eng, f, 4)                                                       # 1H crossed @103
    assert tr.stage == M.WAIT_4H and tr.stop_stage == 1
    assert stop() == pytest.approx(fav(s, 100.08))                        # break-even + 2 fees beats 103-3
    step(eng, f, 5)                                                       # extreme 110
    assert stop() == pytest.approx(fav(s, 107)) and tr.stop_stage == 1
    fb.a["atr15"][6:] = 3.0                                                # bigger ATR would pull the stop back
    step(eng, f, 6)
    assert stop() == pytest.approx(fav(s, 107))                           # never retreats
    step(eng, f, 7)                                                       # 4H crossed -> RIDE, stage 2
    assert tr.stage == M.RIDE and stop() == pytest.approx(fav(s, 107))    # 110-2*2=106 < 107
    step(eng, f, 8)                                                       # extreme 120
    assert stop() == pytest.approx(fav(s, 116)) and tr.stop_stage == 2
    step(eng, f, 9)                                                       # 1H slope < 0 -> stage 3 (ATR15=3 here)
    assert tr.trail_stage == 3 and stop() == pytest.approx(fav(s, 121 - 3 * 1.5)) and tr.stop_stage == 3
    fb.a["atr15"][:] = 1.0
    step(eng, f, 10)                                                      # stays tightened, no stage-2 formula
    assert stop() == pytest.approx(fav(s, 125 - 1.5)) and tr.trail_stage == 3


@pytest.mark.parametrize("s", SIDES)
def test_stop_monotonic_over_random_path(cfg, s):
    rng = np.random.default_rng(7)
    n = 400
    fb = FB(s=s, n=n).cross_at(2)
    fb.set("pmo1h", slice(None), 1).set("pmo4h", slice(None), 1)
    fb.price(slice(None), 100 + np.cumsum(rng.normal(0.05, 0.4, n)))
    fb.a["atr15"][:] = rng.uniform(0.2, 2.0, n)
    fb.a["atr1h"][:] = rng.uniform(0.5, 4.0, n)
    fb.closed1h[::4] = True
    fb.set("slope1h", slice(None), 1)
    fb.a["slope1h"][:] = s * rng.normal(0, 1, n)
    fb.band = 0.0
    eng, f = make_engine(cfg), fb.build()
    prev, seen = None, 0
    for k in range(n):
        step(eng, f, k)
        if "X" in eng.open:
            cur = eng.open["X"].stop
            if prev is not None:
                assert s * (cur - prev) >= -1e-12, "stop moved against the trade"
            prev, seen = cur, seen + 1
        else:
            prev = None
            fb2 = None
    assert seen > 10


@pytest.mark.parametrize("s", SIDES)
def test_stop_uses_previous_bars_stop_not_current(cfg, s):
    """Bar 6 dips to 104 (inside the bar). The stop raised by bar 6's own close must not apply to bar 6."""
    fb = FB(s=s, n=30).cross_at(2).set("pmo1h", slice(None), 1)
    fb.price(slice(3, None), 110)
    fb.lows = {6: 104}                       # wick under what the NEW stop (107) will be
    fb.a["atr15"][:] = 1.0
    eng, f = make_engine(cfg), fb.build()
    for k in range(6):
        step(eng, f, k)
    assert eng.open["X"].stop == pytest.approx(fav(s, 107))   # from bar 5's close (direct WAIT_4H -> trail 1)
    step(eng, f, 6)                                           # low 104 <= 107 -> hit with bar-5 stop
    assert eng.closed[0].exit_price == pytest.approx(fav(s, 107))
    assert eng.closed[0].exit_reason == M.STOP_TRAIL_1


@pytest.mark.parametrize("s", SIDES)
def test_wick_before_stop_update_does_not_use_unclosed_value(cfg, s):
    """A wick that would hit a stop computed from the *current* (not yet closed) bar must not exit."""
    fb = FB(s=s, n=30).cross_at(2).set("pmo1h", slice(None), 1)
    fb.price(slice(3, 5), 100).price(slice(5, None), 120)     # bar 5 closes at 120 -> new stop 117
    fb.lows = {5: 110}                                        # bar 5 wick 110 < 117 but > old stop 97
    eng, f = make_engine(cfg), fb.build()
    for k in range(6):
        step(eng, f, k)
    assert "X" in eng.open and eng.open["X"].stop == pytest.approx(fav(s, 117))


@pytest.mark.parametrize("s", SIDES)
def test_gap_fills_at_open(cfg, s):
    fb = FB(s=s, n=30).cross_at(2)
    fb.price(slice(4, None), 90)
    fb.opens = {4: 92}                                          # opens beyond the 97 stop (gap)
    eng, f = make_engine(cfg), fb.build()
    for k in range(5):
        step(eng, f, k)
    t = eng.closed[0]
    assert t.exit_reason == M.STOP_INITIAL and t.exit_price == pytest.approx(fav(s, 92))     # filled at the open, not at 97
