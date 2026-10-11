"""New report sections: transition prices, per-phase profit split, ETA accuracy, level-1/2 tables."""
import pytest

from conftest import FB, make_engine, ovr, step
from tradebot.strategies.pmo_zero_cross import model as M
from tradebot.strategies.pmo_zero_cross.report import (SLOPE_CLASSES, cell_tables, eta_accuracy,
                                                       key_sub, phase_distribution)


def run_ladder(cfg, s):
    fb = FB(s=s, n=30).cross_at(2).set("pmo1h", slice(None), -2).set("slope1h", slice(None), 1.0)   # ETA_1H = 2
    fb.price(slice(3, 4), 101).price(slice(4, 5), 103)
    fb.closed1h[4] = True
    fb.set("pmo1h", slice(4, None), 1)
    fb.price(slice(5, 7), 110).price(slice(7, None), 112)
    fb.closed4h[7] = True
    fb.set("pmo4h", slice(7, None), 1)
    fb.price(slice(8, None), 120)
    fb.lows = {9: 80}                                       # wick through every stop -> exit on bar 9
    eng, f = make_engine(cfg), fb.build()
    for k in range(10):
        step(eng, f, k)
    return eng


@pytest.mark.parametrize("s", [1, -1])
def test_transition_prices_and_phase_split(cfg, s):
    eng = run_ladder(cfg, s)
    tr = eng.closed[0]
    assert tr.exit_reason.startswith("STOP")
    assert tr.h1_cross_price == pytest.approx(100 + s * 3) and tr.h4_cross_price == pytest.approx(100 + s * 12)
    p = tr.phase_moves
    assert p[0] == pytest.approx(3.0) and p[1] == pytest.approx(9.0)          # 100->103, 103->112 (favorable %)
    assert sum(p) == pytest.approx(tr.gross_ret_pct)                           # phases add up to the whole move
    assert p[2] == pytest.approx(tr.gross_ret_pct - 12.0)
    assert tr.phase_mfe[0] == pytest.approx(3.0, abs=0.2)                      # best price in WAIT_1H ~103(+band)
    dist, tot = phase_distribution([tr])
    assert [round(d["share"], 6) for d in dist] == [round(x / tot, 6) for x in p]
    assert tr.deadline_log and tr.deadline_log[0][1] - tr.deadline_log[0][0] == 16


@pytest.mark.parametrize("s", [1, -1])
def test_phase_split_when_failing_in_first_phase(cfg, s):
    fb = FB(s=s, n=30).cross_at(2).price(slice(3, None), 104)
    fb.set("pmo15", slice(6, None), -1)                       # PMO15 back -> FAIL_1H
    eng, f = make_engine(cfg), fb.build()
    for k in range(8):
        step(eng, f, k)
    tr = eng.closed[0]
    assert tr.exit_reason == M.FAIL_1H
    assert tr.phase_moves[1] == 0 and tr.phase_moves[2] == 0 and tr.phase_moves[0] == pytest.approx(tr.gross_ret_pct)


@pytest.mark.parametrize("s", [1, -1])
def test_eta_accuracy(cfg, s):
    eng = run_ladder(cfg, s)
    tr = eng.closed[0]
    assert tr.snap["h1"]["eta"] == pytest.approx(2.0)          # |-2| / 1
    e = eta_accuracy([tr])
    assert e["n"] == 1 and e["mean_err"] == pytest.approx(tr.h1_cross_bars / 4 - 2.0)
    assert tr.h1_cross_bars == 2                                # crossed 2 bars (0.5h) after entry vs ETA 2h
    assert e["mean_err"] == pytest.approx(-1.5)


def test_level1_and_level2_tables(cfg):
    trades = []
    for s in (1, -1):
        trades += [t for t in run_ladder(cfg, s).closed]
    tabs = cell_tables(trades, cfg)
    assert list(tabs["main"]) == SLOPE_CLASSES
    assert tabs["main"]["with|with"]["n"] == 2
    assert sum(v["n"] for v in tabs["sub"].values()) == sum(v["n"] for v in tabs["main"].values()) == 2
    assert tabs["sub"][key_sub(trades[0])]["n"] == 2
    assert tabs["main"]["with|with"]["low_sample"] is True      # < 30 trades -> flagged
