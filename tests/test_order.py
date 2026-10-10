"""Section 7 processing order + sections 2 and 8 entry rules."""
import pytest

from conftest import FB, fav, make_engine, ovr, step
from tradebot.strategies.pmo_zero_cross import model as M

SIDES = [1, -1]


@pytest.mark.parametrize("s", SIDES)
def test_trace_order_when_1h_and_4h_close_together(cfg, s):
    fb = FB(s=s).cross_at(2)
    fb.closed1h[6] = fb.closed4h[6] = True
    fb.set("pmo1h", slice(6, None), 1).set("pmo4h", slice(6, None), 1)
    eng, f = make_engine(cfg, trace=True), fb.build()
    for k in range(8):
        step(eng, f, k)
    at6 = [(st) for ts, st, sym in eng.trace if ts == 6 * 900]
    assert at6 == ["A", "B", "C", "D", "E", "F"]
    assert eng.open["X"].stage == M.RIDE and eng.open["X"].h1_cross_bars == eng.open["X"].h4_cross_bars == 4


@pytest.mark.parametrize("s", SIDES)
def test_stop_check_precedes_update_on_double_close_bar(cfg, s):
    """Bar 6: 1H+4H close, price 110 but the wick dips to 98 (> old stop 97, < the new stop 106)."""
    fb = FB(s=s, n=30).cross_at(2)
    fb.price(slice(3, 6), 100).price(slice(6, None), 110)
    fb.closed1h[6] = fb.closed4h[6] = True
    fb.set("pmo1h", slice(6, None), 1).set("pmo4h", slice(6, None), 1)
    fb.lows = {6: 98}
    eng, f = make_engine(cfg), fb.build()
    for k in range(7):
        step(eng, f, k)
    tr = eng.open["X"]                                    # A used the old stop -> survived bar 6
    assert tr.stage == M.RIDE and tr.stop == pytest.approx(fav(s, 110 - 2 * 2))
    step(eng, f, 7)                                       # next bar low (109.9) > 106: alive
    assert "X" in eng.open
    fb.lows = {7: 99}
    f2 = fb.build()
    eng2 = make_engine(cfg)
    for k in range(8):
        step(eng2, f2, k)
    assert eng2.closed[0].exit_reason == M.STOP_TRAIL_2 and eng2.closed[0].exit_price == pytest.approx(fav(s, 106))


@pytest.mark.parametrize("s", SIDES)
def test_failure_checked_after_transition_on_same_bar(cfg, s):
    """1H closes >=0 with a negative slope AND the 4H closes >=0: C promotes to RIDE before D -> no failure."""
    fb = FB(s=s).cross_at(2)
    fb.closed1h[6] = fb.closed4h[6] = True
    fb.set("pmo1h", slice(6, None), 0.4).set("slope1h", slice(6, None), -0.2).set("pmo4h", slice(6, None), 0.4)
    eng, f = make_engine(cfg), fb.build()
    for k in range(7):
        step(eng, f, k)
    tr = eng.open["X"]
    assert tr.stage == M.RIDE
    assert tr.trail_stage == 3            # E: RIDE + closed 1H candle with SLOPE_1H < 0 -> tightened immediately


@pytest.mark.parametrize("s", SIDES)
def test_slot_freed_by_stop_is_usable_by_entries_same_bar(cfg, s):
    c = ovr(cfg, **{"trading.max_open_trades": 1})
    a = FB("A", s=s, n=20).cross_at(2)
    a.price(slice(5, None), 90)                    # stop 97 hit on bar 5
    b = FB("B", s=s, n=20)
    b.set("pmo15", slice(None), -1).set("pmo15", slice(5, None), 1)    # B crosses on bar 5
    eng = make_engine(c)
    feats = {"A": a.build(), "B": b.build()}
    for k in range(6):
        step(eng, feats, k)
    assert [t.symbol for t in eng.closed] == ["A"] and "B" in eng.open and not eng.rejected


@pytest.mark.parametrize("s", SIDES)
def test_no_reentry_on_exit_candle_but_on_later_cross(cfg, s):
    fb = FB(s=s, n=30)
    fb.set("pmo15", slice(0, 2), -1).set("pmo15", slice(2, 5), 1)      # cross at 2
    fb.set("pmo15", slice(5, 8), -1)                                    # back under at 5 (ignored while open)
    fb.set("pmo15", slice(8, None), 1)                                  # cross at 8
    fb.price(slice(5, 8), 90)                                           # stop hit on bar 5
    eng, f = make_engine(cfg), fb.build()
    for k in range(7):
        step(eng, f, k)
    assert len(eng.closed) == 1 and not eng.open                        # no re-entry while flat w/o a new cross
    step(eng, f, 7)
    step(eng, f, 8)
    assert "X" in eng.open and eng.open["X"].entry_ts == 8 * 900


@pytest.mark.parametrize("s", SIDES)
def test_cross_on_exit_candle_is_ignored(cfg, s):
    fb = FB(s=s, n=30)
    fb.set("pmo15", slice(0, 3), -1).set("pmo15", slice(3, 5), 1)       # cross at 3 (entry)
    fb.set("pmo15", slice(5, 6), 1).set("pmo15", slice(6, None), -1)    # a (new-direction) cross at 6
    fb.price(slice(4, None), 90)                                        # stop hit on bar 4 (exit candle)
    fb.set("pmo15", slice(4, 5), 1)
    eng, f = make_engine(cfg), fb.build()
    for k in range(5):
        step(eng, f, k)
    assert len(eng.closed) == 1 and eng.closed[0].exit_ts == 4 * 900 and not eng.open
    step(eng, f, 5)
    step(eng, f, 6)
    assert "X" in eng.open and eng.open["X"].entry_ts == 6 * 900   # only the cross AFTER the exit candle


@pytest.mark.parametrize("s", SIDES)
def test_cross_while_open_in_either_direction_ignored(cfg, s):
    fb = FB(s=s, n=30).cross_at(2)
    fb.set("pmo15", slice(6, 8), -1).set("pmo15", slice(8, None), 1)   # reversal and re-cross while open (RIDE)
    fb.set("pmo1h", slice(None), 1).set("pmo4h", slice(None), 1)
    eng, f = make_engine(cfg), fb.build()
    for k in range(12):
        step(eng, f, k)
    assert len(eng.open) == 1 and eng.open["X"].entry_ts == 2 * 900 and not eng.closed


# ───────────── section 8 priorities ─────────────
@pytest.mark.parametrize("s", SIDES)
def test_priority_stage_then_eta_then_liquidity(cfg, s):
    c = ovr(cfg, **{"trading.max_open_trades": 3})
    def mk(name, p1, p4, slope1=1.0, pmo1=-1.0, liq=1e6):
        fb = FB(name, s=s, n=10)
        fb.set("pmo15", slice(None), -1).set("pmo15", slice(3, None), 1)
        fb.set("pmo1h", slice(None), pmo1 if p1 < 0 else 1).set("pmo4h", slice(None), -1 if p4 < 0 else 1)
        fb.set("slope1h", slice(None), slope1).a["liq24"][:] = liq
        return fb.build()
    feats = {
        "RIDE_": mk("RIDE_", 1, 1),
        "W4": mk("W4", 1, -1),
        "ETA_SMALL": mk("ETA_SMALL", -1, -1, slope1=1.0, pmo1=-1.0),     # eta 1
        "ETA_BIG": mk("ETA_BIG", -1, -1, slope1=0.2, pmo1=-1.0),         # eta 5
        "ETA_INF": mk("ETA_INF", -1, -1, slope1=-0.5),                    # inf
        "LIQ_HI": mk("LIQ_HI", -1, -1, slope1=-0.5, liq=9e9),             # inf, higher liquidity
    }
    eng = make_engine(c)
    for k in range(4):
        step(eng, feats, k)
    assert set(eng.open) == {"RIDE_", "W4", "ETA_SMALL"}
    rej = {r["symbol"]: r["reason"] for r in eng.rejected}
    assert rej == {n: "max_open_full" for n in ["ETA_BIG", "ETA_INF", "LIQ_HI"]}
    # liquidity tiebreak: raise the cap to 4 -> among equal-ETA (inf) the more liquid goes first
    c4 = ovr(cfg, **{"trading.max_open_trades": 4})
    eng4 = make_engine(c4)
    for k in range(4):
        step(eng4, feats, k)
    assert "ETA_BIG" in eng4.open and "LIQ_HI" not in eng4.open
    c5 = ovr(cfg, **{"trading.max_open_trades": 5})
    eng5 = make_engine(c5)
    for k in range(4):
        step(eng5, feats, k)
    assert "LIQ_HI" in eng5.open and "ETA_INF" not in eng5.open   # 6 signals, 5 slots: liquidity breaks the inf-ETA tie
    # inf-vs-inf tiebreak by liquidity with exactly one slot left for them
    feats2 = {k: v for k, v in feats.items() if k in ("ETA_INF", "LIQ_HI")}
    c1 = ovr(cfg, **{"trading.max_open_trades": 1})
    e1 = make_engine(c1)
    for k in range(4):
        step(e1, feats2, k)
    assert list(e1.open) == ["LIQ_HI"]


# ───────────── section 6 filters ─────────────
@pytest.mark.parametrize("s", SIDES)
@pytest.mark.parametrize("flag,breaker,reason", [
    ("filter_1h_slope", lambda fb: fb.set("slope1h", slice(None), -1), "FILTER_1H_SLOPE"),
    ("filter_1h_signal", lambda fb: fb.set("pmo1h", slice(None), -1).set("signal1h", slice(None), 1), "FILTER_1H_SIGNAL"),
    ("filter_eta_max", lambda fb: fb.set("slope1h", slice(None), 0.05), "FILTER_ETA_MAX"),
    ("filter_4h_slope", lambda fb: fb.set("slope4h", slice(None), -1), "FILTER_4H_SLOPE"),
])
def test_filters(cfg, s, flag, breaker, reason):
    c = ovr(cfg, **{f"pmo_zero_cross.filters.{flag}": True})
    ok = FB(s=s, n=6).cross_at(2).set("pmo1h", slice(None), -0.5).set("signal1h", slice(None), -1)
    bad = FB(s=s, n=6).cross_at(2).set("pmo1h", slice(None), -0.5).set("signal1h", slice(None), -1)
    breaker(bad)
    for fb, expect_open in ((ok, True), (bad, False)):
        eng, f = make_engine(c), fb.build()
        for k in range(4):
            step(eng, f, k)
        assert ("X" in eng.open) == expect_open
        if not expect_open:
            assert eng.rejected[0]["reason"] == reason
    # default config (filters off) takes the bad one too
    eng, f = make_engine(cfg), bad.build()
    for k in range(4):
        step(eng, f, k)
    assert "X" in eng.open


@pytest.mark.parametrize("s", SIDES)
def test_filter_btc(cfg, s):
    c = ovr(cfg, **{"pmo_zero_cross.filters.filter_btc": True})
    x = FB("X", s=s, n=6).cross_at(2).build()
    for btc_fav, expect in ((1, True), (-1, False)):
        btc = FB("BTC_USDT", s=s, n=6).set("pmo15", slice(None), btc_fav).build()
        eng = make_engine(c)
        for k in range(4):
            eng.process_bar(k * 900, {"X": x}, btc)
        assert ("X" in eng.open) == expect


@pytest.mark.parametrize("s", SIDES)
def test_entry_snapshot_recorded(cfg, s):
    fb = FB(s=s, n=6).cross_at(2).set("pmo1h", slice(None), -2).set("slope1h", slice(None), 0.5)
    btc = FB("BTC_USDT", s=s, n=6).set("pmo15", slice(None), 0.7).build()
    eng, f = make_engine(cfg), fb.build()
    for k in range(4):
        eng.process_bar(k * 900, {"X": f}, btc)
    sn = eng.open["X"].snap
    assert sn["h1"]["pos"] == "below" and sn["h1"]["slope_dir"] == "with"
    assert sn["h1"]["eta"] == pytest.approx(4.0)                      # |-2| / 0.5
    assert sn["h4"]["pos"] == "below" and sn["btc15"]["pmo"] == pytest.approx(0.7 * s)
    assert "btc1h" in sn
