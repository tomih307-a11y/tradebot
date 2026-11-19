from tools.watch_levels import check


def test_check_long_levels_and_stop():
    lv = {"tp": [3.37, 3.53, 3.63], "stop": 3.0, "long": True}
    hit = set()
    assert check(lv, hit, 3.30, 3.20) == []
    assert check(lv, hit, 3.55, 3.30) == ["TP1 بلغ 3.37", "TP2 بلغ 3.53"]
    assert check(lv, hit, 3.55, 3.30) == []                      # no repeats
    assert check(lv, hit, 3.40, 2.99) == ["الوقف لُمس عند 3.0"]


def test_trail_step_ratchets_and_hits():
    from tools.watch_levels import trail_step
    hit, stop, hh = trail_step(100.0, 95.0, 110.0, 100.0, 5.0, 95.0)
    assert (hit, stop, hh) == (False, 105.0, 110.0)               # new high lifts the stop
    hit, stop2, hh2 = trail_step(hh, stop, 108.0, 106.0, 5.0, 95.0)
    assert (hit, stop2, hh2) == (False, 105.0, 110.0)             # stop never falls
    assert trail_step(hh, stop, 109.0, 104.0, 5.0, 95.0)[0] is True   # low touches the stop


def test_resim_trail_hit_and_open():
    from tools.resim_trail import simulate
    st, stop, hh, te, px = simulate([(1, 101, 99.5), (2, 106, 100), (3, 105, 103.0)], 100.0, 4.0)
    assert st == "سارية" and hh == 106 and stop == 102.0
    st, stop, hh, te, px = simulate([(1, 101, 99.5), (2, 106, 100), (3, 105, 101.9)], 100.0, 4.0)
    assert st == "ضُرب" and te == 3 and px == 102.0
