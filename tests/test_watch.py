from tools.watch_levels import check


def test_check_long_levels_and_stop():
    lv = {"tp": [3.37, 3.53, 3.63], "stop": 3.0, "long": True}
    hit = set()
    assert check(lv, hit, 3.30, 3.20) == []
    assert check(lv, hit, 3.55, 3.30) == ["TP1 بلغ 3.37", "TP2 بلغ 3.53"]
    assert check(lv, hit, 3.55, 3.30) == []                      # no repeats
    assert check(lv, hit, 3.40, 2.99) == ["الوقف لُمس عند 3.0"]
