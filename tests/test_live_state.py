"""Live pieces that can run offline: DRY_RUN executor + state save/restore round trip."""
import json

from conftest import FB, make_engine, ovr, step
from tradebot.mexc.client import MexcClient
from tradebot.strategies.pmo_zero_cross.live import LiveExecutor, load_state, save_state
from tradebot.strategies.pmo_zero_cross.engine import PMOEngine
from tradebot.strategies.pmo_zero_cross.costs import CostModel
from tradebot.strategies.pmo_zero_cross.notify import Notifier


def test_dry_run_sends_nothing_and_state_roundtrips(cfg, tmp_path):
    assert cfg.trading.dry_run is True
    client = MexcClient(cfg.mexc)
    client._request = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network call in DRY_RUN"))
    msgs = []
    notifier = Notifier(cfg)
    notifier.send = msgs.append
    eng = PMOEngine(cfg, LiveExecutor(cfg, client, notifier), CostModel(cfg), on_event=notifier.handle)
    fb = FB(s=1, n=20).cross_at(2).set("pmo1h", slice(None), 1)
    fb.price(slice(3, None), 110)
    f = fb.build()
    for k in range(7):
        step(eng, f, k)
    assert any("دخول" in m for m in msgs) and any("تحريك الوقف" in m for m in msgs)
    assert any("لم يُرسل" in m for m in msgs)

    path = tmp_path / "s.json"
    save_state(path, eng, 6 * 900)
    eng2 = PMOEngine(cfg, LiveExecutor(cfg, client, notifier), CostModel(cfg))
    assert load_state(path, eng2) == 6 * 900
    a, b = eng.open["X"], eng2.open["X"]
    assert a.to_dict() == b.to_dict() and a.stop == b.stop and b.stage == a.stage
    # restored engine continues exactly like the original
    step(eng, f, 7); step(eng2, f, 7)
    assert eng.open["X"].to_dict() == eng2.open["X"].to_dict()
