import pytest

from tradebot.config import load_config
from tradebot.strategies.telegram_signals.gui import GuiController


def test_start_requires_channel():
    c = GuiController(load_config("config.toml"), "")
    with pytest.raises(ValueError):
        c.start()


def test_snapshot_before_start_shows_initial_balance():
    cfg = load_config("config.toml")
    s = GuiController(cfg, "x").snapshot()
    assert s["balance"] == float(cfg.telegram_signals.paper_balance_usdt) and s["positions"] == []


def test_live_mode_refused(monkeypatch):
    cfg = load_config("config.toml")
    cfg.trading.dry_run = False
    with pytest.raises(SystemExit):
        GuiController(cfg, "x").start()
