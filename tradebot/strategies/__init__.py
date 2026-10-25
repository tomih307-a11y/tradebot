"""Strategy registry — the strategy is chosen by [general] strategy in config.toml."""
from __future__ import annotations

REGISTRY = {
    "pmo_zero_cross": "tradebot.strategies.pmo_zero_cross",
    "telegram_signals": "tradebot.strategies.telegram_signals",
    "multi_agent": "tradebot.strategies.multi_agent",
}


def load_strategy(name: str):
    import importlib
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}")
    return importlib.import_module(REGISTRY[name])
