"""Config loader. Every value comes from config.toml; there are no defaults in code,
so a missing key fails loudly instead of silently using a hidden number."""
from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    import tomli as tomllib
from datetime import datetime, timezone
from pathlib import Path


class Section:
    def __init__(self, data: dict, path: str):
        self._path = path
        self._data = {}
        for k, v in data.items():
            self._data[k] = Section(v, f"{path}.{k}") if isinstance(v, dict) else v

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise KeyError(f"missing config key: [{self._path}] {name}") from None

    def get(self, name, default=None):
        return self._data.get(name, default)

    def has(self, name) -> bool:
        return name in self._data

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Section) else v for k, v in self._data.items()}

    def override(self, **kw) -> "Section":
        """Return a copy with top-level keys replaced (used by sweeps)."""
        d = self.to_dict()
        d.update(kw)
        return Section(d, self._path)


def load_config(path: str | Path = "config.toml") -> Section:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = Section(raw, "root")
    _validate(cfg)
    return cfg


def parse_time(s: str) -> int:
    """ISO-8601 (Z) -> epoch seconds."""
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp())


def _validate(cfg: Section) -> None:
    t = cfg.trading
    if t.margin_type not in ("isolated", "cross"):
        raise ValueError("trading.margin_type must be isolated|cross")
    if t.max_open_trades < 1 or t.leverage < 1 or t.trade_size_usdt <= 0:
        raise ValueError("invalid trading size/leverage/max_open_trades")
    if t.stop_loss_pct <= 0:
        raise ValueError("trading.stop_loss_pct must be > 0")
