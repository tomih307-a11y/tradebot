"""Records every published recommendation with each agent's vote, and its realised outcome — the data to learn which agents are right."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _iso(t: int) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Ledger:
    def __init__(self, path: str):
        self.path = Path(path)

    def _write(self, row: list, header: list) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow(row)

    def published(self, rec, now: int) -> None:
        self._write([_iso(now), "published", rec.symbol, rec.side, f"{rec.confidence:.3f}",
                     json.dumps({v.agent: round(v.score, 3) for v in rec.votes}, ensure_ascii=False), "", ""],
                    ["time", "event", "symbol", "side", "confidence", "votes", "realized_usdt", "exit_reason"])

    def outcome(self, pos, votes: dict, now: int) -> None:
        self._write([_iso(now), "closed", pos.symbol, pos.side, "", json.dumps(votes, ensure_ascii=False),
                     f"{pos.realized:.4f}", pos.exit_reason],
                    ["time", "event", "symbol", "side", "confidence", "votes", "realized_usdt", "exit_reason"])
