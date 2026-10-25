"""Collaboration rule: every agent votes 0..1 (and may veto); a recommendation is published only if the weighted
confidence reaches the threshold and nobody vetoed."""
from __future__ import annotations

from .types import Vote


class Consensus:
    def __init__(self, cfg):
        self.c = cfg.agents.consensus

    def decide(self, votes: list[Vote]) -> tuple[float, bool, str]:
        w = self.c.weights
        tot = sum(float(w.get(v.agent)) for v in votes)
        conf = sum(float(w.get(v.agent)) * v.score for v in votes) / tot if tot else 0.0
        vetoers = [f"{v.agent}: {v.note}" for v in votes if v.veto]
        ok = not vetoers and conf >= float(self.c.min_confidence)
        return conf, ok, "؛ ".join(vetoers)
