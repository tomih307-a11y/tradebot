"""Agent 3 — news analyst: reads RSS headlines, scores them per coin and for the market.
Scoring: Claude (if ANTHROPIC_API_KEY is set and [agents.news] use_claude = true), else a plain keyword lexicon."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

from .types import NewsVerdict

log = logging.getLogger("multi_agent")

POS = ("approve", "approval", "etf", "partnership", "adopt", "listing", "surge", "rally", "record high", "upgrade", "inflow",
       "bullish", "breakout", "buyback", "launch")
NEG = ("hack", "exploit", "lawsuit", "sue", "ban", "crackdown", "delist", "outflow", "bearish", "liquidat", "crash", "plunge",
       "fraud", "sec charges", "rug", "bankrupt", "investigation", "downgrade", "selloff", "sell-off")


def score_headlines(titles: list[str]) -> float:
    """-1..+1 from keyword hits (pure)."""
    pos = sum(any(w in t.lower() for w in POS) for t in titles)
    neg = sum(any(w in t.lower() for w in NEG) for t in titles)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


class NewsAgent:
    def __init__(self, cfg):
        self.c = cfg.agents.news
        self._cache: list[tuple[float, str]] = []
        self._fetched = 0.0

    def _headlines(self) -> list[str]:
        if time.time() - self._fetched < float(self.c.ttl_sec) and self._cache:
            return [t for _, t in self._cache]
        items = []
        for url in self.c.feeds:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                root = ET.fromstring(urllib.request.urlopen(req, timeout=15).read())
                items += [(0.0, (i.findtext("title") or "").strip()) for i in root.iter("item")][: int(self.c.per_feed)]
            except Exception as e:
                log.warning("feed failed %s: %s", url, e)
        self._cache, self._fetched = items, time.time()
        return [t for _, t in items]

    def verdict(self, symbol: str) -> NewsVerdict:
        base = symbol.split("_")[0]
        pat = re.compile(rf"\b{re.escape(base)}\b", re.I)
        titles = self._headlines()
        mine = [t for t in titles if pat.search(t)]
        market = [t for t in titles if re.search(r"\b(bitcoin|btc|crypto|fed|sec)\b", t, re.I)]
        pick = (mine or market)[: int(self.c.max_headlines)]
        if not pick:
            return NewsVerdict(0.0, [])
        bias = self._claude(base, pick)
        if bias is None:
            bias = score_headlines(pick)
        return NewsVerdict(max(-1.0, min(1.0, bias)), pick)

    def _claude(self, base: str, titles: list[str]) -> float | None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not (key and self.c.use_claude):
            return None
        prompt = (f"Headlines about {base} / the crypto market:\n- " + "\n- ".join(titles) +
                  "\nRate the likely short-term price impact on a futures trade from -1 (very bearish) to +1 (very bullish). "
                  "Reply with only a number.")
        try:
            body = json.dumps({"model": self.c.claude_model, "max_tokens": 10, "messages": [{"role": "user", "content": prompt}]}).encode()
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
                "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=30).read())
            return float(re.search(r"-?\d+(?:\.\d+)?", out["content"][0]["text"]).group())
        except Exception as e:
            log.warning("claude news scoring failed: %s", e)
            return None
