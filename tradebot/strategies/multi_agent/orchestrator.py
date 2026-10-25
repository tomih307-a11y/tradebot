"""Connects the five agents: scanner -> technical -> news -> entry -> (SignalTrader opens) -> exit.
Opening/limits (10% margin, max 2 concurrent) are enforced by SignalTrader exactly as for channel signals."""
from __future__ import annotations

import logging
import time

import numpy as np

from tradebot.data.klines import drop_unclosed, fetch_range
from tradebot.indicators import atr_rma
from tradebot.mexc.client import TF_SECONDS
from tradebot.strategies.telegram_signals.parser import Signal

from .entry import EntryAgent
from .exit import ExitAgent
from .news import NewsAgent
from .scanner import ScannerAgent
from .technical import TechnicalAgent
from .types import Recommendation

log = logging.getLogger("multi_agent")


class Orchestrator:
    def __init__(self, cfg, client, trader):
        self.cfg, self.a, self.client, self.trader = cfg, cfg.agents, client, trader
        self.scanner, self.tech, self.news = ScannerAgent(cfg), TechnicalAgent(cfg), NewsAgent(cfg)
        self.entry, self.exit = EntryAgent(cfg), ExitAgent(cfg)
        self.pending: dict[str, Recommendation] = {}
        self.cooldown: dict[str, int] = {}
        self._last_scan = 0
        self._seq = 0

    def _klines(self, symbol: str, tf: str, bars: int, now: int) -> dict:
        step = TF_SECONDS[tf]
        return drop_unclosed(fetch_range(self.client, symbol, tf, int(now // step * step) - bars * step, now), tf, now)

    def scan(self, now: int) -> None:
        nt = self.trader.nt
        cands = self.scanner.rank(self.client.tickers())
        found = 0
        for cd in cands:
            if cd.symbol in self.pending or cd.symbol in self.trader.open or now < self.cooldown.get(cd.symbol, 0):
                continue
            try:
                rec = self.tech.analyze(cd, self._klines(cd.symbol, "Min60", 300, now), self._klines(cd.symbol, "Hour4", 300, now), now)
            except Exception as e:
                log.warning("technical failed %s: %s", cd.symbol, e)
                continue
            if not rec:
                continue
            nv = self.news.verdict(cd.symbol)
            s = 1 if rec.side == "long" else -1
            rec.news_bias = nv.bias
            if s * nv.bias <= -float(self.a.news.veto_bias):
                nt.send(f"🚫 الأخبار ترفض {rec.symbol} ({rec.side}): {nv.headlines[:1]}")
                continue
            rec.score += float(self.a.news.weight) * s * nv.bias
            self.pending[rec.symbol] = rec
            found += 1
            nt.send(f"📊 المحلل: {rec.symbol} {rec.side} درجة {rec.score:.2f} منطقة {rec.zone[0]:.6g}–{rec.zone[1]:.6g} "
                    f"وقف {rec.stop:.6g} ({'، '.join(rec.reasons)})")
        # keep only the best few
        keep = sorted(self.pending.values(), key=lambda r: -r.score)[: int(self.a.max_pending)]
        self.pending = {r.symbol: r for r in keep}

    def try_entries(self, now: int) -> None:
        for sym, rec in list(self.pending.items()):
            if now - rec.created_ts > int(self.a.entry.expire_sec):
                del self.pending[sym]
                continue
            if len(self.trader.open) >= int(self.cfg.telegram_signals.max_concurrent):
                return
            price = self.client.last_price(sym)
            k1m = self._klines(sym, "Min1", 10, now)
            plan = self.entry.plan(rec, price, k1m)
            if plan.action == "wait":
                continue
            del self.pending[sym]
            if plan.action == "skip":
                self.trader.nt.send(f"⏭ الدخول: {sym} تخطّي — {plan.reason}")
                self.cooldown[sym] = now + int(self.a.cooldown_sec)
                continue
            self._seq += 1
            sig = Signal(msg_id=-self._seq, ts=now, symbol=sym, side=rec.side, entry=price, stop=rec.stop,
                         tps=rec.tps, leverage=None)
            self.trader.nt.send(f"🎯 الدخول: {sym} — {plan.reason}")
            self.trader.try_open(sig, now)
            self.cooldown[sym] = now + int(self.a.cooldown_sec)

    def manage_exits(self, now: int) -> None:
        for sym, pos in list(self.trader.open.items()):
            if pos.closed or pos.signal_msg_id >= 0:        # only positions opened by the agents
                continue
            try:
                k = self._klines(sym, "Min60", 60, now)
                atr = float(atr_rma(k["high"], k["low"], k["close"], int(self.a.technical.atr_len))[-1])
                act = self.exit.manage(pos, self.client.last_price(sym), atr, self.news.verdict(sym).bias, now)
            except Exception as e:
                log.warning("exit failed %s: %s", sym, e)
                continue
            if act.close:
                self.trader.ex.close(pos, now)
                self.trader.nt.send(f"🚪 الخروج: {sym} إغلاق — {act.reason}")
            elif act.new_stop is not None:
                self.trader.ex.move_stop(pos, act.new_stop)
                self.trader.nt.send(f"🔧 الخروج: {sym} الوقف إلى {act.new_stop:.6g}")

    def cycle(self, now: int) -> None:
        if now - self._last_scan >= int(self.a.scan_interval_sec):
            self._last_scan = now
            self.scan(now)
        self.try_entries(now)
        self.manage_exits(now)
        self.trader.tick(now)

    def run(self, once: bool = False) -> None:
        while True:
            try:
                self.cycle(int(time.time()))
            except Exception:
                log.exception("cycle failed")
            if once:
                return
            time.sleep(int(self.cfg.telegram_signals.poll_interval_sec))
