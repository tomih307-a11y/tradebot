"""Connects the five agents: scanner -> technical -> news -> entry -> (SignalTrader opens) -> exit.
Opening/limits (10% margin, max 2 concurrent) are enforced by SignalTrader exactly as for channel signals."""
from __future__ import annotations

import logging
import time

import numpy as np

from tradebot.data.klines import drop_unclosed, fetch_range
from tradebot.indicators import atr_rma
from tradebot.mexc.client import TF_SECONDS
from tradebot.strategies.telegram_signals.parser import Signal, parse_signal, parse_update

from .entry import EntryAgent
from .consensus import Consensus
from .execution import ExecutionAgent
from .exit import ExitAgent
from .ledger import Ledger
from .news import NewsAgent
from .scanner import ScannerAgent
from .technical import TechnicalAgent
from .types import Recommendation, Vote

log = logging.getLogger("multi_agent")


class Orchestrator:
    def __init__(self, cfg, client, trader, reader=None, execution=None):
        self.cfg, self.a, self.client, self.trader, self.reader = cfg, cfg.agents, client, trader, reader
        self._first = "seen" not in trader.state
        self.consensus, self.ledger = Consensus(cfg), Ledger(self.a.consensus.ledger_file)
        self.execution = execution if execution is not None else (ExecutionAgent(cfg, trader) if self.a.execution.enabled else None)
        self.votes: dict[str, dict] = {}
        trader.on_close = lambda pos: self.ledger.outcome(pos, self.votes.pop(pos.symbol, {}), int(time.time()))
        self.scanner, self.tech, self.news = ScannerAgent(cfg), TechnicalAgent(cfg), NewsAgent(cfg)
        self.entry, self.exit = EntryAgent(cfg), ExitAgent(cfg)
        self.pending: dict[str, Recommendation] = {}
        self.cooldown: dict[str, int] = {}
        self._last_scan = 0
        self._seq = 0

    def _klines(self, symbol: str, tf: str, bars: int, now: int) -> dict:
        step = TF_SECONDS[tf]
        return drop_unclosed(fetch_range(self.client, symbol, tf, int(now // step * step) - bars * step, now), tf, now)

    # ── channel signals pass through the agents (entry timing, news veto); targets and stop stay the channel's ──
    def poll_channel(self, now: int) -> None:
        ch, nt, st = self.a.channel, self.trader.nt, self.trader.state
        msgs = self.reader.fetch()
        if self._first:                                         # never trade the existing backlog
            st["seen"] = max([m["id"] for m in msgs] or [0])
            self._first = False
            nt.send(f"▶️ بدء متابعة القناة من الرسالة {st['seen']} (لن تُنفَّذ الرسائل السابقة)")
        for m in msgs:
            if m["id"] <= st["seen"]:
                continue
            st["seen"] = m["id"]
            u = parse_update(m["text"], m["id"], m["ts"])
            if u and u.symbol in self.pending and self.pending[u.symbol].signal is not None:
                r = self.pending[u.symbol]
                if r.side == "long" and u.new_stop > r.stop or r.side == "short" and u.new_stop < r.stop:
                    r.stop = u.new_stop
                continue
            if u or not ch.refine_entry:
                self.trader.on_message(m, now)                  # stop update of an open trade, or direct entry
                continue
            sig = parse_signal(m["text"], m["id"], m["ts"])
            if sig:
                self._queue_channel_signal(sig, now)

    def _queue_channel_signal(self, sig: Signal, now: int) -> None:
        nt, ch = self.trader.nt, self.a.channel
        if sig.symbol in self.trader.open or sig.symbol in self.pending:
            nt.send(f"⏭ تجاهل توصية {sig.symbol}: صفقة أو توصية قائمة")
            return
        s = 1 if sig.side == "long" else -1
        if bool(ch.news_veto):
            nv = self.news.verdict(sig.symbol)
            if s * nv.bias <= -float(self.a.news.veto_bias):
                nt.send(f"🚫 الأخبار ترفض توصية {sig.symbol} ({sig.side}): {nv.headlines[:1]}")
                return
        try:
            k = self._klines(sig.symbol, "Min60", 60, now)
            atr = float(atr_rma(k["high"], k["low"], k["close"], int(self.a.technical.atr_len))[-1])
        except Exception:
            atr = float("nan")
        if not np.isfinite(atr) or atr <= 0:
            atr = sig.entry * 0.01
        z = float(ch.zone_pct) / 100.0
        rec = Recommendation(sig.symbol, sig.side, 1.0, sig.entry, atr, (sig.entry * (1 - z), sig.entry * (1 + z)),
                             sig.stop, list(sig.tps), ["توصية القناة"], now, signal=sig)
        self.pending[sig.symbol] = rec
        nt.send(f"📨 توصية القناة {sig.symbol} {sig.side} → وكيل الدخول يراقب منطقة {rec.zone[0]:.6g}–{rec.zone[1]:.6g}")

    def scan(self, now: int) -> None:
        nt = self.trader.nt
        cands = self.scanner.rank(self.client.tickers())
        top = cands[0].score if cands else 0.0
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
            rec.votes = [Vote("scanner", cd.score / top if top > 0 else 0.0, False, f"سيولة/حركة {cd.change_pct:+.1f}%"),
                         Vote("technical", rec.score, False, "، ".join(rec.reasons)),
                         Vote("news", (s * nv.bias + 1) / 2, s * nv.bias <= -float(self.a.news.veto_bias),
                              (nv.headlines[:1] or ["لا أخبار"])[0]),
                         self.exit.assess(rec)]
            rec.confidence, ok, why = self.consensus.decide(rec.votes)
            if not ok:
                log.info("rejected %s %s conf=%.2f %s", rec.symbol, rec.side, rec.confidence, why)
                if why:
                    nt.send(f"🚫 رُفضت {rec.symbol} ({rec.side}) — {why}")
                continue
            self.ledger.published(rec, now)
            found += 1
            nt.send(f"✅ توصية {rec.symbol} {rec.side} · ثقة {rec.confidence:.0%} · منطقة {rec.zone[0]:.6g}–{rec.zone[1]:.6g} · "
                    f"وقف {rec.stop:.6g} · أهداف {', '.join(f'{t:.6g}' for t in rec.tps)} · "
                    + " | ".join(f"{v.agent} {v.score:.0%}" for v in rec.votes))
            if self.execution is not None:
                self.pending[rec.symbol] = rec
                self.votes[rec.symbol] = {v.agent: round(v.score, 3) for v in rec.votes}
        # keep only the best few
        mine = [r for r in self.pending.values() if r.signal is None]
        drop = {r.symbol for r in sorted(mine, key=lambda r: -r.score)[int(self.a.max_pending):]}
        self.pending = {k: v for k, v in self.pending.items() if k not in drop}

    def try_entries(self, now: int) -> None:
        for sym, rec in list(self.pending.items()):
            ttl = int(self.a.channel.expire_sec) if rec.signal is not None else int(self.a.entry.expire_sec)
            if now - rec.created_ts > ttl:
                if rec.signal is not None:
                    self.trader.nt.send(f"⌛ انتهت صلاحية توصية القناة {sym} دون دخول مناسب")
                del self.pending[sym]
                continue
            if not self.execution.can_open():
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
            if rec.signal is not None:                      # channel trade: its stop, targets and leverage
                from tradebot.strategies.telegram_signals.parser import Signal as _S
                o = rec.signal
                self.trader.try_open(_S(msg_id=o.msg_id, ts=now, symbol=sym, side=rec.side, entry=price, stop=rec.stop,
                                        tps=list(o.tps), leverage=o.leverage), now)
            else:
                self.trader.nt.send(f"🎯 الدخول: {sym} — {plan.reason}")
                self.execution.open(rec, price, now)
            self.cooldown[sym] = now + int(self.a.cooldown_sec)

    def manage_exits(self, now: int) -> None:
        for sym, pos in list(self.trader.open.items()):
            if pos.closed or (pos.signal_msg_id >= 0 and not bool(self.a.channel.manage_exit)):
                continue
            try:
                k = self._klines(sym, "Min60", 60, now)
                atr = float(atr_rma(k["high"], k["low"], k["close"], int(self.a.technical.atr_len))[-1])
                act = self.exit.manage(pos, self.client.last_price(sym), atr, self.news.verdict(sym).bias, now)
            except Exception as e:
                log.warning("exit failed %s: %s", sym, e)
                continue
            if act.close:
                self.execution.close(pos, now)
                self.trader.nt.send(f"🚪 الخروج: {sym} إغلاق — {act.reason}")
            elif act.new_stop is not None:
                self.execution.move_stop(pos, act.new_stop)
                self.trader.nt.send(f"🔧 الخروج: {sym} الوقف إلى {act.new_stop:.6g}")

    def cycle(self, now: int) -> None:
        if self.reader is not None:
            try:
                self.poll_channel(now)
            except Exception:
                log.exception("channel poll failed")
        if bool(self.a.own_analysis) and now - self._last_scan >= int(self.a.scan_interval_sec):
            self._last_scan = now
            self.scan(now)
        if self.execution is not None:
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
