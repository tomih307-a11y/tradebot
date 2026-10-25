"""Trade the channel's signals. dry_run = true: PAPER (fills simulated with real MEXC 1m candles, no orders sent).
dry_run = false: REAL orders on MEXC via live.LiveExchange (needs API keys in the environment)."""
from __future__ import annotations

import csv
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from tradebot.mexc.client import MexcClient

from . import paper
from .parser import Signal, parse_signal, parse_update
from .paper import Position

log = logging.getLogger("telegram_signals")


class SkipSignal(Exception):
    """A valid signal that we decide not to take (reason in the message)."""


# ───────────────────────── notifier ─────────────────────────
class Notifier:
    def __init__(self, cfg):
        live = cfg.pmo_zero_cross.live
        self.token = os.environ.get(live.telegram_token_env, "")
        self.chat = os.environ.get(live.telegram_chat_id_env, "")
        self.tag = "[ورقي] "

    def send(self, text: str) -> None:
        log.info(self.tag + text.replace("\n", " | "))
        if self.token and self.chat:
            try:
                data = urllib.parse.urlencode({"chat_id": self.chat, "text": self.tag + text}).encode()
                urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendMessage", data=data), timeout=10)
            except Exception as e:
                log.warning("notify failed: %s", e)


# ───────────────────────── paper exchange ─────────────────────────
class PaperExchange:
    def __init__(self, cfg, client: MexcClient, state: dict):
        t = cfg.telegram_signals
        self.cfg, self.t, self.client, self.state = cfg, t, client, state
        state.setdefault("balance", float(t.paper_balance_usdt))
        self._detail: dict = {}
        self.cfg_ts = {"fee_pct": t.paper_fee_pct, "slip_pct": t.paper_slippage_pct, "breakeven_after_tp1": bool(t.breakeven_after_tp1)}

    def balance(self) -> float:
        return self.state["balance"]

    def contract(self, symbol: str) -> dict | None:
        if symbol not in self._detail:
            try:
                self._detail[symbol] = self.client.contract_detail(symbol)
            except Exception:
                self._detail[symbol] = None
        return self._detail[symbol]

    def last_price(self, symbol: str) -> float:
        return self.client.last_price(symbol)

    def _apply(self, pos: Position) -> None:
        delta = pos.realized - pos.ext.get("applied", 0.0)
        self.state["balance"] += delta
        pos.ext["applied"] = pos.realized

    def open(self, sig: Signal, leverage: int, margin: float, now: int) -> Position:
        px = self.last_price(sig.symbol)
        s = 1 if sig.side == "long" else -1
        fill = px * (1 + s * self.t.paper_slippage_pct / 100.0)
        pos = paper.open_position(sig, fill, self.balance(), self.t.margin_pct_of_balance, leverage, now,
                                  self.t.stop_mode, self.t.paper_fee_pct, self.t.maintenance_margin)
        self._apply(pos)
        return pos

    def advance(self, pos: Position, now: int) -> list:
        """Feed all closed 1m candles since the last processed one. Returns new events [(ts, kind, price, frac, pnl)]."""
        n0 = len(pos.events)
        if pos.closed:
            return []
        try:
            d = self.client.klines(pos.symbol, "Min1", max(pos.last_ts, pos.opened_ts), now)
        except Exception as e:
            log.warning("klines failed %s: %s", pos.symbol, e)
            return []
        ts = d.get("time", [])
        for i in range(len(ts)):
            if ts[i] + 60 > now:
                break
            paper.step(pos, (int(ts[i]), float(d["open"][i]), float(d["high"][i]), float(d["low"][i]), float(d["close"][i])), self.cfg_ts)
            if pos.closed:
                break
        self._apply(pos)
        return pos.events[n0:]

    def move_stop(self, pos: Position, price: float) -> None:
        pos.stop = price

    def close(self, pos: Position, now: int) -> None:
        s = pos.s
        px = self.last_price(pos.symbol) * (1 - s * self.t.paper_slippage_pct / 100.0)
        paper._close(pos, now, px, pos.remaining, "agent_exit", self.t.paper_fee_pct)
        pos.closed, pos.exit_reason, pos.exit_ts = True, "agent_exit", now
        self._apply(pos)


# ───────────────────────── trader ─────────────────────────
class SignalTrader:
    def __init__(self, cfg, exchange, notifier, state: dict, state_path: str, log_path: str):
        self.cfg, self.t, self.ex, self.nt, self.state = cfg, cfg.telegram_signals, exchange, notifier, state
        self.state_path, self.log_path = state_path, log_path
        self.open: dict[str, Position] = {k: Position.from_dict(v) for k, v in state.get("open", {}).items()}

    def save(self) -> None:
        self.state["open"] = {k: v.to_dict() for k, v in self.open.items()}
        p = Path(self.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        tmp.replace(p)

    def _log_close(self, pos: Position) -> None:
        p = Path(self.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if t else ""
        with p.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["opened", "closed", "symbol", "side", "entry", "leverage", "margin", "tp_hit", "exit_reason",
                            "realized_usdt", "fees", "balance_after"])
            w.writerow([iso(pos.opened_ts), iso(pos.exit_ts), pos.symbol, pos.side, pos.entry, pos.leverage, f"{pos.margin:.4f}",
                        pos.tp_hit, pos.exit_reason, f"{pos.realized:.4f}", f"{pos.fees:.4f}", f"{self.ex.balance():.4f}"])

    def leverage_for(self, sig: Signal, detail: dict) -> int:
        lev = sig.leverage if (self.t.leverage_mode == "signal" and sig.leverage) else int(self.t.fixed_leverage)
        mx = int(detail.get("maxLeverage") or lev)
        return max(1, min(lev, mx))

    def try_open(self, sig: Signal, now: int) -> str:
        try:
            if sig.symbol in self.open:
                raise SkipSignal("صفقة مفتوحة على نفس العملة")
            if len(self.open) >= int(self.t.max_concurrent):
                raise SkipSignal(f"الحد الأقصى للصفقات المتزامنة ({self.t.max_concurrent}) ممتلئ")
            if sig.ts and now - sig.ts > int(self.t.max_signal_age_sec):
                raise SkipSignal("التوصية قديمة")
            d = self.ex.contract(sig.symbol)
            if not d:
                raise SkipSignal("العقد غير متاح على MEXC")
            px = self.ex.last_price(sig.symbol)
            s = 1 if sig.side == "long" else -1
            if abs(px - sig.entry) / sig.entry * 100 > float(self.t.max_entry_deviation_pct):
                raise SkipSignal(f"السعر الحالي {px} بعيد عن الدخول {sig.entry}")
            if s * (px - sig.tps[0]) >= 0 or s * (px - sig.stop) <= 0:
                raise SkipSignal("السعر تجاوز الهدف الأول أو الوقف")
            lev = self.leverage_for(sig, d)
            margin = self.ex.balance() * float(self.t.margin_pct_of_balance) / 100.0
            pos = self.ex.open(sig, lev, margin, now)
        except SkipSignal as e:
            self.nt.send(f"⏭ تجاهل توصية {sig.symbol} ({sig.side}): {e}")
            return f"skip: {e}"
        except Exception as e:
            log.exception("open failed")
            self.nt.send(f"❗ فشل فتح {sig.symbol}: {e}")
            return f"error: {e}"
        self.open[sig.symbol] = pos
        self.nt.send(f"✅ دخول {sig.symbol} {'لونق' if sig.side == 'long' else 'شورت'} @ {pos.entry:.6g} · رافعة {pos.leverage}x · "
                     f"هامش {pos.margin:.2f}$ · وقف {pos.stop:.6g} · أهداف {len(pos.tps)}")
        return "opened"

    def on_message(self, m: dict, now: int) -> str | None:
        u = parse_update(m["text"], m["id"], m["ts"])
        if u and u.symbol in self.open:
            pos = self.open[u.symbol]
            if pos.s * (u.new_stop - pos.stop) > 0:               # only ever tighten
                self.ex.move_stop(pos, u.new_stop)
                self.nt.send(f"🔧 القناة نقلت وقف {u.symbol} إلى {u.new_stop}")
            return "stop-updated"
        sig = parse_signal(m["text"], m["id"], m["ts"])
        return self.try_open(sig, now) if sig else None

    def tick(self, now: int) -> None:
        for sym in list(self.open):
            pos = self.open[sym]
            for ts, kind, price, frac, pnl in self.ex.advance(pos, now):
                if kind != "open":
                    self.nt.send(f"{'🎯' if kind.startswith('tp') else '⛔'} {sym} {kind} @ {price:.6g} (حصة {frac:.0%}) · {pnl:+.3f}$")
            if pos.closed:
                self._log_close(pos)
                self.nt.send(f"🏁 أُغلقت {sym}: {pos.exit_reason} · صافي {pos.realized:+.3f}$ · الرصيد {self.ex.balance():.2f}$")
                del self.open[sym]
        self.save()

    def run(self, reader, once: bool = False) -> None:
        first = "seen" not in self.state
        while True:
            now = int(time.time())
            try:
                msgs = reader.fetch()
                if first:                                         # never trade the existing backlog
                    self.state["seen"] = max([m["id"] for m in msgs] or [0])
                    first = False
                    self.nt.send(f"▶️ بدء المتابعة من الرسالة {self.state['seen']} (لن تُنفَّذ الرسائل السابقة)")
                for m in msgs:
                    if m["id"] > self.state["seen"]:
                        self.state["seen"] = m["id"]
                        self.on_message(m, now)
                self.tick(now)
            except Exception:
                log.exception("cycle failed")
            if once:
                return
            time.sleep(int(self.t.poll_interval_sec))


def build(cfg):
    client = MexcClient(cfg.mexc)
    t = cfg.telegram_signals
    sp = Path(t.state_file)
    state = json.loads(sp.read_text()) if sp.exists() else {}
    if cfg.trading.dry_run:
        return SignalTrader(cfg, PaperExchange(cfg, client, state), Notifier(cfg), state, t.state_file, t.log_file)
    if not (client.api_key and client.api_secret):
        raise SystemExit("التنفيذ الحقيقي يحتاج مفاتيح MEXC في متغيرات البيئة (انظر docs/WINDOWS_TELEGRAM.md).")
    from .live import LiveExchange
    nt = Notifier(cfg)
    nt.tag = "[حقيقي] "
    return SignalTrader(cfg, LiveExchange(cfg, client, state), nt, state, t.state_file, t.log_file)
