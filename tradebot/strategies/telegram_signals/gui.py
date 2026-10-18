"""Desktop window for the Telegram-signals module (paper mode). Logic lives in GuiController (no tkinter), so it is testable."""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from .source import PublicChannelReader
from .trader import Notifier, build

SETTINGS = Path("state/gui_settings.json")


class QueueNotifier(Notifier):
    def __init__(self, cfg, q: queue.Queue):
        super().__init__(cfg)
        self.q = q

    def send(self, text: str) -> None:
        self.q.put(("log", time.strftime("%H:%M:%S") + "  " + text))
        super().send(text)


class GuiController:
    def __init__(self, cfg, channel: str = ""):
        self.cfg, self.channel = cfg, channel.lstrip("@").strip()
        self.q: queue.Queue = queue.Queue()
        self.trader = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.running:
            return
        if not self.channel:
            raise ValueError("اكتب اسم القناة أولاً")
        self.trader = build(self.cfg)                      # refuses unless dry_run = true (paper)
        self.trader.nt = QueueNotifier(self.cfg, self.q)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        reader = PublicChannelReader(self.channel)
        poll = max(1, int(self.cfg.telegram_signals.poll_interval_sec))
        while not self._stop.is_set():
            self.trader.run(reader, once=True)
            self._stop.wait(poll)
        self.q.put(("log", time.strftime("%H:%M:%S") + "  ⏹ تم الإيقاف"))

    def snapshot(self) -> dict:
        t = self.trader
        if not t:
            return {"balance": float(self.cfg.telegram_signals.paper_balance_usdt), "positions": []}
        rows = []
        for p in t.open.values():
            try:
                px = t.ex.last_price(p.symbol)
            except Exception:
                px = float("nan")
            upnl = p.remaining * p.notional * p.s * (px - p.entry) / p.entry
            rows.append({"symbol": p.symbol, "side": p.side, "entry": p.entry, "price": px, "lev": p.leverage,
                         "margin": p.margin, "stop": p.stop, "tp_hit": p.tp_hit, "tps": len(p.tps),
                         "pnl": p.realized + upnl})
        return {"balance": t.ex.balance(), "positions": rows}


def load_channel() -> str:
    try:
        return json.loads(SETTINGS.read_text()).get("channel", "")
    except Exception:
        return ""


def save_channel(ch: str) -> None:
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps({"channel": ch}))


def main(cfg) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    ctl = GuiController(cfg, load_channel() or cfg.telegram_signals.channel)
    root = tk.Tk()
    root.title("TradeBot — توصيات تلقرام (وضع ورقي)")
    root.geometry("900x560")

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Label(top, text="القناة (بعد t.me/):").pack(side="left")
    ch = tk.StringVar(value=ctl.channel)
    ttk.Entry(top, textvariable=ch, width=26).pack(side="left", padx=6)
    status = tk.StringVar(value="متوقف")
    bal = tk.StringVar()
    btn = ttk.Button(top, text="ابدأ")
    btn.pack(side="left", padx=6)
    ttk.Label(top, textvariable=status).pack(side="left", padx=10)
    ttk.Label(top, textvariable=bal, font=("", 11, "bold")).pack(side="right")

    t = cfg.telegram_signals
    ttk.Label(root, padding=(8, 0), text=f"الحدود: هامش {t.margin_pct_of_balance}% لكل صفقة · أقصى {t.max_concurrent} صفقات متزامنة · "
                                         f"الأهداف والوقف كما في التوصية · محاكاة بدون أوامر حقيقية").pack(anchor="w")

    cols = ("symbol", "side", "entry", "price", "lev", "margin", "stop", "tp", "pnl")
    names = ("العملة", "الاتجاه", "الدخول", "السعر", "رافعة", "الهامش $", "الوقف", "الأهداف", "الربح $")
    tree = ttk.Treeview(root, columns=cols, show="headings", height=6)
    for c, n in zip(cols, names):
        tree.heading(c, text=n)
        tree.column(c, width=90, anchor="center")
    tree.pack(fill="x", padx=8, pady=8)

    logbox = tk.Text(root, height=18, state="disabled", wrap="word")
    logbox.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def toggle():
        if ctl.running:
            ctl.stop()
            btn.config(text="ابدأ")
            status.set("جارٍ الإيقاف…")
            return
        ctl.channel = ch.get().lstrip("@").strip()
        try:
            ctl.start()
        except (ValueError, SystemExit) as e:
            messagebox.showerror("خطأ", str(e))
            return
        save_channel(ctl.channel)
        btn.config(text="إيقاف")
        status.set("يعمل — يراقب " + ctl.channel)

    btn.config(command=toggle)

    def tick():
        try:
            while True:
                kind, text = ctl.q.get_nowait()
                logbox.config(state="normal")
                logbox.insert("end", text + "\n")
                logbox.see("end")
                logbox.config(state="disabled")
        except queue.Empty:
            pass
        if not ctl.running and btn.cget("text") == "إيقاف":
            btn.config(text="ابدأ")
            status.set("متوقف")
        snap = ctl.snapshot()
        bal.set(f"الرصيد: {snap['balance']:.2f}$")
        tree.delete(*tree.get_children())
        for r in snap["positions"]:
            tree.insert("", "end", values=(r["symbol"], "لونق" if r["side"] == "long" else "شورت", f"{r['entry']:.6g}",
                                           f"{r['price']:.6g}", f"{r['lev']}x", f"{r['margin']:.2f}", f"{r['stop']:.6g}",
                                           f"{r['tp_hit']}/{r['tps']}", f"{r['pnl']:+.3f}"))
        root.after(3000, tick)

    root.after(500, tick)
    root.mainloop()
