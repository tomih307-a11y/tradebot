"""Alert formatting (Arabic) + console/Telegram delivery."""
from __future__ import annotations

import json
import logging
import math
import os
import urllib.parse
import urllib.request

from . import model as M

log = logging.getLogger("pmo.alerts")

_AR_POS = {"above": "فوق الصفر", "below": "تحت الصفر"}
_AR_SLOPE = {"with": "مع الاتجاه", "against": "ضد الاتجاه"}


def _eta(x):
    return "لا نهائي" if math.isinf(x) else f"{x:.1f} شمعة ساعة"


def _frame(label, d):
    return (f"{label}: {_AR_POS[d['pos']]} · ميل {_AR_SLOPE[d['slope_dir']]} · PMO {d['pmo']:.3f} "
            f"· SIGNAL {d['signal']:.3f} · SLOPE {d['slope']:.3f}")


class Notifier:
    def __init__(self, cfg, cells: dict | None = None):
        live = cfg.pmo_zero_cross.live
        self.token = os.environ.get(live.telegram_token_env, "")
        self.chat = os.environ.get(live.telegram_chat_id_env, "")
        self.cells = cells or {}
        self.min_sample = cfg.pmo_zero_cross.backtest.min_sample
        self.dry = cfg.trading.dry_run

    def send(self, text: str) -> None:
        prefix = "[DRY_RUN] " if self.dry else ""
        log.info(prefix + text.replace("\n", " | "))
        if self.token and self.chat:
            try:
                data = urllib.parse.urlencode({"chat_id": self.chat, "text": prefix + text}).encode()
                urllib.request.urlopen(
                    urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendMessage", data=data), timeout=10)
            except Exception as e:                      # alerts must never break trading
                log.warning("telegram send failed: %s", e)

    def hist_rate(self, tr) -> str:
        """Historical 1H-cross rate of the same state cell: level 2 (slope+position) if it has enough
        samples, otherwise level 1 (slope only)."""
        sn = tr.snap
        k1 = sn["slope_class"]
        k2 = f"{k1}|{sn['h1']['pos']}|{sn['h4']['pos']}"
        for label, cell in (("ميل+موقع", self.cells.get("sub", {}).get(k2)), ("ميل", self.cells.get("main", {}).get(k1))):
            if cell and cell.get("n") and (cell["n"] >= self.min_sample or label == "ميل"):
                warn = " (عينة غير كافية)" if cell["n"] < self.min_sample else ""
                return f"{cell['h1_rate'] * 100:.1f}% من {cell['n']} صفقة — خلية {label}{warn}"
        return "غير متوفرة (لا يوجد باكتست)"

    @staticmethod
    def _deadline_line(tr) -> str:
        if tr.wait_deadline is None:
            return "المهلة: لا تنطبق (الفريم عبر الصفر)"
        bars = tr.wait_deadline - tr.entry_ts // 900
        return f"المهلة المحسوبة: {bars} شمعة 15د (~{bars / 4:.1f} ساعة)"

    def handle(self, kind: str, tr, info: dict) -> None:
        if kind == "entry":
            d = "لونق 🟢" if tr.side == "long" else "شورت 🔴"
            self.send("\n".join([
                f"دخول {d} — {tr.symbol}", f"السعر: {tr.entry_price:.6g}",
                f"المرحلة الابتدائية: {M.STAGE_AR[tr.initial_stage]} ({tr.initial_stage})",
                _frame("الساعة", tr.snap["h1"]), _frame("الـ4 ساعات", tr.snap["h4"]),
                f"ETA_1H: {_eta(tr.snap['h1']['eta'])}", self._deadline_line(tr),
                f"الوقف الابتدائي: {tr.initial_stop:.6g}",
                f"نسبة عبور الساعة التاريخية لنفس الخلية: {self.hist_rate(tr)}"]))
        elif kind == "stage_1h":
            self.send(f"✅ الساعة عبرت الصفر — {tr.symbol} ({tr.side}) @ {info['price']:.6g} → الانتظار للـ4 ساعات أو RIDE")
        elif kind == "stage_4h":
            self.send(f"✅ الـ4 ساعات عبرت الصفر — {tr.symbol} ({tr.side}) @ {info['price']:.6g} → RIDE")
        elif kind == "stop_move":
            self.send(f"🔧 تحريك الوقف — {tr.symbol}: {info['old']:.6g} → {info['new']:.6g} (مرحلة {tr.stop_stage})")
        elif kind == "exit":
            self.send(f"⛔ خروج — {tr.symbol} ({tr.side}) السبب: {M.REASON_AR.get(info['reason'], info['reason'])} "
                      f"· السعر {info['price']:.6g}" + (f" · صافي {tr.net_ret_pct:.2f}%" if tr.net_ret_pct is not None else ""))
        elif kind == "reject":
            if info["reason"] == "max_open_full":
                self.send(f"⚠️ إشارة مرفوضة — {info['symbol']} ({info['side']}): الحد الأقصى ممتلئ")
