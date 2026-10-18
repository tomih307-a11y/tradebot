"""Telegram signals module: parser (real channel messages), paper simulation rules, trader flow with a fake client."""
import json

import pytest

from conftest import ovr
from tradebot.strategies.telegram_signals import paper
from tradebot.strategies.telegram_signals.parser import Signal, parse_signal, parse_update
from tradebot.strategies.telegram_signals.source import load_export, parse_preview_html
from tradebot.strategies.telegram_signals.trader import PaperExchange, SignalTrader

Q_SIGNAL = """🚨 فرصة جديدة | CRYPTO

🪙 العملة: Q / USDT
📍 الاتجاه: LONG 🟢
💼 السوق: سبوت + فيوتشرز 💎
⚡ نوع الصفقة: سكالب
⏱ الفريم: 5M
⚙️ الرافعة: 10X

🎯 منطقة الدخول: 0.070963
🛑 وقف الخسارة: 0.069189 (-2.5%)
📌 إلغاء السيناريو: إغلاق شمعة 4H تحت 0.069189

الأهداف:
✅ TP1: 0.072737 (+2.5%) | ROI +25%
✅ TP2: 0.074511 (+5%) | ROI +50%
✅ TP3: 0.076285 (+7.5%) | ROI +75%
✅ TP4: 0.078059 (+10%) | ROI +100%

الحالة: مغلقة
"""
BAT_SIGNAL = Q_SIGNAL.replace("Q / USDT", "BAT / USDT").replace("0.070963", "0.18156").replace("0.069189", "0.17611") \
    .replace("0.072737", "0.1861").replace("0.074511", "0.19064").replace("0.076285", "0.19518").replace("0.078059", "0.19972")
BE_UPDATE = """🛡 تأمين الصفقة — نقطة التعادل

SAND — الوقف انتقل إلى 0.058450
الصفقة الآن بلا مخاطرة ✅"""
TP_REPORT = """🏆 حصاد النخبة | تحقق الهدف الأول
SAND • LONG • سبوت + فيوتشرز

✅ TP1 — 0.06031 (+3.18%)"""


def test_parse_real_signal():
    s = parse_signal(Q_SIGNAL, 7, 1000)
    assert (s.symbol, s.side, s.entry, s.stop, s.leverage) == ("Q_USDT", "long", 0.070963, 0.069189, 10)
    assert s.tps == [0.072737, 0.074511, 0.076285, 0.078059]


def test_non_signals_ignored():
    for t in (TP_REPORT, BE_UPDATE, "وين صور الربح ؟", "🔥👌🏻", "🏆 تقرير النخبة اليومي ✅ SIREN · BUY 🔼"):
        assert parse_signal(t) is None


def test_inconsistent_signal_rejected():
    bad = Q_SIGNAL.replace("وقف الخسارة: 0.069189", "وقف الخسارة: 0.075000")        # stop above entry on a long
    assert parse_signal(bad) is None
    assert parse_signal(Q_SIGNAL.replace("TP2: 0.074511", "TP2: 0.071000")) is None   # targets not ascending


def test_short_signal_parsed():
    t = (Q_SIGNAL.replace("LONG 🟢", "SHORT 🔴").replace("0.069189", "0.072737").replace("0.072737 (+2.5%)", "0.068000 (-2.5%)")
         .replace("0.074511", "0.066000").replace("0.076285", "0.064000").replace("0.078059", "0.062000"))
    t = t.replace("وقف الخسارة: 0.072737 (-2.5%)", "وقف الخسارة: 0.072737 (+2.5%)")
    s = parse_signal(t)
    assert s and s.side == "short" and s.stop > s.entry > s.tps[0] > s.tps[1]


def test_update_parsed():
    u = parse_update(BE_UPDATE, 9, 5)
    assert (u.symbol, u.new_stop) == ("SAND_USDT", 0.05845)
    assert parse_update(Q_SIGNAL) is None


def test_preview_html_and_export(tmp_path):
    page = ('<div class="tgme_widget_message_wrap js-widget_message_wrap"><div class="tgme_widget_message" data-post="chan/12">'
            '<div class="tgme_widget_message_text js-message_text" dir="auto">🪙 العملة: Q / USDT<br/>الاتجاه: LONG</div>'
            '<a class="tgme_widget_message_date"><time datetime="2026-10-17T11:22:00+00:00" class="time">14:22</time></a></div></div>')
    m = parse_preview_html(page)
    assert m == [{"id": 12, "ts": 1792236120, "text": "🪙 العملة: Q / USDT\nالاتجاه: LONG"}]
    p = tmp_path / "result.json"
    p.write_text(json.dumps({"messages": [{"id": 5, "type": "message", "date": "2026-10-17T11:22:00", "date_unixtime": "1792236120",
                                           "text": ["🪙 العملة: ", {"type": "bold", "text": "Q / USDT"}]}]}))
    assert load_export(str(p))[0]["text"] == "🪙 العملة: Q / USDT"


# ───────────── paper simulation ─────────────
CFG = {"fee_pct": 0.0, "slip_pct": 0.0, "breakeven_after_tp1": True}


def mk(side="long", lev=10, stop_mode="channel_4h_close", entry=100.0, tps=(102.5, 105.0, 107.5, 110.0), stop=97.5):
    sg = Signal(1, 0, "X_USDT", side, entry, stop, list(tps), lev)
    return paper.open_position(sg, entry, 25.0, 10.0, lev, 0, stop_mode, 0.0, 0.005)


def candle(i, o, h, l, c):
    return (60 * i, o, h, l, c)


def test_margin_is_10_percent_of_balance():
    p = mk()
    assert p.margin == pytest.approx(2.5) and p.notional == pytest.approx(25.0)


def test_all_targets_equal_shares_and_breakeven():
    p = mk()
    for i, (h, l) in enumerate([(102.6, 100), (105.1, 101), (107.6, 103), (110.1, 105)], 1):
        paper.step(p, candle(i, 100, h, l, 101), CFG)
    assert p.closed and p.exit_reason == "all_tp" and p.tp_hit == 4
    assert p.stop == 100.0                                           # moved to entry after TP1
    # 25$ notional: each quarter earns 6.25 * (2.5%,5%,7.5%,10%) = 0.15625+0.3125+0.46875+0.625
    assert p.realized == pytest.approx(1.5625)


def test_liquidation_before_4h_stop_loses_margin():
    p = mk(lev=10)                                                   # liquidation at ~ -9.5%
    assert p.liq_price == pytest.approx(100 * (1 - (0.1 - 0.005)))
    paper.step(p, candle(1, 100, 100, 90.0, 91.0), CFG)
    assert p.exit_reason == "liquidated" and p.realized == pytest.approx(-p.margin)


def test_channel_rule_ignores_wick_below_stop_but_exits_on_4h_close():
    p = mk(lev=3)                                                    # liq far away, ~ -32%
    paper.step(p, candle(1, 100, 100, 96.0, 99.0), CFG)              # wick through the stop: ignored
    assert not p.closed
    paper.step(p, candle(239, 99, 99, 96, 97.0), CFG)                # last minute of the 4H window, closes below stop
    assert p.closed and p.exit_reason == "stop_4h_close"
    assert p.realized == pytest.approx(-0.03 * p.notional)           # -3% of the 7.5$ notional (2.5$ margin x 3)


def test_exchange_stop_mode_exits_at_stop_and_gaps_at_open():
    p = mk(lev=3, stop_mode="exchange")
    paper.step(p, candle(1, 100, 100, 97.0, 98.0), CFG)
    assert p.exit_reason == "stop" and p.events[-1][2] == pytest.approx(97.5)
    q = mk(lev=3, stop_mode="exchange")
    paper.step(q, candle(1, 95.0, 96.0, 94.0, 95.0), CFG)            # opens beyond the stop
    assert q.events[-1][2] == pytest.approx(95.0)


def test_adverse_first_inside_one_candle():
    p = mk(lev=3, stop_mode="exchange")
    paper.step(p, candle(1, 100, 106.0, 97.0, 100.0), CFG)           # touches stop AND targets in one minute
    assert p.exit_reason == "stop" and p.tp_hit == 0


def test_short_mirror():
    p = mk(side="short", entry=100.0, tps=(97.5, 95.0, 92.5, 90.0), stop=102.5)
    for i, (h, l) in enumerate([(100, 97.4), (99, 94.9), (97, 92.4), (95, 89.9)], 1):
        paper.step(p, candle(i, 100, h, l, 98), CFG)
    assert p.exit_reason == "all_tp" and p.realized == pytest.approx(1.5625)


# ───────────── trader flow with a fake client ─────────────
class FakeClient:
    def __init__(self, price=0.0709):
        self.price, self.candles = price, []

    def contract_detail(self, sym):
        return {"maxLeverage": 20} if sym != "NOPE_USDT" else None

    def last_price(self, sym):
        return self.price

    def klines(self, sym, tf, a, b):
        c = [x for x in self.candles if a <= x[0] <= b]
        return {"time": [x[0] for x in c], "open": [x[1] for x in c], "high": [x[2] for x in c], "low": [x[3] for x in c], "close": [x[4] for x in c]}


class Quiet:
    def __init__(self):
        self.log = []

    def send(self, t):
        self.log.append(t)


def trader(cfg, tmp_path, client):
    state = {}
    ex = PaperExchange(cfg, client, state)
    nt = Quiet()
    return SignalTrader(cfg, ex, nt, state, str(tmp_path / "st.json"), str(tmp_path / "log.csv")), nt, client


def test_two_slot_limit_and_ten_percent_sizing(cfg, tmp_path):
    tr, nt, cl = trader(cfg, tmp_path, FakeClient(0.070963))
    s1 = parse_signal(Q_SIGNAL, 1, 1000)
    assert tr.try_open(s1, 1005) == "opened"
    pos = tr.open["Q_USDT"]
    assert pos.margin == pytest.approx(2.5) and pos.leverage == 10
    cl.price = 0.18156
    bal_before_second = tr.ex.balance()
    assert tr.try_open(parse_signal(BAT_SIGNAL, 2, 1000), 1010) == "opened"
    assert tr.open["BAT_USDT"].margin == pytest.approx(bal_before_second * 0.10)
    third = Signal(3, 1000, "ETH_USDT", "long", 0.18156, 0.17, [0.19], 5)
    assert tr.try_open(third, 1020).startswith("skip") and len(tr.open) == 2


def test_stale_ranaway_and_unknown_symbol_are_skipped(cfg, tmp_path):
    tr, nt, cl = trader(cfg, tmp_path, FakeClient(0.0800))                  # price ran away (+12%)
    assert tr.try_open(parse_signal(Q_SIGNAL, 1, 1000), 1005).startswith("skip")
    cl.price = 0.070963
    assert tr.try_open(parse_signal(Q_SIGNAL, 1, 1000), 1000 + 10_000).startswith("skip")   # too old
    unk = Signal(4, 1000, "NOPE_USDT", "long", 0.0709, 0.069, [0.073], 5)
    assert tr.try_open(unk, 1005).startswith("skip") and "غير متاح" in nt.log[-1]
    assert not tr.open


def test_tick_closes_on_targets_and_updates_balance(cfg, tmp_path):
    tr, nt, cl = trader(cfg, tmp_path, FakeClient(0.070963))
    tr.try_open(parse_signal(Q_SIGNAL, 1, 1000), 1000)
    bal0 = tr.ex.balance()
    cl.candles = [(1080 + 60 * i, 0.0710, h, 0.0709, 0.0711) for i, h in enumerate([0.0728, 0.0746, 0.0763, 0.0781])]
    tr.tick(1080 + 60 * 5)
    assert not tr.open and tr.ex.balance() > bal0
    assert (tmp_path / "log.csv").exists() and "all_tp" in (tmp_path / "log.csv").read_text()


def test_update_only_tightens(cfg, tmp_path):
    tr, nt, cl = trader(cfg, tmp_path, FakeClient(0.070963))
    tr.try_open(parse_signal(Q_SIGNAL, 1, 1000), 1000)
    old = tr.open["Q_USDT"].stop
    tr.on_message({"id": 9, "ts": 1100, "text": "Q — الوقف انتقل إلى 0.0600"}, 1100)       # looser: ignored
    assert tr.open["Q_USDT"].stop == old
    tr.on_message({"id": 10, "ts": 1100, "text": "Q — الوقف انتقل إلى 0.0709"}, 1100)      # tighter: applied
    assert tr.open["Q_USDT"].stop == pytest.approx(0.0709)
