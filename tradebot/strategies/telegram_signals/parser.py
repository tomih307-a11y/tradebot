"""Parse the channel's Arabic signal / update messages into data. Pure functions, no network."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

_NUM = r"([0-9]+(?:[.,][0-9]+)?)"
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫", "0123456789.")


def _f(x: str) -> float:
    return float(x.replace(",", "."))


@dataclass
class Signal:
    msg_id: int
    ts: int                        # message time (epoch s, UTC); 0 if unknown
    symbol: str                    # MEXC contract symbol, e.g. "Q_USDT"
    side: str                      # "long" | "short"
    entry: float
    stop: float
    tps: list[float]
    leverage: int | None = None
    raw: str = field(default="", repr=False)


@dataclass
class StopUpdate:
    msg_id: int
    ts: int
    symbol: str
    new_stop: float


def normalize(text: str) -> str:
    return html.unescape(text).translate(_AR_DIGITS)


def parse_signal(text: str, msg_id: int = 0, ts: int = 0) -> Signal | None:
    """Returns a Signal for a 'new opportunity' message, else None. Rejects internally inconsistent signals."""
    t = normalize(text)
    if "فرصة جديدة" not in t:
        return None
    m_sym = re.search(r"العملة\s*[:：]\s*([A-Za-z0-9]+)\s*/\s*USDT", t)
    m_side = re.search(r"الاتجاه\s*[:：]\s*(LONG|SHORT|BUY|SELL|شراء|بيع)", t, re.I)
    m_entry = re.search(r"منطقة الدخول\s*[:：]\s*" + _NUM, t)
    m_stop = re.search(r"وقف الخسارة\s*[:：]\s*" + _NUM, t)
    tps = re.findall(r"TP(\d+)\s*[:：]\s*" + _NUM, t)
    if not (m_sym and m_side and m_entry and m_stop and tps):
        return None
    side_word = m_side.group(1).upper()
    side = "long" if side_word in ("LONG", "BUY", "شراء") else "short"
    entry, stop = _f(m_entry.group(1)), _f(m_stop.group(1))
    levels = [_f(p) for _, p in sorted(tps, key=lambda x: int(x[0]))]
    s = 1 if side == "long" else -1
    # sanity: stop on the losing side, targets ascending in the trade direction
    if not (s * (stop - entry) < 0 and all(s * (p - entry) > 0 for p in levels)
            and all(s * (b - a) > 0 for a, b in zip(levels, levels[1:]))):
        return None
    m_lev = re.search(r"الرافعة\s*[:：]\s*(\d+)\s*[xX]", t)
    return Signal(msg_id=msg_id, ts=ts, symbol=f"{m_sym.group(1).upper()}_USDT", side=side, entry=entry,
                  stop=stop, tps=levels, leverage=int(m_lev.group(1)) if m_lev else None, raw=text)


def parse_update(text: str, msg_id: int = 0, ts: int = 0) -> StopUpdate | None:
    """'SAND — الوقف انتقل إلى 0.058450' (break-even notice) -> StopUpdate."""
    t = normalize(text)
    m = re.search(r"([A-Za-z0-9]+)\s*[—\-–]\s*الوقف انتقل إلى\s*" + _NUM, t)
    if not m:
        return None
    return StopUpdate(msg_id=msg_id, ts=ts, symbol=f"{m.group(1).upper()}_USDT", new_stop=_f(m.group(2)))
