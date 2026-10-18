"""Read a PUBLIC Telegram channel from its public web preview (https://t.me/s/<channel>).
No login, no account access, read-only. Also loads a Telegram Desktop 'Export chat history' JSON."""
from __future__ import annotations

import html
import json
import re
import urllib.request
from datetime import datetime, timezone

from .parser import normalize


def parse_preview_html(page: str) -> list[dict]:
    """-> [{id, ts, text}] oldest first."""
    out = []
    for block in re.split(r'<div class="tgme_widget_message_wrap', page)[1:]:
        m_id = re.search(r'data-post="[^"/]+/(\d+)"', block)
        if not m_id:
            continue
        m_txt = re.search(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', block, re.S)
        m_time = re.search(r'<time[^>]*datetime="([^"]+)"', block)
        text = ""
        if m_txt:
            text = re.sub(r"<br\s*/?>", "\n", m_txt.group(1))
            text = html.unescape(re.sub(r"<[^>]+>", "", text))
        ts = 0
        if m_time:
            try:
                ts = int(datetime.fromisoformat(m_time.group(1)).astimezone(timezone.utc).timestamp())
            except ValueError:
                pass
        out.append({"id": int(m_id.group(1)), "ts": ts, "text": text})
    return sorted(out, key=lambda x: x["id"])


class PublicChannelReader:
    def __init__(self, channel: str, timeout: int = 15):
        self.channel, self.timeout = channel.lstrip("@"), timeout

    def fetch(self) -> list[dict]:
        req = urllib.request.Request(f"https://t.me/s/{self.channel}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return parse_preview_html(r.read().decode("utf-8", "replace"))


def load_export(path: str) -> list[dict]:
    """Telegram Desktop export (result.json) -> [{id, ts, text}] oldest first. `text` may be a list of fragments."""
    data = json.load(open(path, encoding="utf-8"))
    msgs = data["messages"] if isinstance(data, dict) else data
    out = []
    for m in msgs:
        if m.get("type", "message") != "message":
            continue
        t = m.get("text", "")
        if isinstance(t, list):
            t = "".join(x if isinstance(x, str) else x.get("text", "") for x in t)
        ts = int(m["date_unixtime"]) if m.get("date_unixtime") else int(
            datetime.fromisoformat(m["date"]).replace(tzinfo=timezone.utc).timestamp()) if m.get("date") else 0
        out.append({"id": int(m["id"]), "ts": ts, "text": normalize(t)})
    return sorted(out, key=lambda x: x["id"])


class TelethonReader:
    """Read a PRIVATE channel with the user's own Telegram account (Telethon). Read-only; the session file stays on the user's PC.
    channel: numeric id (e.g. -1002639765041) or username. Login once with:  python -m tradebot tg-login"""

    def __init__(self, channel: str, api_id: int, api_hash: str, session: str = "state/tg_session", limit: int = 60):
        self.channel, self.api_id, self.api_hash, self.session, self.limit = channel, api_id, api_hash, session, limit
        self._client = None
        self._entity = None

    def _connect(self):
        import asyncio

        from telethon import TelegramClient
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        c = TelegramClient(self.session, self.api_id, self.api_hash)
        c.connect()
        if not c.is_user_authorized():
            raise RuntimeError("حساب تلقرام غير مسجَّل دخوله — شغّل أولاً: python -m tradebot tg-login")
        c.get_dialogs(limit=300)                     # fills the entity cache so a numeric id resolves
        ch = self.channel.strip()
        self._entity = c.get_entity(int(ch) if ch.lstrip("-").isdigit() else ch)
        self._client = c

    def fetch(self) -> list[dict]:
        if self._client is None:
            self._connect()
        out = []
        for m in self._client.get_messages(self._entity, limit=self.limit):
            if m.id and m.date:
                out.append({"id": int(m.id), "ts": int(m.date.timestamp()), "text": normalize(m.message or "")})
        return sorted(out, key=lambda x: x["id"])


def make_reader(channel: str):
    """Public username -> web preview reader. Numeric id / private channel -> Telethon (needs TG_API_ID, TG_API_HASH)."""
    import os
    ch = channel.strip()
    if ch.lstrip("-").isdigit():
        aid, ah = os.environ.get("TG_API_ID", ""), os.environ.get("TG_API_HASH", "")
        if not (aid and ah):
            raise SystemExit("القناة خاصة: ضع TG_API_ID و TG_API_HASH (من my.telegram.org) في متغيرات البيئة، ثم python -m tradebot tg-login")
        return TelethonReader(ch, int(aid), ah)
    return PublicChannelReader(ch)
