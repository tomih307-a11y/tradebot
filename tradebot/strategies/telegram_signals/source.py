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
