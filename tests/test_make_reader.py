import pytest

from tradebot.strategies.telegram_signals.source import PublicChannelReader, make_reader


def test_username_uses_public_reader():
    assert isinstance(make_reader("@some_channel"), PublicChannelReader)


def test_numeric_id_needs_api_keys(monkeypatch):
    monkeypatch.delenv("TG_API_ID", raising=False)
    monkeypatch.delenv("TG_API_HASH", raising=False)
    with pytest.raises(SystemExit):
        make_reader("-1002639765041")
