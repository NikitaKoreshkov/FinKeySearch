"""Serper key-pool rotation on credits/auth exhaustion."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from finkey_search.backends.serper import (
    SerperSearchBackend,
    reset_serper_key_pool_for_tests,
)


def test_serper_rotates_on_not_enough_credits(monkeypatch):
    reset_serper_key_pool_for_tests()
    monkeypatch.setenv("SERPER_API_KEY", "deadkey11111111111111111111111111111111")
    monkeypatch.setenv(
        "SERPER_API_KEYS",
        "deadkey11111111111111111111111111111111,livekey22222222222222222222222222222222",
    )
    monkeypatch.setenv("SERPER_MAX_RETRIES", "0")

    be = SerperSearchBackend(endpoint="https://example.invalid/search")

    class ECredits(Exception):
        response = MagicMock(status_code=400, text='{"message":"Not enough credits"}')

    good = {
        "organic": [
            {
                "link": "https://example.com/a",
                "title": "ok",
                "snippet": "snippet text long enough for filter",
            }
        ]
    }

    def _once(url, payload, *, api_key=""):
        if api_key.startswith("dead"):
            raise ECredits("400 Not enough credits")
        return good

    with patch.object(be, "_post_json_once", side_effect=_once) as mocked:
        out = be.search("q", max_results=5, language="lang_en", date_restrict="none")
    assert len(out) == 1
    assert mocked.call_count == 2
    assert be._key.startswith("live")


def test_serper_retries_on_429(monkeypatch):
    reset_serper_key_pool_for_tests()
    monkeypatch.setenv("SERPER_MAX_RETRIES", "2")
    monkeypatch.setenv("SERPER_RETRY_BASE_SEC", "0.01")
    monkeypatch.delenv("SERPER_API_KEYS", raising=False)

    be = SerperSearchBackend(api_key="k", endpoint="https://example.invalid/search")

    class E429(Exception):
        response = MagicMock(status_code=429)

    side = [
        E429(),
        E429(),
        {
            "organic": [
                {"link": "https://a.ru/x", "title": "t", "snippet": "snippet text long enough"}
            ]
        },
    ]

    with patch.object(be, "_post_json_once", side_effect=side) as mocked:
        out = be.search(
            "q",
            max_results=5,
            language="lang_en",
            date_restrict="none",
        )
    assert len(out) == 1
    assert mocked.call_count == 3
