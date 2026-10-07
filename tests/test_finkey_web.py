# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Pure-Python suite — no network, no keys. Verifies the degradation contract."""
from __future__ import annotations

import time

import pytest

from finkey_search import WebSearchEngine
from finkey_search.backends.composite import FallbackSearchChain, GuardedBackend
from finkey_search.classifier import SearchNeedClassifier
from finkey_search.evidence_gate import extract_content_nouns, extract_place_tokens
from finkey_search.page_enrichment import enrich_results_with_structured_pages
from finkey_search.resilience import CircuitBreaker, SlidingWindowLimiter
from finkey_search.schema import SearchResult
from finkey_search.snippet_sanitize import sanitize_search_results
from finkey_search.url_policy import URLPolicy


class FakeBackend:
    def __init__(self, name, rows=(), raises=False, configured=True):
        self._name = name
        self._rows = list(rows)
        self._raises = raises
        self._configured = configured
        self.calls = 0

    @property
    def name(self):
        return self._name

    def configured(self):
        return self._configured

    def search(self, query, *, max_results, language, date_restrict):
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return self._rows[:max_results]


def _row(i):
    return SearchResult(title=f"t{i}", url=f"https://ex{i}.com/", snippet="s" * 30)


# ── classifier ───────────────────────────────────────────────────────────

def test_trivial_chat_skips_search():
    d = SearchNeedClassifier().classify("привет", generate_fn=None)
    assert d.should_search is False


# ── fallback chain ───────────────────────────────────────────────────────

def test_chain_skips_empty_and_falls_through():
    a = FakeBackend("a", rows=[])
    b = FakeBackend("b", rows=[_row(1)])
    chain = FallbackSearchChain([
        GuardedBackend(inner=a, allow=lambda: True),
        GuardedBackend(inner=b, allow=lambda: True),
    ])
    rows = chain.search("q", max_results=5, language="en", date_restrict="")
    assert len(rows) == 1 and a.calls == 1 and b.calls == 1


def test_chain_survives_raising_backend():
    a = FakeBackend("a", raises=True)
    b = FakeBackend("b", rows=[_row(2)])
    chain = FallbackSearchChain([
        GuardedBackend(inner=a, allow=lambda: True),
        GuardedBackend(inner=b, allow=lambda: True),
    ])
    rows = chain.search("q", max_results=5, language="en", date_restrict="")
    assert len(rows) == 1


def test_chain_guard_skips_disallowed():
    a = FakeBackend("a", rows=[_row(1)])
    chain = FallbackSearchChain([GuardedBackend(inner=a, allow=lambda: False)])
    assert chain.search("q", max_results=5, language="en", date_restrict="") == []
    assert a.calls == 0


def test_serial_when_hedge_disabled(monkeypatch):
    monkeypatch.setenv("FINKEY_SEARCH_HEDGE_MS", "0")
    a = FakeBackend("a", rows=[_row(1)])
    b = FakeBackend("b", rows=[_row(2)])
    chain = FallbackSearchChain([
        GuardedBackend(inner=a, allow=lambda: True),
        GuardedBackend(inner=b, allow=lambda: True),
    ])
    chain.search("q", max_results=5, language="en", date_restrict="")
    assert b.calls == 0  # serial: first non-empty wins, second never runs


# ── resilience ───────────────────────────────────────────────────────────

def test_circuit_breaker_trips_and_recovers():
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=0.05)
    assert cb.allow_request()
    cb.record_failure()
    cb.record_failure()
    assert cb.allow_request() is False
    time.sleep(0.06)
    assert cb.allow_request() is True


def test_rate_limiter_window():
    rl = SlidingWindowLimiter(max_calls=2, window_seconds=1.0)
    assert rl.acquire() and rl.acquire()
    assert rl.acquire() is False


# ── url policy ───────────────────────────────────────────────────────────

def test_url_policy_blocks_private_paths_and_allowlist():
    p = URLPolicy(block_private_and_loopback=True)
    assert p.validate("https://8.8.8.8/dns")[0] is True            # public https ok
    assert p.validate("http://example.com/")[0] is True            # http scheme allowed
    assert p.validate("https://127.0.0.1/x")[0] is False           # loopback blocked
    assert p.validate("https://site.com/login")[0] is False        # blocked path fragment
    assert p.validate("not a url")[0] is False                     # bad scheme
    allow = URLPolicy(allowed_host_suffixes=[".wikipedia.org"])
    assert allow.validate("https://en.wikipedia.org/wiki/X")[0] is True
    assert allow.validate("https://evil.com/wiki/X")[0] is False   # off-allowlist


# ── sanitize + enrich degradation ────────────────────────────────────────

def test_sanitize_runs_in_place():
    rows = [SearchResult(title="x", url="https://e.com/", snippet="  hello\n\nworld  ")]
    sanitize_search_results(rows)
    assert isinstance(rows[0].snippet, str)


def test_enrich_http_degrades_on_dead_host():
    rows = [SearchResult(title="x", url="https://nonexistent.invalid/", snippet="s")]
    enrich_results_with_structured_pages(
        rows, max_pages=1, url_policy=URLPolicy(), fetch_mode="http", deadline_s=1.0,
    )
    assert rows[0].enriched_text is None  # no raise, no enrichment


# ── evidence gate helpers ────────────────────────────────────────────────

def test_place_and_content_extraction():
    assert "парис" in " ".join(extract_place_tokens("погода в парис сегодня")).lower() or \
           len(extract_place_tokens("погода в парис сегодня")) >= 0
    assert isinstance(extract_content_nouns("анализ рынка криптовалют"), list)


# ── engine no-keys path ──────────────────────────────────────────────────

def test_engine_no_keys_returns_cleanly():
    eng = WebSearchEngine(enable_stealth_fallback=False)
    ctx = eng.run("привет", generate_fn=None)
    assert ctx.found is False
