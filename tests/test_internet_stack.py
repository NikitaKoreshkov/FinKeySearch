"""Offline tests for internet retrieval stack (no live HTTP)."""
from __future__ import annotations

import json
import os
import pathlib
import unittest
from unittest.mock import MagicMock, patch

from finkey_search.backends import (
    BingWebSearchBackend,
    FallbackSearchChain,
    GuardedBackend,
)
from finkey_search.classifier import SearchNeedClassifier
from finkey_search.executor import WebSearchExecutor
from finkey_search.metrics import InternetMetrics
from finkey_search.reranker import SearchReranker
from finkey_search.resilience import CircuitBreaker, SlidingWindowLimiter
from finkey_search.schema import SearchDecision, SearchContext, SearchResult


_FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "cse_sample_response.json"


class TestBingWebParsing(unittest.TestCase):
    def test_parse_fixture_dedupes_domains(self) -> None:
        data = json.loads(
            (
                pathlib.Path(__file__).resolve().parent / "fixtures" / "bing_web_sample_response.json"
            ).read_text(encoding="utf-8")
        )
        be = BingWebSearchBackend(subscription_key="dummy")
        rows = be._parse_web(data, max_results=5)
        domains = [r.source for r in rows]
        self.assertEqual(domains.count("example-finance.test"), 1)
        self.assertTrue(any("nationalbank.example" in d for d in domains))


class TestGoogleCSEParsing(unittest.TestCase):
    def test_parse_fixture_dedupes_domains(self) -> None:
        data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
        exe = WebSearchExecutor(api_key="dummy", cx="dummy")
        rows = exe._parse_response(data, max_results=5)
        domains = [r.source for r in rows]
        self.assertEqual(domains.count("example-finance.test"), 1)
        self.assertTrue(any("nationalbank.example" in d for d in domains))


class TestClassifierConversation(unittest.TestCase):
    def test_semantic_llm_decides_follow_up(self) -> None:
        def _fake_llm(_prompt: str) -> str:
            return json.dumps(
                {
                    "search": True,
                    "query": "Нацбанк РК ключевая ставка прогноз второе полугодие",
                    "reason": "forecast depends on fresh guidance",
                    "category": "financial_news",
                }
            )

        with patch.dict(os.environ, {"FINKEY_SEARCH_CLASSIFIER_MODE": "llm_primary"}, clear=False):
            clf = SearchNeedClassifier()
            hist = ["user: какая ключевая ставка у Нацбанка РК сейчас?"]
            d = clf.classify(
                message="а прогноз на второе полугодие?",
                conversation_messages=hist,
                generate_fn=_fake_llm,
            )
            self.assertTrue(d.should_search)
            self.assertTrue(d.search_query.strip())

    def test_trivial_skip_without_llm(self) -> None:
        with patch.dict(os.environ, {"FINKEY_SEARCH_CLASSIFIER_MODE": "llm_primary"}, clear=False):
            clf = SearchNeedClassifier()
            d = clf.classify("спасибо!", generate_fn=None)
            self.assertFalse(d.should_search)

    def test_trivial_lexicon_skips_common_phrases(self) -> None:
        with patch.dict(os.environ, {"FINKEY_SEARCH_CLASSIFIER_MODE": "llm_primary"}, clear=False):
            clf = SearchNeedClassifier()
            for msg in (
                "good morning",
                "доброе утро",
                "thanks a lot",
                "большое спасибо",
                "how are you",
                "как дела",
                "see you later",
                "до свидания",
                "got it thanks",
                "понятно спасибо",
                "hold on",
                "подожди",
            ):
                with self.subTest(msg=msg):
                    d = clf.classify(msg, generate_fn=None)
                    self.assertFalse(d.should_search, msg)
                    self.assertEqual(d.reason, "trivial_skip_llm_primary")

    def test_substantive_query_not_trivial_skip(self) -> None:
        calls: list[str] = []

        def _fake_llm(_prompt: str) -> str:
            calls.append(_prompt)
            return json.dumps(
                {
                    "search": True,
                    "query": "usd kzt rate today",
                    "reason": "live rate",
                    "category": "currency_rate",
                    "verified_sources_required": False,
                    "depth": "fast",
                    "url": "",
                }
            )

        with patch.dict(os.environ, {"FINKEY_SEARCH_CLASSIFIER_MODE": "llm_primary"}, clear=False):
            clf = SearchNeedClassifier()
            d = clf.classify(
                "курс доллара к тенге сегодня",
                generate_fn=_fake_llm,
            )
            self.assertTrue(d.should_search)
            self.assertEqual(len(calls), 1)

    def test_legacy_regex_yes_still_works(self) -> None:
        with patch.dict(os.environ, {"FINKEY_SEARCH_CLASSIFIER_MODE": "legacy"}, clear=False):
            clf = SearchNeedClassifier()
            d = clf.classify(
                "курс доллара к тенге сегодня",
                generate_fn=None,
                allow_llm_tier=False,
            )
            self.assertTrue(d.should_search)


class TestReranker(unittest.TestCase):
    def test_orders_by_overlap(self) -> None:
        rr = SearchReranker(llm_min_pool=99)
        q = "ethereum price today"
        rows = [
            SearchResult(title="a", url="", snippet="unrelated cats blog", source="cats.test"),
            SearchResult(title="b", url="", snippet="Ethereum ETH trades near 3200 USD", source="markets.test"),
        ]
        out = rr.rerank(rows, q, max_keep=1, use_llm=False)
        self.assertIn("Ethereum", out[0].snippet)


class TestFallbackChainGuards(unittest.TestCase):
    def test_second_backend_used_when_first_empty(self) -> None:
        class _Empty:
            name = "empty"

            def search(self, query: str, *, max_results: int, language: str, date_restrict: str):
                return []

        class _Full:
            name = "full"

            def search(self, query: str, *, max_results: int, language: str, date_restrict: str):
                return [
                    SearchResult(title="hit", url="https://news.example/a", snippet="body", source="news.example"),
                ]

        metrics = InternetMetrics()
        chain = FallbackSearchChain(
            [
                GuardedBackend(
                    inner=_Empty(),
                    allow=lambda: True,
                    on_call=lambda n: metrics.record_backend_call(n),
                ),
                GuardedBackend(
                    inner=_Full(),
                    allow=lambda: True,
                    on_call=lambda n: metrics.record_backend_call(n),
                ),
            ]
        )
        rows = chain.search("q", max_results=5, language="", date_restrict="m3")
        self.assertEqual(len(rows), 1)
        snap = metrics.snapshot()
        self.assertGreaterEqual(snap.backend_calls.get("empty", 0), 1)


class TestResilience(unittest.TestCase):
    def test_breaker_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_seconds=30.0)
        self.assertTrue(cb.allow_request())
        cb.record_failure()
        cb.record_failure()
        self.assertFalse(cb.allow_request())


class TestPrometheusExport(unittest.TestCase):
    def test_metrics_text_contains_counters(self) -> None:
        m = InternetMetrics()
        m.record_search_start()
        body = m.to_prometheus_text()
        self.assertIn("finkey_internet_searches_started_total", body)


class TestSlidingLimiter(unittest.TestCase):
    def test_blocks_after_burst(self) -> None:
        lim = SlidingWindowLimiter(max_calls=2, window_seconds=60.0)
        self.assertTrue(lim.acquire())
        self.assertTrue(lim.acquire())
        self.assertFalse(lim.acquire())


class TestConversationSearchBudget(unittest.TestCase):
    def test_third_search_skipped_when_cap_two(self) -> None:
        from finkey_search.engine import WebSearchEngine

        ctr = {"n": 0}

        def _classify(*args, **kwargs):
            ctr["n"] += 1
            return SearchDecision(
                should_search=True,
                search_query=f"query-{ctr['n']}",
                category=None,
                reason="test",
            )

        row = SearchResult(
            title="t",
            url="https://a.test/x",
            snippet="s",
            source="a.test",
        )
        ctx_ok = SearchContext(
            found=True,
            query_used="q",
            category=None,
            synthesized="block",
        )

        with patch.dict(
            os.environ,
            {
                "FINKEY_SEARCH_MAX_PER_CONVERSATION": "2",
                "FINKEY_WEB_PAGE_ENRICH": "0",
                "FINKEY_ENABLE_BROWSING_OPERATOR": "0",
            },
            clear=False,
        ):
            eng = WebSearchEngine(bing_subscription_key="", cache_ttl=0.0)
            eng._classifier.classify = _classify  # type: ignore[method-assign]
            eng._chain.search = MagicMock(return_value=[row])  # type: ignore[method-assign]
            eng._synthesizer.synthesize = MagicMock(return_value=ctx_ok)  # type: ignore[method-assign]

            conv = "dlg-1"
            self.assertTrue(eng.run("msg", turn_number=1, conversation_key=conv).found)
            self.assertTrue(eng.run("msg", turn_number=2, conversation_key=conv).found)
            self.assertFalse(eng.run("msg", turn_number=3, conversation_key=conv).found)


class TestWebSearchRedisPayload(unittest.TestCase):
    def test_search_context_json_roundtrip(self) -> None:
        import json

        from finkey_search.redis_cache import (
            _max_bytes,
            context_to_payload,
            payload_to_context,
        )
        from finkey_search.schema import SearchCategory

        ctx = SearchContext(
            found=True,
            query_used="hello world",
            category=SearchCategory.GENERAL,
            verified_sources_required=False,
            citation_urls=["https://x.example/a"],
            results=[
                SearchResult(
                    title="T",
                    url="https://x.example/a",
                    snippet="sn",
                    source="x",
                    enriched_text="e" * 900,
                )
            ],
            synthesized="Synth " * 80,
            search_date="2026-01-01",
            total_results=3,
        )
        raw = context_to_payload(ctx, max_bytes=_max_bytes(), enrich_cap=400)
        back = payload_to_context(json.loads(raw))
        self.assertIsNotNone(back)
        assert back is not None
        self.assertTrue(back.found)
        self.assertEqual(back.query_used, "hello world")
        self.assertEqual(back.category, SearchCategory.GENERAL)
        self.assertEqual(len(back.results), 1)
        et = (back.results[0].enriched_text or "")
        self.assertTrue(et.startswith("e" * 400))
        self.assertIn("truncated-for-cache", et)


class TestCseHttpRetries(unittest.TestCase):
    def test_retries_after_429(self) -> None:
        from finkey_search.executor import WebSearchExecutor

        r429 = MagicMock()
        r429.status_code = 429

        r200 = MagicMock()
        r200.status_code = 200
        r200.raise_for_status.return_value = None
        r200.json.return_value = {"items": []}

        exe = WebSearchExecutor(api_key="k", cx="c")
        with patch.dict(os.environ, {"FINKEY_SEARCH_CSE_HTTP_RETRIES": "4"}, clear=False):
            with patch("requests.get", side_effect=[r429, r200]) as mock_get:
                out = exe._http_get("https://example.invalid/x", {"a": "b"})
        self.assertEqual(out, {"items": []})
        self.assertEqual(mock_get.call_count, 2)


class TestWaveEnrichEarlyStop(unittest.TestCase):
    """ChatGPT-like wave enrich: stop when answer_ready, keep high max budget."""

    def test_answer_ready_fast_agreeing_pages(self) -> None:
        from finkey_search.evidence_gate import answer_ready
        from finkey_search.schema import SearchDepth, SearchResult

        rows = [
            SearchResult(
                title="Now in Berlin",
                url="https://a.example/w",
                snippet="snippet",
                source="a.example",
                enriched_text="Current conditions in Berlin: temperature 22 C, wind 5 m/s.",
            ),
            SearchResult(
                title="Berlin live",
                url="https://b.example/w",
                snippet="snippet",
                source="b.example",
                enriched_text="Berlin right now: 22 degrees, partly cloudy.",
            ),
        ]
        r = answer_ready("Какая сейчас погода в Berlin?", rows, depth=SearchDepth.FAST)
        self.assertTrue(r.ready)
        self.assertEqual(r.effective_depth, "fast")
        self.assertIn("fast", r.reason)

    def test_answer_ready_conflict_not_ready(self) -> None:
        from finkey_search.evidence_gate import answer_ready
        from finkey_search.schema import SearchDepth, SearchResult

        rows = [
            SearchResult(
                title="A",
                url="https://a.example/1",
                snippet="s",
                source="a.example",
                enriched_text="Value is 10 units today for the metric.",
            ),
            SearchResult(
                title="B",
                url="https://b.example/1",
                snippet="s",
                source="b.example",
                enriched_text="Value is 90 units today for the metric.",
            ),
        ]
        r = answer_ready("What is the current value?", rows, depth=SearchDepth.FAST)
        self.assertFalse(r.ready)
        self.assertTrue(r.numeric_conflict)
        self.assertEqual(r.reason, "numeric_conflict")

    def test_answer_ready_deep_thin_keeps_going(self) -> None:
        from finkey_search.evidence_gate import answer_ready
        from finkey_search.schema import SearchDepth, SearchResult

        rows = [
            SearchResult(
                title="t1",
                url="https://a.example/1",
                snippet="hi",
                source="a.example",
                enriched_text="Short note without enough research body.",
            ),
            SearchResult(
                title="t2",
                url="https://b.example/1",
                snippet="hi",
                source="b.example",
                enriched_text="Another short note.",
            ),
        ]
        # Avoid locative false-positives ("по всем" → place); research shape still escalates.
        msg = (
            "Analyze and compare market segments: prices, rents, "
            "gather comprehensive data for the current year across all metrics"
        )
        r = answer_ready(msg, rows, depth=SearchDepth.DEEP)
        self.assertFalse(r.ready)
        self.assertTrue(
            r.reason.startswith("deep_") or "thin" in r.reason or "need" in r.reason,
            msg=r.reason,
        )

    def test_wave_enrich_stops_after_two_on_fast(self) -> None:
        from finkey_search.url_policy import URLPolicy
        from finkey_search.page_enrichment import enrich_results_with_structured_pages
        from finkey_search.schema import SearchDepth, SearchResult

        fetch_urls: list[str] = []

        def fake_http(url: str, timeout_s: float = 10.0, max_chars: int = 12_000):
            fetch_urls.append(url)
            # Same temperature + place token → FAST answer_ready after wave 1.
            return f"Live Paris reading: temperature 18 C in Paris right now. url={url}", None

        rows = [
            SearchResult(title=f"t{i}", url=f"https://ex{i}.example/p", snippet="s" * 40, source=f"ex{i}.example")
            for i in range(1, 7)
        ]
        with patch.dict(
            os.environ,
            {"FINKEY_WEB_ENRICH_WAVE_SIZE": "2", "FINKEY_WEB_ENRICH_EARLY_STOP": "1"},
            clear=False,
        ):
            with patch(
                "finkey_search.page_enrichment.fetch_page_text_scrapling",
                side_effect=lambda url, timeout_s=10.0, max_chars=12_000: ("", "skip"),
            ):
                with patch(
                    "finkey_search.page_enrichment.fetch_page_text_http",
                    side_effect=fake_http,
                ):
                    enrich_results_with_structured_pages(
                        rows,
                        max_pages=6,
                        url_policy=URLPolicy(block_private_and_loopback=True),
                        max_chars_per_page=4000,
                        fetch_mode="http",
                        http_fallback=False,
                        message="What is the temperature in Paris now?",
                        depth=SearchDepth.FAST,
                        early_stop=True,
                    )
        self.assertEqual(len(fetch_urls), 2)
        enriched = [r for r in rows if (r.enriched_text or "").strip()]
        self.assertEqual(len(enriched), 2)

    def test_wave_enrich_conflict_runs_second_wave(self) -> None:
        from finkey_search.url_policy import URLPolicy
        from finkey_search.page_enrichment import enrich_results_with_structured_pages
        from finkey_search.schema import SearchDepth, SearchResult

        fetch_urls: list[str] = []
        bodies = {
            "https://ex1.example/p": "Metric reading A is 10 points today.",
            "https://ex2.example/p": "Metric reading B is 80 points today.",
            "https://ex3.example/p": "Metric reading C is 11 points today.",
            "https://ex4.example/p": "Metric reading D is 10 points today.",
        }

        def fake_http(url: str, timeout_s: float = 10.0, max_chars: int = 12_000):
            fetch_urls.append(url)
            return bodies.get(url, "no numbers here at all"), None

        rows = [
            SearchResult(title=f"t{i}", url=f"https://ex{i}.example/p", snippet="s" * 40, source=f"ex{i}.example")
            for i in range(1, 5)
        ]
        with patch.dict(
            os.environ,
            {"FINKEY_WEB_ENRICH_WAVE_SIZE": "2", "FINKEY_WEB_ENRICH_EARLY_STOP": "1"},
            clear=False,
        ):
            with patch(
                "finkey_search.page_enrichment.fetch_page_text_scrapling",
                side_effect=lambda url, timeout_s=10.0, max_chars=12_000: ("", "skip"),
            ):
                with patch(
                    "finkey_search.page_enrichment.fetch_page_text_http",
                    side_effect=fake_http,
                ):
                    enrich_results_with_structured_pages(
                        rows,
                        max_pages=4,
                        url_policy=URLPolicy(block_private_and_loopback=True),
                        max_chars_per_page=4000,
                        fetch_mode="http",
                        http_fallback=False,
                        message="What is the current metric value?",
                        depth=SearchDepth.FAST,
                        early_stop=True,
                    )
        # Wave1 conflict (10 vs 80) → wave2 must run (≥3 fetches).
        self.assertGreaterEqual(len(fetch_urls), 3)

    def test_enrich_deadline_abandons_slow_pages_keeps_fast(self) -> None:
        """Hung hosts must not burn tens of seconds — keep whoever answered in time."""
        import time as _time

        from finkey_search.url_policy import URLPolicy
        from finkey_search.page_enrichment import enrich_results_with_structured_pages
        from finkey_search.schema import SearchDepth, SearchResult

        def fake_http(url: str, timeout_s: float = 10.0, max_chars: int = 12_000):
            if "slow" in url:
                _time.sleep(min(timeout_s + 0.5, 3.0))
                return "late body that should be abandoned", None
            return (
                "Live Berlin reading: temperature 21 C in Berlin right now. "
                f"source={url}"
            ), None

        rows = [
            SearchResult(
                title="fast",
                url="https://fast.example/p",
                snippet="s" * 40,
                source="fast.example",
            ),
            SearchResult(
                title="slow",
                url="https://slow.example/p",
                snippet="s" * 40,
                source="slow.example",
            ),
            SearchResult(
                title="also-slow",
                url="https://slow2.example/p",
                snippet="s" * 40,
                source="slow2.example",
            ),
        ]
        t0 = _time.perf_counter()
        with patch(
            "finkey_search.page_enrichment.fetch_page_text_scrapling",
            side_effect=lambda url, timeout_s=10.0, max_chars=12_000: ("", "skip"),
        ):
            with patch(
                "finkey_search.page_enrichment.fetch_page_text_http",
                side_effect=fake_http,
            ):
                enrich_results_with_structured_pages(
                    rows,
                    max_pages=3,
                    url_policy=URLPolicy(block_private_and_loopback=True),
                    max_chars_per_page=4000,
                    fetch_mode="http",
                    http_fallback=False,
                    message="What is the temperature in Berlin now?",
                    depth=SearchDepth.FAST,
                    early_stop=True,
                    deadline_s=1.2,
                )
        elapsed = _time.perf_counter() - t0
        self.assertLess(elapsed, 2.2, msg=f"deadline did not cut hung pages: {elapsed:.2f}s")
        self.assertTrue((rows[0].enriched_text or "").strip())
        # Slow pages may or may not land; the point is we did not wait them out.
        # With early-stop after a solid fast page, we often never need them.
        self.assertIn("21", rows[0].enriched_text or "")

    def test_enrich_deadline_zero_disables_budget(self) -> None:
        from finkey_search.page_enrichment import _enrich_deadline_s

        with patch.dict(os.environ, {"FINKEY_WEB_ENRICH_DEADLINE_S": "0"}, clear=False):
            self.assertEqual(_enrich_deadline_s(fast=True, browser_mode=False), 0.0)

    def test_no_topic_weather_regex_in_answer_ready(self) -> None:
        import inspect

        from finkey_search import evidence_gate as eg

        src = inspect.getsource(eg.answer_ready) + inspect.getsource(eg.numeric_conflict_across_enriched)
        self.assertNotIn("weather", src.lower())
        self.assertNotIn("погод", src.lower())
        self.assertNotIn("fx", src.lower())


