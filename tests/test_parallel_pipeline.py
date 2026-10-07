"""
Timing + thread checks: prove page enrichment runs concurrently.

  PYTHONPATH=src python3 -m pytest tests/test_parallel_pipeline.py
"""
from __future__ import annotations

import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from finkey_search.url_policy import URLPolicy
from finkey_search.page_enrichment import enrich_results_with_structured_pages
from finkey_search.schema import SearchResult


class TestParallelPageEnrich(unittest.TestCase):
    """Structured fetch is mocked slow; wall clock must be ~one sleep, not N×."""

    def test_enrich_three_urls_parallel_faster_than_sequential(self) -> None:
        policy = URLPolicy(block_private_and_loopback=True)
        urls = [
            "https://example-a.test/page1",
            "https://example-b.test/page2",
            "https://example-c.test/page3",
        ]
        results = [
            SearchResult(title="t", url=u, snippet="s", source="example", relevance=0.9)
            for u in urls
        ]

        sleep_s = 0.18
        events: list[tuple[int, float]] = []

        def fake_structured(url: str, _cfg):
            t0 = time.perf_counter()
            tid = threading.get_ident()
            events.append((tid, t0))
            time.sleep(sleep_s)
            m = MagicMock()
            m.ok = True
            m.main_content = "x" * 400
            m.error = ""
            return m

        env = {
            "FINKEY_PAGE_ENRICH_CONCURRENCY": "4",
            "FINKEY_PAGE_ENRICH_MAX_BURST": "12",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch(
                "finkey_search.page_enrichment.fetch_structured_sync",
                side_effect=fake_structured,
            ):
                t_start = time.perf_counter()
                enrich_results_with_structured_pages(
                    results,
                    max_pages=3,
                    url_policy=policy,
                    max_chars_per_page=2000,
                    metrics=None,
                )
                elapsed = time.perf_counter() - t_start

        for r in results:
            self.assertTrue((r.enriched_text or "").strip(), msg=f"missing enrich for {r.url}")

        self.assertLess(
            elapsed,
            2.2 * sleep_s,
            msg=f"too slow for parallel: {elapsed:.3f}s (expected < {2.2 * sleep_s:.3f})",
        )
        if len(events) >= 2:
            events.sort(key=lambda x: x[1])
            span = events[-1][1] - events[0][1]
            self.assertLess(
                span,
                1.2 * sleep_s,
                msg=f"starts too sequential: span={span:.3f}s between first/last fetch start",
            )


if __name__ == "__main__":
    unittest.main()
