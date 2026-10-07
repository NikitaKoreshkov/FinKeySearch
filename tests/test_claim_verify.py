"""Tests for claim_verify + GitHub structured parse (no network)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from finkey_search.claim_verify import (
    extract_claims,
    looks_like_research_answer,
    verify_claims_against_evidence,
)
from finkey_search.research_orchestrator import plans_from_explicit_subqueries
from finkey_search.schema import SearchCategory, SearchContext, SearchResult
from finkey_search.structured.github_repo import (
    parse_github_repos,
)
from finkey_search.tools_schema import run_web_search_tool


class TestClaimVerify(unittest.TestCase):
    def test_extract_stars_and_money(self) -> None:
        text = "LangGraph has ~31k stars. NVIDIA Data Center brought $75.2B."
        claims = extract_claims(text)
        kinds = {c.kind for c in claims}
        self.assertIn("stars", kinds)
        self.assertTrue(any(c.value and c.value > 1e9 for c in claims if c.kind == "number"))

    def test_supported_when_evidence_has_figure(self) -> None:
        draft = "Installed capacity is 3.75 ГВт as of mid-2026."
        evidence = "Установленная мощность объектов ВИЭ достигла 3749.62 МВт (~3.75 ГВт)."
        report = verify_claims_against_evidence(draft, evidence)
        self.assertTrue(report.ok)
        self.assertTrue(any(c.verdict == "supported" for c in report.claims))

    def test_unsupported_figure(self) -> None:
        draft = "The market is exactly $999 billion in 2026."
        evidence = "AI accelerators market reached $174.69 billion in 2026."
        report = verify_claims_against_evidence(draft, evidence)
        self.assertFalse(report.ok)
        self.assertTrue(report.unsupported)

    def test_structured_github_stars_override_blog(self) -> None:
        from finkey_search.claim_verify import extract_structured_github_stars

        evidence = (
            "🔒 STRUCTURED FACTS\n"
            "[STRUCTURED:github_api] repo=langchain-ai/langgraph\n"
            "stars=38214\n"
            "forks=6426\n"
            "Some blog says LangGraph has 135000 stars.\n"
        )
        self.assertEqual(
            extract_structured_github_stars(evidence).get("langchain-ai/langgraph"),
            38214,
        )
        bad = verify_claims_against_evidence(
            "LangGraph has 135000 stars on GitHub.",
            evidence,
        )
        self.assertFalse(bad.ok)
        self.assertTrue(any(c.kind == "stars" for c in bad.unsupported))
        good = verify_claims_against_evidence(
            "LangGraph has 38214 stars on GitHub.",
            evidence,
        )
        self.assertTrue(good.ok)

        ru = verify_claims_against_evidence(
            "Сейчас у langchain-ai/langgraph **38 214** звёзд.",
            evidence,
        )
        self.assertTrue(ru.ok, ru.to_dict())

        rounded = verify_claims_against_evidence(
            "Сейчас у langchain-ai/langgraph **38,2k** звёзд.",
            evidence,
        )
        self.assertTrue(rounded.ok, rounded.to_dict())
        self.assertTrue(
            any(c.value and c.value > 1000 for c in extract_claims("**38,2k** звёзд"))
        )

    def test_primary_host_overrides_blog_number(self) -> None:
        evidence = (
            "Official filing https://www.sec.gov/Archives/edgar/data/1.htm "
            "states revenue of $50.2 billion in 2025.\n"
            "SEO blog https://random-blog.example/ai-market says the market is $999 billion.\n"
        )
        bad = verify_claims_against_evidence(
            "Revenue is exactly $999 billion.",
            evidence,
        )
        self.assertFalse(bad.ok)
        self.assertTrue(bad.conflicts or bad.unsupported)
        good = verify_claims_against_evidence(
            "Revenue is $50.2 billion.",
            evidence,
        )
        self.assertTrue(good.ok)
        self.assertTrue(
            any(
                c.verdict == "supported" and "primary" in (c.evidence_snip or "")
                for c in good.claims
            )
        )

    def test_looks_like_research_answer(self) -> None:
        thin = "Spain won. Score 1:0."
        self.assertFalse(looks_like_research_answer(thin))
        fat = (
            "Market size is $174.69 billion in 2026 with NVIDIA at 75% share. "
            "AMD holds 6% (~$7 billion). Inference is 65% of demand. "
            "HBM market is $2.89 billion. Capacity is 3.8 ГВт with 15% target. "
            "LangGraph has 31k stars and CrewAI has 52k stars on GitHub."
        )
        self.assertTrue(looks_like_research_answer(fat * 3))


class TestGitHubParse(unittest.TestCase):
    def test_parse_url_and_repo_colon(self) -> None:
        repos = parse_github_repos(
            "see https://github.com/langchain-ai/langgraph and repo:crewAIInc/crewAI"
        )
        names = {f"{o}/{r}".lower() for o, r in repos}
        self.assertIn("langchain-ai/langgraph", names)
        self.assertIn("crewaiinc/crewai", names)


class TestExplicitSubqueries(unittest.TestCase):
    def test_plans_skip_duplicates_and_cap(self) -> None:
        plans = plans_from_explicit_subqueries(
            "primary q",
            ["a", "a", "b", "c", "d", "e", "f", "g"],
        )
        qs = [p.query for p in plans]
        self.assertEqual(len(qs), len(set(q.lower() for q in qs)))
        self.assertLessEqual(len(plans), 8)
        self.assertIn("primary q", qs)


class TestToolsSchemaExtended(unittest.TestCase):
    def test_fast_passes_serp_draft(self) -> None:
        engine = MagicMock()
        engine.run.return_value = SearchContext(
            found=True,
            query_used="q",
            category=SearchCategory.GENERAL,
            citation_urls=["https://example.com/a"],
            results=[
                SearchResult(title="A", url="https://example.com/a", snippet="s"),
                SearchResult(title="B", url="https://example.com/b", snippet="s2"),
            ],
            synthesized="LIVE WEB DATA",
            search_date="27 July 2026",
            total_results=2,
        )
        out = json.loads(
            run_web_search_tool(
                "web_search",
                {
                    "query": "q",
                    "depth": "fast",
                    "subqueries": ["q1", "q2"],
                },
                engine=engine,
            )
        )
        self.assertEqual(out["phase"], "draft")
        self.assertIn("https://example.com/b", out["candidate_urls"])
        kwargs = engine.run.call_args.kwargs
        self.assertTrue(kwargs["serp_draft"])
        self.assertEqual(kwargs["explicit_subqueries"], ["q1", "q2"])

    def test_deep_with_urls(self) -> None:
        engine = MagicMock()
        engine.run.return_value = SearchContext(
            found=True,
            query_used="q",
            category=SearchCategory.GENERAL,
            citation_urls=["https://example.com/x"],
            synthesized="DEEP",
            total_results=1,
        )
        out = json.loads(
            run_web_search_tool(
                "web_search",
                {
                    "query": "q",
                    "depth": "deep",
                    "urls": ["https://example.com/x"],
                },
                engine=engine,
            )
        )
        self.assertEqual(out["phase"], "deep")
        kwargs = engine.run.call_args.kwargs
        self.assertFalse(kwargs["serp_draft"])
        self.assertEqual(kwargs["enrich_urls"], ["https://example.com/x"])

    def test_verify_mode(self) -> None:
        engine = MagicMock()
        out = json.loads(
            run_web_search_tool(
                "web_search",
                {
                    "query": "unused",
                    "mode": "verify",
                    "draft_answer": "Revenue is $174.69 billion.",
                    "evidence": "Market size USD 174.69 billion in 2026.",
                },
                engine=engine,
            )
        )
        engine.run.assert_not_called()
        self.assertEqual(out["phase"], "verify")
        self.assertTrue(out["ok"])


class TestEnrichAllowlist(unittest.TestCase):
    def test_ensure_enrich_url_rows_injects_missing(self) -> None:
        from finkey_search.engine import WebSearchEngine

        rows = [
            SearchResult(title="a", url="https://example.com/a", snippet="s"),
        ]
        out = WebSearchEngine._ensure_enrich_url_rows(
            rows,
            ["https://example.com/a", "https://example.com/b"],
        )
        urls = [r.url for r in out]
        self.assertIn("https://example.com/b", urls)
        self.assertEqual(urls.count("https://example.com/a"), 1)

    def test_norm_url_key(self) -> None:
        from finkey_search.page_enrichment import _norm_url_key

        self.assertEqual(
            _norm_url_key("https://www.Example.com/Path/#frag"),
            "https://example.com/path",
        )


if __name__ == "__main__":
    unittest.main()
