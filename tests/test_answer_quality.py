"""Answer quality gate + GitHub name aliases."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from finkey_search.answer_quality import (
    assess_answer_quality,
    looks_like_collapsed_answer,
)
from finkey_search.schema import SearchResult
from finkey_search.structured.github_repo import (
    merge_github_structured_results,
    parse_github_repos,
)


class TestAnswerQuality(unittest.TestCase):
    def test_phrase_loop_detected(self) -> None:
        text = "AMD: " + ("однозначные " * 40) + "доля рынка 5%."
        report = assess_answer_quality(text, research=True)
        self.assertFalse(report.ok)
        self.assertTrue(looks_like_collapsed_answer(text))
        self.assertTrue(any("loop" in r or "repetition" in r for r in report.reasons))

    def test_clean_research_ok(self) -> None:
        text = (
            "### Market\n"
            "NVIDIA holds about 80% of AI accelerator revenue in 2026. "
            "Data Center segment reported $75.2 billion in Q1 FY2027. "
            "AMD remains a distant second near 6%. Inference now dominates demand.\n"
            "### Forecast\n"
            "Base case keeps NVIDIA above 70% through 2027 while custom ASICs grow.\n"
        ) * 3
        report = assess_answer_quality(text, research=True)
        self.assertTrue(report.ok)

    def test_markdown_table_rules_not_phrase_loop(self) -> None:
        text = (
            "### Сравнение\n\n"
            "| Проект | Stars | Лицензия |\n"
            "|--------|-------|----------|\n"
            "| LangGraph | 30k | MIT |\n"
            "| CrewAI | 50k | MIT |\n"
            "| AutoGen | 40k | MIT |\n\n"
            "LangGraph лучше для B2B checkpointing. CrewAI проще для демо. "
            "Выбор зависит от требований к durable memory и MCP.\n"
        )
        report = assess_answer_quality(text, research=True)
        self.assertTrue(report.ok, report.reasons)


class TestGitHubAliases(unittest.TestCase):
    def test_langgraph_name_maps_to_repo(self) -> None:
        repos = parse_github_repos(
            "Compare LangGraph and CrewAI multi-agent frameworks on GitHub stars"
        )
        keys = {f"{o}/{r}".lower() for o, r in repos}
        self.assertIn("langchain-ai/langgraph", keys)
        self.assertIn("crewaiinc/crewai", keys)

    def test_langgraph_pair_does_not_also_pull_langchain(self) -> None:
        """``langchain-ai/langgraph`` must not trigger the bare ``langchain`` alias."""
        repos = parse_github_repos(
            "Сколько stars у langchain-ai/langgraph? LangGraph GitHub stars"
        )
        keys = [f"{o}/{r}".lower() for o, r in repos]
        self.assertEqual(keys[0], "langchain-ai/langgraph")
        self.assertNotIn("langchain-ai/langchain", keys)

    def test_merge_injects_structured(self) -> None:
        fake = SearchResult(
            title="langchain-ai/langgraph (GitHub API)",
            url="https://github.com/langchain-ai/langgraph",
            snippet="stars=33900",
            source="github_api",
            enriched_text="[STRUCTURED:github_api] repo=langchain-ai/langgraph\nstars=33900\n",
            relevance=1.2,
        )
        with patch(
            "finkey_search.structured.github_repo.fetch_github_repo_facts",
            return_value=fake,
        ):
            out = merge_github_structured_results(
                [SearchResult(title="blog", url="https://example.com/x", snippet="LangGraph 100k stars")],
                query="LangGraph GitHub stars production",
                message="Find LangGraph open source agent framework",
            )
        self.assertEqual(out[0].source, "github_api")
        self.assertIn("33900", out[0].enriched_text or "")


if __name__ == "__main__":
    unittest.main()
