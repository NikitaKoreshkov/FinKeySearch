"""Native web_search tool schema + executor."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from finkey_search.schema import SearchCategory, SearchContext
from finkey_search.tools_schema import (
    WEB_SEARCH_TOOL_NAMES,
    WEB_SEARCH_TOOLS,
    run_web_search_tool,
)


class TestWebSearchToolsSchema(unittest.TestCase):
    def test_schema_has_web_search(self) -> None:
        self.assertIn("web_search", WEB_SEARCH_TOOL_NAMES)
        self.assertEqual(WEB_SEARCH_TOOLS[0]["function"]["name"], "web_search")
        desc = WEB_SEARCH_TOOLS[0]["function"]["description"].lower()
        self.assertIn("not certain", desc)
        self.assertIn("do not claim you lack internet", desc)

    def test_description_teaches_the_blind_spot_rule(self) -> None:
        """
        The tool must tell the model that an unremembered past event was missed,
        not pending — that confusion is what makes it answer "hasn't happened yet"
        instead of searching.
        """
        desc = WEB_SEARCH_TOOLS[0]["function"]["description"].lower()
        self.assertIn("you missed it", desc)
        self.assertIn("already-past date", desc)
        self.assertIn("training cutoff", desc)

    def test_description_does_not_rely_on_trigger_words(self) -> None:
        """Routing must follow the substance of the question, not its vocabulary."""
        desc = WEB_SEARCH_TOOLS[0]["function"]["description"].lower()
        self.assertIn("not its wording", desc)

    def test_run_requires_query(self) -> None:
        engine = MagicMock()
        out = json.loads(run_web_search_tool("web_search", {}, engine=engine))
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "query_required")
        engine.run.assert_not_called()

    def test_run_force_search(self) -> None:
        engine = MagicMock()
        engine.run.return_value = SearchContext(
            found=True,
            query_used="курс доллара",
            category=SearchCategory.CURRENCY_RATE,
            citation_urls=["https://example.com"],
            synthesized="LIVE WEB DATA\n1 USD = 500",
            search_date="09 July 2026",
            total_results=3,
        )
        out = json.loads(
            run_web_search_tool(
                "web_search",
                {"query": "курс доллара", "depth": "fast"},
                engine=engine,
                conversation_key="c1",
                turn_number=2,
            )
        )
        self.assertTrue(out["ok"])
        self.assertTrue(out["useful"])
        self.assertIn("LIVE WEB", out["synthesized"])
        kwargs = engine.run.call_args.kwargs
        self.assertTrue(kwargs["force_search"])
        self.assertFalse(kwargs["allow_llm_tier"])
        self.assertTrue(kwargs.get("serp_draft"))
        self.assertEqual(out.get("phase"), "draft")

    def test_user_goal_drives_message_serp_query_override(self) -> None:
        """Full user goal → orch/deep intent; short tool query → SERP override."""
        engine = MagicMock()
        engine.run.return_value = SearchContext(
            found=True,
            query_used="AI accelerator market size",
            category=SearchCategory.GENERAL,
            citation_urls=["https://example.com"],
            synthesized="LIVE WEB DATA\nok",
            search_date="27 July 2026",
            total_results=5,
        )
        goal = (
            "Проанализируй глобальный рынок AI-ускорителей и дай прогноз к 2027: "
            "TAM, доли NVIDIA/AMD, bottlenecks"
        )
        out = json.loads(
            run_web_search_tool(
                "web_search",
                {"query": "AI accelerator market size 2026", "depth": "fast"},
                engine=engine,
                user_goal=goal,
                turn_number=1,
            )
        )
        self.assertTrue(out["ok"])
        kwargs = engine.run.call_args.kwargs
        self.assertEqual(kwargs["message"], goal)
        self.assertEqual(kwargs["search_query_override"], "AI accelerator market size 2026")

    def test_essay_query_is_compacted_for_serp(self) -> None:
        """Full research essay must not be sent to Serper/Brave as q=."""
        engine = MagicMock()
        engine.run.return_value = SearchContext(
            found=True,
            query_used="AI",
            category=SearchCategory.GENERAL,
            citation_urls=["https://example.com"],
            synthesized="LIVE",
            search_date="27 July 2026",
            total_results=1,
        )
        essay = (
            "Проанализируй глобальный рынок AI-ускорителей на сегодняшний день: "
            "объём рынка, доля NVIDIA / AMD, узкие места HBM, и дай прогноз к 2027 "
            "на основе свежих отчётов. Обязательно опирайся на конкретные источники."
        )
        out = json.loads(
            run_web_search_tool(
                "web_search",
                {"query": essay, "depth": "fast"},
                engine=engine,
                user_goal=essay,
            )
        )
        self.assertTrue(out["ok"])
        kwargs = engine.run.call_args.kwargs
        self.assertEqual(kwargs["message"], essay)
        serp = kwargs["search_query_override"] or ""
        self.assertLessEqual(len(serp), 240)
        self.assertLessEqual(serp.count(" "), 24)
        self.assertNotIn("Обязательно опирайся", serp)


if __name__ == "__main__":
    unittest.main()
