"""compact_serp_query — keep backends free of instruction essays."""

from __future__ import annotations

import unittest

from finkey_search.live_query import compact_serp_query


class TestCompactSerpQuery(unittest.TestCase):
    def test_keeps_short_factual_query(self) -> None:
        q = "USD KZT exchange rate today"
        self.assertEqual(compact_serp_query(q), q)

    def test_compacts_russian_research_essay(self) -> None:
        essay = (
            "Проанализируй глобальный рынок AI-ускорителей (GPU/ASIC/NPU) на "
            "сегодняшний день: объём рынка (TAM/revenue где есть цифры), "
            "доля NVIDIA / AMD / Google TPU, ключевые драйверы спроса, "
            "и на основе свежих отчётов дай прогноз к концу 2027 года."
        )
        out = compact_serp_query(essay, goal=essay)
        self.assertTrue(out)
        self.assertLessEqual(len(out), 240)
        self.assertLess(len(out), len(essay))
        self.assertNotIn("дай прогноз", out.lower())

    def test_prefers_short_tool_query_over_long_goal(self) -> None:
        goal = "Проанализируй рынок AI-ускорителей и дай прогноз к 2027 со источниками"
        q = "AI accelerator market size 2026 NVIDIA share"
        self.assertEqual(compact_serp_query(q, goal=goal), q)


if __name__ == "__main__":
    unittest.main()
