"""Tests for SERP / operator URL quality helpers."""
import unittest

from finkey_search.url_quality import (
    is_blocked_for_operator_browse,
    is_primary_source_url,
    operator_entry_priority_score,
    serp_url_quality_adjustment,
)


class TestUrlQuality(unittest.TestCase):
    def test_operator_blocks_instagram_reel(self) -> None:
        self.assertTrue(
            is_blocked_for_operator_browse("https://www.instagram.com/reel/AbCdEf/")
        )

    def test_operator_blocks_youtube(self) -> None:
        self.assertTrue(
            is_blocked_for_operator_browse("https://www.youtube.com/watch?v=xyz")
        )

    def test_operator_allows_youtube_when_video_intent(self) -> None:
        self.assertFalse(
            is_blocked_for_operator_browse(
                "https://www.youtube.com/watch?v=xyz",
                media_kind="video",
            )
        )

    def test_operator_allows_pinterest_when_image_intent(self) -> None:
        self.assertFalse(
            is_blocked_for_operator_browse(
                "https://www.pinterest.com/pin/123/",
                media_kind="image",
            )
        )

    def test_operator_allows_reuters(self) -> None:
        url = "https://www.reuters.com/world/middle-east/article-slug-2026/"
        self.assertFalse(is_blocked_for_operator_browse(url))

    def test_operator_blocks_login_next(self) -> None:
        self.assertTrue(
            is_blocked_for_operator_browse(
                "https://www.instagram.com/accounts/login/?next=/foo"
            )
        )

    def test_serp_penalty_social_strong(self) -> None:
        self.assertLess(
            serp_url_quality_adjustment("https://tiktok.com/@user/video/1"),
            -1.5,
        )

    def test_serp_boosts_youtube_when_video_intent(self) -> None:
        plain = serp_url_quality_adjustment("https://www.youtube.com/watch?v=1")
        boosted = serp_url_quality_adjustment(
            "https://www.youtube.com/watch?v=1",
            media_kind="video",
        )
        self.assertGreater(boosted, plain)

    def test_operator_prefers_news_over_facebook_discouraged(self) -> None:
        r = operator_entry_priority_score(
            "https://www.reuters.com/world/article", base_relevance=0.5
        )
        f = operator_entry_priority_score(
            "https://www.facebook.com/story.php", base_relevance=0.5
        )
        self.assertGreater(r, f)

    def test_serp_boosts_gov_edu_structural(self) -> None:
        gov = serp_url_quality_adjustment("https://www.sec.gov/Archives/edgar/data/1.htm")
        edu = serp_url_quality_adjustment("https://www.mit.edu/research/ai")
        blog = serp_url_quality_adjustment("https://random-blog.example/posts/ai")
        self.assertGreater(gov, blog)
        self.assertGreater(edu, blog)
        self.assertGreater(gov, 0.5)

    def test_serp_boosts_investor_press_paths(self) -> None:
        ir = serp_url_quality_adjustment(
            "https://investor.nvidia.com/news/press-releases/default.aspx"
        )
        plain = serp_url_quality_adjustment("https://www.example.com/blog/ai-chips")
        self.assertGreater(ir, plain)

    def test_serp_demotes_social_harder_than_before(self) -> None:
        fb = serp_url_quality_adjustment("https://www.facebook.com/story.php")
        self.assertLess(fb, -1.0)

    def test_serp_boosts_github_host(self) -> None:
        gh = serp_url_quality_adjustment("https://github.com/langchain-ai/langgraph")
        other = serp_url_quality_adjustment("https://example.com/langgraph")
        self.assertGreater(gh, other)

    def test_is_primary_source_url(self) -> None:
        self.assertTrue(is_primary_source_url("https://www.sec.gov/Archives/edgar/data/1.htm"))
        self.assertTrue(is_primary_source_url("https://github.com/langchain-ai/langgraph"))
        self.assertTrue(
            is_primary_source_url(
                "https://investor.nvidia.com/news/press-releases/default.aspx"
            )
        )
        self.assertFalse(is_primary_source_url("https://seo-market.example/report/ai-2026"))
        self.assertFalse(is_primary_source_url("https://www.example.com/data/blog-post"))
        self.assertFalse(is_primary_source_url("https://www.facebook.com/story.php"))
        self.assertTrue(
            is_primary_source_url("https://www.example.com/investor/annual-reports/2025/")
        )

    def test_serp_demotes_seo_market_report_path(self) -> None:
        seo = serp_url_quality_adjustment(
            "https://www.example.com/market-research/ai-chips-report-2026"
        )
        ir = serp_url_quality_adjustment(
            "https://investor.nvidia.com/news/press-releases/default.aspx"
        )
        self.assertGreater(ir, seo)


if __name__ == "__main__":
    unittest.main()
