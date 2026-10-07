import unittest

from finkey_search.visual_media_intent import (
    augment_search_query_for_media,
    detect_visual_media_intent,
)


class TestVisualMediaIntent(unittest.TestCase):
    def test_detect_video_ru(self) -> None:
        self.assertEqual(
            detect_visual_media_intent("найди видео про инфляцию на ютубе"),
            "video",
        )

    def test_detect_image_en(self) -> None:
        self.assertEqual(
            detect_visual_media_intent("stock photo of kazakhstan flag on freepik"),
            "image",
        )

    def test_detect_mixed(self) -> None:
        self.assertEqual(
            detect_visual_media_intent("youtube tutorial and freepik banner"),
            "mixed",
        )

    def test_augment_tiktok_prefers_site(self) -> None:
        q = augment_search_query_for_media("cats dance", "тикток коты", "video")
        self.assertIn("site:tiktok.com", q.lower())

    def test_augment_default_video_adds_youtube(self) -> None:
        q = augment_search_query_for_media(
            "inflation explained", "find a video", "video"
        )
        low = q.lower()
        self.assertTrue(
            "site:youtube.com" in low or "youtu.be" in low,
            msg=q,
        )


if __name__ == "__main__":
    unittest.main()
