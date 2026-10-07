# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Detect when the user wants discoverable video / stock imagery / mood boards.

Used to relax operator URL blocks, boost SERP scores for media hosts, and
nudge the search query toward site-scoped retrieval (without hard-coding full SERP logic).
"""
from __future__ import annotations

import re
from typing import Literal, Optional

VisualMediaKind = Literal["video", "image", "mixed"]

_VIDEO_MARKERS = re.compile(
    r"(youtube|youtu\.be|ютуб|rutube|рутуб|vimeo|tiktok|тикток|"
    r"видео|ролик|клип|трейлер|trailer|live\s*stream|стрим|watch\s+on|найди\s+видео|"
    r"подборка\s+видео|video\s+about|youtube\s+search)",
    re.I,
)

_IMAGE_MARKERS = re.compile(
    r"(freepik|фрипик|pinterest|пинтерест|unsplash|shutterstock|pixabay|depositphotos|"
    r"istock|adobe\s*stock|getty|стоков|stock\s*photo|stock\s*image|фото\s+на|картинк|"
    r"изображени|иллюстрац|обои|wallpaper|moodboard|референс|reference\s+photo|"
    r"найди\s+фото|поиск\s+фото|stock\s+footage)",
    re.I,
)


def detect_visual_media_intent(message: str) -> Optional[VisualMediaKind]:
    if not (message or "").strip():
        return None
    v = bool(_VIDEO_MARKERS.search(message))
    i = bool(_IMAGE_MARKERS.search(message))
    if v and i:
        return "mixed"
    if v:
        return "video"
    if i:
        return "image"
    return None


def augment_search_query_for_media(
    search_query: str,
    message: str,
    kind: Optional[VisualMediaKind],
) -> str:
    """
    Append site hints so SERP leans toward the right verticals.
    Keeps the classifier-optimized query as prefix to avoid losing intent.
    """
    if not kind:
        return search_query
    q = (search_query or "").strip()
    low = (q + " " + message).lower()
    extras: list[str] = []
    if kind in ("video", "mixed"):
        if "tiktok" in low or "тикток" in low:
            if "site:tiktok.com" not in low:
                extras.append("(site:tiktok.com)")
        elif "rutube" in low or "рутуб" in low:
            if "site:rutube.ru" not in low:
                extras.append("(site:rutube.ru)")
        elif "vimeo" in low:
            if "site:vimeo.com" not in low:
                extras.append("(site:vimeo.com)")
        elif "site:youtube.com" not in low and "youtu.be" not in low:
            extras.append("(site:youtube.com OR site:youtu.be)")
    if kind in ("image", "mixed"):
        if "freepik" not in low and "site:freepik.com" not in low:
            extras.append("(site:freepik.com OR site:pinterest.com OR site:unsplash.com)")
    if not extras:
        return q
    return f"{q} {' '.join(extras)}".strip()
