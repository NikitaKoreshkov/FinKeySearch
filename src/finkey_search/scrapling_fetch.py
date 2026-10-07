# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Scrapling-based HTTP page fetcher.

Tier 2 of page enrichment (after Playwright, before plain httpx).
Advantages over plain HTTP:
  • TLS fingerprint spoofing via curl-cffi → bypasses many anti-bot filters
  • Scrapling Selector gives clean get_all_text() — strips scripts/styles/nav
  • Much faster than a full Playwright browser
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_NOISE_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript")

_BLOCK_PHRASES = (
    "access denied",
    "403 forbidden",
    "cloudflare to continue",
    "please verify you are a human",
    "enable javascript",
    "browser doesn't support javascript",
)


def _is_blocked(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _BLOCK_PHRASES) and len(text) < 2000


def fetch_page_text_scrapling(
    url: str,
    *,
    timeout_s: float = 20.0,
    max_chars: int = 65536,
) -> tuple[str, Optional[str]]:
    """
    Fetch ``url`` with Scrapling's TLS-fingerprint-spoofing HTTP client
    and return ``(clean_text, error_or_None)``.

    Falls back to returning ("", error_message) on any failure.
    """
    try:
        from scrapling.fetchers import Fetcher
        from scrapling.parser import Selector
    except ImportError:
        return "", "scrapling not installed"

    try:
        page = Fetcher(auto_match=False).get(
            url,
            timeout=timeout_s,
            stealthy_headers=True,
            follow_redirects=True,
        )
    except Exception as exc:
        return "", f"scrapling_fetch error: {exc}"

    if page is None:
        return "", "scrapling_fetch: no response"

    try:
        text = page.get_all_text(ignore_tags=_NOISE_TAGS, separator="\n")
    except Exception as exc:
        try:
            raw_html = getattr(page, "html_content", None) or str(page)
            text = re.sub(r"<[^>]+>", " ", raw_html)
            text = re.sub(r"\s{3,}", "\n\n", text).strip()
        except Exception:
            return "", f"scrapling text extraction error: {exc}"

    text = text.strip()
    if not text or _is_blocked(text):
        return "", "scrapling_fetch: blocked or empty"

    if max_chars > 0:
        text = text[:max_chars]

    return text, None
