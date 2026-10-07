# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Strip noisy patterns from search snippets before synthesis / reranking."""
from __future__ import annotations

import re

from finkey_search.schema import SearchResult

_SPAM_FRAGMENTS = (
    "click here",
    "subscribe now",
    "buy now",
    "casino",
    "cialis",
    "viagra",
    "выигра",
    "заработай",
    "перейди по ссылке",
)

_MAX_URL_RUN = 120


def sanitize_search_results(results: list[SearchResult]) -> None:
    """Mutates ``snippet`` / ``title`` in place — best-effort corporate-safe cleanup."""
    for r in results:
        if r.snippet:
            r.snippet = _clean_blob(r.snippet)
        if r.title:
            r.title = _clean_blob(r.title, title=True)
        if r.enriched_text:
            r.enriched_text = _clean_blob(r.enriched_text)


def _clean_blob(text: str, title: bool = False) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = re.sub(r"https?://\S+", lambda m: m.group(0)[:_MAX_URL_RUN], s)
    s = re.sub(r"\s+", " ", s).strip()
    low = s.lower()
    if any(sp in low for sp in _SPAM_FRAGMENTS):
        return ""
    if title:
        return s[:240]
    return s[:8000]
