# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Google Programmable Search (Custom Search JSON API)."""
from __future__ import annotations

from finkey_web.executor import WebSearchExecutor
from finkey_web.schema import SearchResult


class GoogleCSEBackend:
    """Thin adapter around ``WebSearchExecutor``."""

    def __init__(self, executor: WebSearchExecutor | None = None) -> None:
        self._exe = executor or WebSearchExecutor()

    @property
    def name(self) -> str:
        return "google_cse"

    def search(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        date_restrict: str,
    ) -> list[SearchResult]:
        return self._exe.search(
            query,
            max_results=max_results,
            language=language,
            date_restrict=date_restrict,
        )
