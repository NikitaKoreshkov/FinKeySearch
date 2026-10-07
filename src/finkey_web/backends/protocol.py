# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Abstract search backends — Bing Web Search, Tavily, legacy Google CSE, custom indexes."""
from __future__ import annotations

from typing import Protocol

from finkey_web.schema import SearchResult


class SearchBackend(Protocol):
    """Pluggable web retrieval: returns normalized ``SearchResult`` rows."""

    @property
    def name(self) -> str:
        ...

    def search(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        date_restrict: str,
    ) -> list[SearchResult]:
        ...
