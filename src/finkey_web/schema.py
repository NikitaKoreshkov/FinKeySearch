# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Web Search Schema — all types for the intelligent search layer.

FinKey's web search works in 3 tiers:
  Tier 1 (< 1ms)   — fast heuristic "definitely skip"
  Tier 2 (< 1ms)   — fast heuristic "definitely search"
  Tier 3 (50-100ms) — LLM classifier for ambiguous queries

The result feeds into the system prompt as a "LIVE WEB DATA" block.
FinKey then uses this data naturally — she doesn't say "I found on Google",
she just knows and cites the source if asked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SearchTier(str, Enum):
    HEURISTIC_NO   = "heuristic_no"
    HEURISTIC_YES  = "heuristic_yes"
    LLM_CLASSIFIED = "llm_classified"


class SearchCategory(str, Enum):
    CURRENCY_RATE     = "currency_rate"
    CRYPTO_PRICE      = "crypto_price"
    STOCK_PRICE       = "stock_price"
    INTEREST_RATE     = "interest_rate"
    FINANCIAL_NEWS    = "financial_news"
    ECONOMIC_DATA     = "economic_data"
    COMPANY_INFO      = "company_info"
    REGULATORY        = "regulatory"
    PRODUCT_RATES     = "product_rates"
    GENERAL_FINANCE   = "general_finance"
    GENERAL           = "general"
    BROWSER_ACTION    = "browser_action"


class SearchDepth(str, Enum):
    """
    How much retrieval work a search needs — decided by the AI classifier, not heuristics.

    FAST  — quick lookup: SERP + light page enrichment only, no cross-check, no browsing
            operator. For single current facts (a price/rate/score, one headline) where
            search snippets already contain the answer. Optimised for time-to-first-token.
    DEEP  — full research: adaptive multi-page enrichment + browsing operator fallback.
            For multi-source synthesis, reports, comparisons, or when full articles matter.
    """
    FAST = "fast"
    DEEP = "deep"


@dataclass
class SearchDecision:
    """Result of the classifier: should we search? what for?"""
    should_search:   bool
    tier:            SearchTier             = SearchTier.HEURISTIC_NO
    category:        Optional[SearchCategory] = None
    search_query:    str                    = ""
    original_intent: str                   = ""
    confidence:      float                 = 1.0
    reason:          str                   = ""
    verified_sources_required: bool = False
    browser_action_url: str = ""
    search_depth:    SearchDepth            = SearchDepth.DEEP


@dataclass
class SearchResult:
    """A single search result (one page/snippet)."""
    title:       str
    url:         str
    snippet:     str
    date:        Optional[str]  = None
    source:      str            = ""
    relevance:   float          = 1.0
    enriched_text: Optional[str] = None


@dataclass
class SearchContext:
    """
    What gets injected into the system prompt.
    Clean, factual, dated. FinKey uses it as if she already knew it.
    """
    found:           bool
    query_used:      str
    category:        Optional[SearchCategory]
    verified_sources_required: bool = False
    citation_urls:   list[str]            = field(default_factory=list)
    results:         list[SearchResult]   = field(default_factory=list)
    synthesized:     str                  = ""
    search_date:     str                  = ""
    total_results:   int                  = 0

    def is_useful(self) -> bool:
        return self.found and bool(self.synthesized.strip())
