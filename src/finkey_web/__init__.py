# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey Web Search — intelligent internet awareness.

LLM-first classifier (`llm_primary`/`llm_only`), fallback backends (Serper, Brave, Bing, Tavily),
cross-check for sensitive quotes, adaptive page fetch, sanitization,
Prometheus-style metrics / optional OTEL hooks.
"""
from .backends import (
    BingWebSearchBackend,
    FallbackSearchChain,
    GoogleCSEBackend,
    GuardedBackend,
    SearchBackend,
    TavilySearchBackend,
)
from .classifier import SearchNeedClassifier
from .engine import WebSearchEngine
from .executor import WebSearchExecutor
from .metrics import InternetMetrics, InternetMetricsSnapshot
from .page_enrichment import adaptive_enrich_budget, enrich_results_with_structured_pages
from .reranker import SearchReranker
from .snippet_sanitize import sanitize_search_results
from .resilience import CircuitBreaker, SlidingWindowLimiter
from .schema import (
    SearchCategory,
    SearchContext,
    SearchDecision,
    SearchResult,
    SearchTier,
)
from .research_orchestrator import (
    orchestrate_search,
    should_orchestrate,
    orchestrator_enabled,
)
from .synthesizer import ResultSynthesizer

__all__ = [
    "WebSearchEngine",
    "SearchNeedClassifier",
    "WebSearchExecutor",
    "ResultSynthesizer",
    "SearchBackend",
    "BingWebSearchBackend",
    "GoogleCSEBackend",
    "TavilySearchBackend",
    "FallbackSearchChain",
    "GuardedBackend",
    "InternetMetrics",
    "InternetMetricsSnapshot",
    "SearchReranker",
    "CircuitBreaker",
    "SlidingWindowLimiter",
    "SearchDecision",
    "SearchResult",
    "SearchContext",
    "SearchTier",
    "SearchCategory",
    "adaptive_enrich_budget",
    "enrich_results_with_structured_pages",
    "sanitize_search_results",
    "orchestrate_search",
    "should_orchestrate",
    "orchestrator_enabled",
]
