# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
WebSearchEngine — the master orchestrator of FinKey's internet awareness.

Pipeline:
  1. SearchNeedClassifier   — tiers 1–3 + optional dialogue context
  2. FallbackSearchChain    — Serper / Brave / Bing / Tavily (when configured)
     Stealth SERP fallback if the chain returns empty and stealth is enabled
  3. SearchReranker         — heuristic ordering (+ optional LLM trim)
  4. Structured page fetch  — many top HTTPS hits via Playwright extraction (large defaults)
  5. ResultSynthesizer      — dated knowledge block for the system prompt
  6. Optional BrowsingOperator — multi-step browsing when explicitly enabled

Design principles:
  • Graceful degradation — missing keys, limits, or Playwright never crash FinKey
  • Conversation-aware classifier tail + repetition guard + TTL cache
  • Observability via ``internet_metrics``
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import replace
from typing import Callable, Optional

from finkey_search.url_policy import URLPolicy
from finkey_search.progress import emit_pipeline_stage, emit_web_progress

from .backends import (
    BingWebSearchBackend,
    BraveSearchBackend,
    FallbackSearchChain,
    GuardedBackend,
    SerperSearchBackend,
    TavilySearchBackend,
)
from .classifier import SearchNeedClassifier
from .metrics import InternetMetrics
from .page_enrichment import adaptive_enrich_budget, enrich_results_with_structured_pages


def _web_profile_defaults() -> dict[str, int]:
    """
    ``FINKEY_WEB_PROFILE``:
      • ``balanced`` (default) — ChatGPT-like: достаточно SERP + enrich без минут ожидания
      • ``light`` — минимальный latency, тонкий контекст
      • ``full`` / ``heavy`` / ``max`` — исследовательский режим
    Явные ``FINKEY_WEB_*`` env всегда перекрывают профиль.
    """
    profile = (os.getenv("FINKEY_WEB_PROFILE") or "balanced").strip().lower()
    if profile in ("full", "heavy", "max"):
        return {
            "max_results": 20,
            "page_enrich_n": 12,
            "page_enrich_max_chars": 500_000,
            "search_max_per_conv": 48,
            "rerank_llm_min_pool": 7,
            "session_memory_total": 24,
        }
    if profile in ("light", "fast", "minimal"):
        return {
            "max_results": 8,
            "page_enrich_n": 4,
            "page_enrich_max_chars": 24_000,
            "search_max_per_conv": 8,
            "rerank_llm_min_pool": 5,
            "session_memory_total": 10,
        }
    # balanced — default
    return {
        "max_results": 12,
        "page_enrich_n": 8,
        "page_enrich_max_chars": 96_000,
        "search_max_per_conv": 16,
        "rerank_llm_min_pool": 6,
        "session_memory_total": 16,
    }


def _env_int_or_profile(name: str, profile_key: str, profile: dict[str, int]) -> int:
    raw = (os.getenv(name) or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return int(profile[profile_key])


def _fast_enrich_cap() -> int:
    """Max pages enriched on the AI-chosen FAST path (no adaptive raise)."""
    try:
        v = int(os.getenv("FINKEY_WEB_FAST_ENRICH_MAX", "6"))
    except ValueError:
        v = 6
    return max(0, min(v, 10))


def _fast_enrich_timeout_ms() -> int:
    """
    Per-page fetch timeout on the FAST path.

    Kept short on purpose: the global enrich deadline (see page_enrichment) is the
    real budget. A hung host must not burn the whole wave — abandon and use the
    pages that already answered. Floor is 1s so TLS+TTFB still has a chance.
    """
    try:
        v = int(os.getenv("FINKEY_WEB_FAST_ENRICH_TIMEOUT_MS", "2000"))
    except ValueError:
        v = 2000
    return max(1_000, min(v, 8_000))


def _operator_on_deep_always() -> bool:
    """Run the browsing operator on every DEEP search even with a useful SERP (legacy)."""
    raw = (os.getenv("FINKEY_BROWSING_OPERATOR_ON_DEEP", "0") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")
from .reranker import SearchReranker
from .resilience import CircuitBreaker, SlidingWindowLimiter
from .schema import (
    SearchCategory,
    SearchContext,
    SearchDecision,
    SearchDepth,
    SearchResult,
    SearchTier,
)
from .synthesizer import ResultSynthesizer
from .url_quality import is_blocked_for_operator_browse, operator_entry_priority_score

logger = logging.getLogger(__name__)

_CROSSCHECK_CATEGORIES = frozenset(
    {
        SearchCategory.CURRENCY_RATE,
        SearchCategory.CRYPTO_PRICE,
        SearchCategory.STOCK_PRICE,
        SearchCategory.INTEREST_RATE,
        SearchCategory.FINANCIAL_NEWS,
    }
)


class _CacheEntry:
    def __init__(self, ctx: SearchContext, ttl: float = 300):
        self.ctx      = ctx
        self._expires = time.monotonic() + ttl

    def is_valid(self) -> bool:
        return time.monotonic() < self._expires


class WebSearchEngine:
    """
    One instance per session (or shared with TTL cache).

    Environment shortcuts:
      • ``FINKEY_SEARCH_CLASSIFIER_MODE`` — ``llm_only`` / ``llm_primary`` / ``legacy``
      • ``FINKEY_SEARCH_CROSSCHECK`` — ``0`` disables second-pass query for market-sensitive categories
      • ``FINKEY_SEARCH_BING_MKT`` — рынок Bing (например ``ru-RU``); иначе из ``lr`` сообщения
      • ``FINKEY_WEB_PAGE_ENRICH`` — ``0`` disables structured page enrichment
      • ``FINKEY_WEB_PAGE_ENRICH_MAX`` — baseline pages to enrich (adaptive budget may raise it; cap via ``FINKEY_WEB_PAGE_ENRICH_ADAPTIVE_MAX``)
      • ``FINKEY_WEB_FAST_ENRICH_MAX`` — pages enriched on the AI-chosen FAST path (default ``2``; no adaptive raise)
      • ``FINKEY_WEB_FAST_ENRICH_TIMEOUT_MS`` — per-page Playwright timeout on FAST (default ``9000``; FAST also skips the Scrapling/HTTP retry tail and the browsing operator unless SERP yields nothing useful)
      • ``FINKEY_WEB_SEARCH_MAX_RESULTS`` — hits kept after rerank (default raised; max 48)
      • ``FINKEY_PAGE_ENRICH_BROWSER_TIMEOUT_MS`` — Playwright goto timeout для enrich (по умолч. ``38000``)
      • ``FINKEY_PAGE_ENRICH_NAV_WAIT_UNTIL`` — ``domcontentloaded`` / ``load`` / ``commit`` / ``networkidle`` (по умолч. ``domcontentloaded``)
      • ``FINKEY_PAGE_ENRICH_HTTP_TIMEOUT_S`` — таймаут HTTPS fallback после structured miss (иначе считается от browser timeout)
      • ``FINKEY_SEARCH_RERANK_LLM`` — ``1`` enables LLM rerank for large pools
      • ``FINKEY_SEARCH_BING_PER_MINUTE`` (fallback: ``FINKEY_SEARCH_CSE_PER_MINUTE``) / ``FINKEY_SEARCH_TAVILY_PER_MINUTE`` — soft limits
      • ``FINKEY_SEARCH_MAX_PER_CONVERSATION`` — cap on fresh searches per dialog key after classifier/cache miss (``<=0`` = unlimited); TTL cache hits do not consume it
      • ``TAVILY_API_KEY`` — optional second retrieval hop
      • ``REDIS_URL`` + ``FINKEY_WEB_SEARCH_REDIS_CACHE`` — cross-process cache for identical queries (see ``websearch/redis_cache.py``)
    """

    def __init__(
        self,
        bing_subscription_key: str = "",
        max_results: int = 20,
        cache_ttl:   float = 300,
        enable_stealth_fallback: bool = True,
        enable_browsing_operator: Optional[bool] = None,
        operator_max_steps: Optional[int] = None,
        operator_allowlist_suffixes: Optional[list[str]] = None,
        search_chain: Optional[FallbackSearchChain] = None,
    ):
        self._classifier  = SearchNeedClassifier()
        self._synthesizer = ResultSynthesizer()
        _profile = _web_profile_defaults()
        raw_mr = (os.getenv("FINKEY_WEB_SEARCH_MAX_RESULTS") or "").strip()
        if raw_mr:
            try:
                max_results = int(raw_mr)
            except ValueError:
                pass
        else:
            max_results = _profile["max_results"]
        self._max_results = max(1, min(int(max_results), 48))
        self._cache_ttl   = cache_ttl
        self._cache: dict[str, _CacheEntry] = {}
        self._enable_stealth = enable_stealth_fallback

        self._bing_key = bing_subscription_key
        self._metrics = InternetMetrics()
        self._chain   = search_chain or self._build_fallback_chain()

        try:
            llm_min = _env_int_or_profile(
                "FINKEY_SEARCH_RERANK_LLM_MIN_POOL", "rerank_llm_min_pool", _profile
            )
        except ValueError:
            llm_min = _profile["rerank_llm_min_pool"]
        self._reranker = SearchReranker(llm_min_pool=max(4, llm_min))
        self._rerank_llm = os.environ.get("FINKEY_SEARCH_RERANK_LLM", "1").strip().lower() in (
            "1", "true", "yes", "on",
        )

        pe = os.environ.get("FINKEY_WEB_PAGE_ENRICH", "1").strip().lower()
        self._page_enrich = pe not in ("0", "false", "no", "off")
        self._page_enrich_n = _env_int_or_profile(
            "FINKEY_WEB_PAGE_ENRICH_MAX", "page_enrich_n", _profile
        )

        raw_mc = _env_int_or_profile(
            "FINKEY_WEB_PAGE_ENRICH_MAX_CHARS", "page_enrich_max_chars", _profile
        )
        self._page_enrich_max_chars = max(4_000, min(raw_mc, 500_000))

        if enable_browsing_operator is None:
            raw_op = os.environ.get("FINKEY_ENABLE_BROWSING_OPERATOR", "1").strip().lower()
            enable_browsing_operator = raw_op not in ("0", "false", "no", "off")
        self._enable_browsing_operator = bool(enable_browsing_operator)

        if operator_max_steps is not None:
            self._operator_max_steps = max(4, min(int(operator_max_steps), 80))
        else:
            try:
                raw_ms = int(os.environ.get("FINKEY_BROWSING_OPERATOR_MAX_STEPS", "35"))
            except ValueError:
                raw_ms = 35
            self._operator_max_steps = max(4, min(raw_ms, 80))

        if operator_allowlist_suffixes is not None:
            suff = [s.strip().lower() for s in operator_allowlist_suffixes if s.strip()]
            self._operator_allowlist: Optional[list[str]] = suff or None
        else:
            self._operator_allowlist = self._parse_operator_allowlist_env()

        self._conv_last_search_fp: dict[str, tuple[str, int]] = {}
        self._min_turns_between_same = 4
        self._conv_search_counts: dict[str, int] = {}
        self._search_max_per_conv = _env_int_or_profile(
            "FINKEY_SEARCH_MAX_PER_CONVERSATION", "search_max_per_conv", _profile
        )
        self._session_memory_total = _env_int_or_profile(
            "FINKEY_WEB_SESSION_MEMORY_TOTAL", "session_memory_total", _profile
        )

        self._crosscheck_enabled_flag = os.getenv(
            "FINKEY_SEARCH_CROSSCHECK", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        self._session_snippets: dict[str, list[str]] = {}

    @staticmethod
    def _skipped_context(decision: SearchDecision) -> SearchContext:
        return SearchContext(
            found=False,
            query_used="",
            category=None,
            verified_sources_required=decision.verified_sources_required,
            citation_urls=[],
        )

    def clear_session_snippets(self, conversation_key: str) -> None:
        self._session_snippets.pop(conversation_key or "", None)

    @staticmethod
    def _apply_portal_query_rewrite(message: str, decision: SearchDecision) -> SearchDecision:
        """Bias SERP toward official ticketing domains when a portal crawl is intended."""
        if not decision.should_search:
            return decision
        try:
            from .portal_browse import augment_classifier_query_for_portals
        except ImportError:
            return decision
        new_q = augment_classifier_query_for_portals(message, decision.search_query)
        if new_q != decision.search_query:
            logger.info(
                "Portal SERP query rewrite: %r -> %r",
                (decision.search_query or "")[:120],
                (new_q or "")[:120],
            )
            return replace(decision, search_query=new_q)
        return decision

    def _duplicate_query_cooldown(
        self,
        conversation_key: Optional[str],
        search_query: str,
        turn_number: int,
    ) -> bool:
        ck = (conversation_key or "").strip()
        qfp = (search_query or "").lower().strip()[:240]
        if not ck or not qfp:
            return False
        prev_fp, prev_turn = self._conv_last_search_fp.get(ck, ("", -(10**9)))
        return qfp == prev_fp and (turn_number - prev_turn) < self._min_turns_between_same

    def _remember_query_fingerprint(
        self,
        conversation_key: Optional[str],
        search_query: str,
        turn_number: int,
    ) -> None:
        ck = (conversation_key or "").strip()
        qfp = (search_query or "").lower().strip()[:240]
        if ck and qfp:
            self._conv_last_search_fp[ck] = (qfp, turn_number)

    def _prepend_session_snippets(
        self, conversation_key: Optional[str], ctx: SearchContext
    ) -> None:
        if not conversation_key or not (ctx.synthesized or "").strip():
            return
        try:
            n = int(os.getenv("FINKEY_WEB_SESSION_PREVIEW_LINES", "8"))
        except ValueError:
            n = 8
        n = max(2, min(n, 24))
        blob = self._session_snippets.get(conversation_key, [])
        if not blob:
            return
        head = (
            "\n--- SESSION_WEB_MEMORY (reuse only if relevant; cite URLs verbatim) ---\n"
            + "\n".join(blob[-n:])
            + "\n--------------------------------------------------------------------\n\n"
        )
        ctx.synthesized = head + ctx.synthesized

    def _remember_session_snippets(
        self, conversation_key: Optional[str], ctx: SearchContext
    ) -> None:
        if not conversation_key or not ctx.is_useful():
            return
        bucket = self._session_snippets.setdefault(conversation_key, [])
        cap = max(5, min(self._session_memory_total, 80))
        for r in ctx.results[:16]:
            u = (r.url or "").strip()
            excerpt = ((r.enriched_text or r.snippet or "")[:720]).strip()
            if not u.startswith("https://") or len(excerpt) < 40:
                continue
            line = f"• [{r.source}] {excerpt}\n  {u}"
            if line not in bucket[-cap:]:
                bucket.append(line)
        bucket[:] = bucket[-cap:]

    def _decide_search(
        self,
        message: str,
        turn_number: int,
        *,
        force_search: bool,
        allow_llm_tier: bool,
        generate_fn: Optional[Callable[[str], str]],
        conversation_messages: Optional[list[str]],
    ) -> SearchDecision:
        if force_search:
            from .classifier import _optimize_query

            qq = _optimize_query(message.strip(), SearchCategory.GENERAL).strip()[:500]
            if not qq:
                qq = message.strip()[:400]
            return SearchDecision(
                should_search=True,
                tier=SearchTier.LLM_CLASSIFIED,
                category=SearchCategory.GENERAL,
                search_query=qq,
                original_intent="FINKEY_WEB_MODE=always",
                reason="force_search",
                confidence=1.0,
                verified_sources_required=False,
            )
        return self._classifier.classify(
            message=message,
            turn_number=turn_number,
            allow_llm_tier=allow_llm_tier,
            generate_fn=generate_fn,
            conversation_messages=conversation_messages,
        )

    def _log_classifier_decision(self, decision: SearchDecision) -> None:
        depth_val = getattr(decision, "search_depth", SearchDepth.DEEP)
        depth_str = depth_val.value if isinstance(depth_val, SearchDepth) else str(depth_val)
        logger.info(
            "Search decision: should_search=%s depth=%s verified_sources=%s category=%s reason=%s query=%r",
            decision.should_search,
            depth_str,
            decision.verified_sources_required,
            decision.category,
            decision.reason,
            decision.search_query[:120],
        )
        logger.info(
            "classifier_structured %s",
            json.dumps(
                {
                    "should_search":              decision.should_search,
                    "depth":                     depth_str,
                    "verified_sources_required": decision.verified_sources_required,
                    "category":                  decision.category.value if decision.category else None,
                    "query":                     decision.search_query[:240],
                    "reason":                    (decision.reason or "")[:280],
                },
                ensure_ascii=False,
            ),
        )


    @property
    def internet_metrics(self) -> InternetMetrics:
        return self._metrics

    def _build_fallback_chain(self) -> FallbackSearchChain:
        metrics = self._metrics

        try:
            serper_pm = int(os.getenv("FINKEY_SEARCH_SERPER_PER_MINUTE", "60"))
        except ValueError:
            serper_pm = 60
        try:
            brave_pm = int(os.getenv("FINKEY_SEARCH_BRAVE_PER_MINUTE", "48"))
        except ValueError:
            brave_pm = 48
        try:
            bing_pm = int(
                os.getenv(
                    "FINKEY_SEARCH_BING_PER_MINUTE",
                    os.getenv("FINKEY_SEARCH_CSE_PER_MINUTE", "48"),
                )
            )
        except ValueError:
            bing_pm = 48
        try:
            tv_pm = int(os.getenv("FINKEY_SEARCH_TAVILY_PER_MINUTE", "36"))
        except ValueError:
            tv_pm = 36

        layers: list[GuardedBackend] = []

        serper = SerperSearchBackend()
        if serper.configured():
            lim_serper = SlidingWindowLimiter(max_calls=max(1, serper_pm), window_seconds=60.0)
            br_serper = CircuitBreaker()

            def guard_serper() -> bool:
                return lim_serper.acquire() and br_serper.allow_request()

            layers.append(
                GuardedBackend(
                    inner=serper,
                    allow=guard_serper,
                    on_call=lambda n: metrics.record_backend_call(n),
                    on_success=lambda n, c: (
                        metrics.record_backend_success(n, c),
                        br_serper.record_success(),
                    ),
                    on_failure=lambda n, e: (
                        metrics.record_backend_error(n),
                        br_serper.record_failure(),
                    ),
                )
            )

        brave = BraveSearchBackend()
        if brave.configured():
            lim_brave = SlidingWindowLimiter(max_calls=max(1, brave_pm), window_seconds=60.0)
            br_brave = CircuitBreaker()

            def guard_brave() -> bool:
                return lim_brave.acquire() and br_brave.allow_request()

            layers.append(
                GuardedBackend(
                    inner=brave,
                    allow=guard_brave,
                    on_call=lambda n: metrics.record_backend_call(n),
                    on_success=lambda n, c: (
                        metrics.record_backend_success(n, c),
                        br_brave.record_success(),
                    ),
                    on_failure=lambda n, e: (
                        metrics.record_backend_error(n),
                        br_brave.record_failure(),
                    ),
                )
            )

        bing_inner = BingWebSearchBackend(subscription_key=self._bing_key)
        if bing_inner.configured():
            lim_bing = SlidingWindowLimiter(max_calls=max(1, bing_pm), window_seconds=60.0)
            br_bing = CircuitBreaker()

            def guard_bing() -> bool:
                return lim_bing.acquire() and br_bing.allow_request()

            layers.append(
                GuardedBackend(
                    inner=bing_inner,
                    allow=guard_bing,
                    on_call=lambda n: metrics.record_backend_call(n),
                    on_success=lambda n, c: (
                        metrics.record_backend_success(n, c),
                        br_bing.record_success(),
                    ),
                    on_failure=lambda n, e: (
                        metrics.record_backend_error(n),
                        br_bing.record_failure(),
                    ),
                )
            )

        tv = TavilySearchBackend()
        if tv.configured():
            lim_tv = SlidingWindowLimiter(max_calls=max(1, tv_pm), window_seconds=60.0)
            br_tv  = CircuitBreaker()

            def guard_tv() -> bool:
                return lim_tv.acquire() and br_tv.allow_request()

            layers.append(
                GuardedBackend(
                    inner=tv,
                    allow=guard_tv,
                    on_call=lambda n: metrics.record_backend_call(n),
                    on_success=lambda n, c: (
                        metrics.record_backend_success(n, c),
                        br_tv.record_success(),
                    ),
                    on_failure=lambda n, e: (
                        metrics.record_backend_error(n),
                        br_tv.record_failure(),
                    ),
                )
            )

        return FallbackSearchChain(layers)

    def _search_policy(self) -> URLPolicy:
        return URLPolicy(
            allowed_host_suffixes=self._operator_allowlist,
            block_private_and_loopback=True,
        )

    def _conversation_budget_ok(self, conversation_key: Optional[str]) -> bool:
        if not conversation_key:
            return True
        lim = self._search_max_per_conv
        if lim <= 0:
            return True
        return self._conv_search_counts.get(conversation_key, 0) < lim

    def _bump_conversation_search(self, conversation_key: Optional[str]) -> None:
        if not conversation_key:
            return
        self._conv_search_counts[conversation_key] = (
            self._conv_search_counts.get(conversation_key, 0) + 1
        )

    def reset_conversation_search_budget(self, conversation_key: str) -> None:
        self._conv_search_counts.pop(conversation_key, None)

    def _effective_cache_ttl(self, category: Optional[SearchCategory]) -> float:
        if self._cache_ttl <= 0:
            return 0.0
        base = float(self._cache_ttl)
        if category in (
            SearchCategory.CURRENCY_RATE,
            SearchCategory.CRYPTO_PRICE,
            SearchCategory.STOCK_PRICE,
        ):
            try:
                mx = float(os.getenv("FINKEY_SEARCH_CACHE_TTL_MARKETS", "120"))
            except ValueError:
                mx = 120.0
            return min(base, mx)
        if category == SearchCategory.FINANCIAL_NEWS:
            try:
                mx = float(os.getenv("FINKEY_SEARCH_CACHE_TTL_NEWS", "180"))
            except ValueError:
                mx = 180.0
            return min(base, mx)
        return base

    def _redis_cache_key(
        self, cache_scope: str, cache_key: str, lang: str, date_r: str
    ) -> str:
        from .redis_cache import make_redis_key

        return make_redis_key(
            cache_scope or "global",
            query_key=cache_key,
            lang=lang,
            date_restrict=date_r,
            max_results=self._max_results,
        )

    def _try_redis_cache_get(
        self,
        cache_scope: str,
        cache_key: str,
        lang: str,
        date_r: str,
        decision: SearchDecision,
    ) -> Optional[SearchContext]:
        from .redis_cache import redis_cache_enabled, redis_get_context
        from .redis_client import get_shared_redis_optional

        if not redis_cache_enabled():
            return None
        cli = get_shared_redis_optional()
        if cli is None:
            return None
        rk = self._redis_cache_key(cache_scope, cache_key, lang, date_r)
        ctx = redis_get_context(cli, rk)
        if ctx is None:
            return None
        self._metrics.record_redis_cache_hit()
        if self._cache_ttl > 0:
            ttl_eff = self._effective_cache_ttl(decision.category)
            if ttl_eff > 0:
                self._cache[cache_key] = _CacheEntry(ctx, ttl=ttl_eff)
        return ctx

    def _try_redis_cache_set(
        self,
        cache_scope: str,
        cache_key: str,
        lang: str,
        date_r: str,
        category: Optional[SearchCategory],
        ctx: SearchContext,
    ) -> None:
        from .redis_cache import (
            _ttl_default,
            redis_cache_enabled,
            redis_set_context,
        )
        from .redis_client import get_shared_redis_optional

        if not ctx.is_useful() or not redis_cache_enabled():
            return
        cli = get_shared_redis_optional()
        if cli is None:
            return
        rk = self._redis_cache_key(cache_scope, cache_key, lang, date_r)
        eff = int(self._effective_cache_ttl(category))
        cap_env = _ttl_default()
        ttl = min(cap_env, eff) if eff > 0 else cap_env
        ttl = max(30, ttl)
        redis_set_context(cli, rk, ctx, ttl_sec=ttl)

    @staticmethod
    def _merge_search_rows(a: list[SearchResult], b: list[SearchResult]) -> list[SearchResult]:
        seen: set[str] = set()
        out: list[SearchResult] = []
        for r in a + b:
            key = (r.url or "").strip().lower()
            if not key:
                key = ((r.snippet or "")[:72] + "|" + r.source).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _maybe_cross_check_with_flag(
        self,
        results: list[SearchResult],
        decision: SearchDecision,
        pool: int,
        lang: str,
        date_r: str,
    ) -> tuple[list[SearchResult], bool]:
        if (
            not self._crosscheck_enabled_flag
            or not results
            or decision.category not in _CROSSCHECK_CATEGORIES
        ):
            return results, False
        q = (decision.search_query or "").strip()
        if not q:
            return results, False
        alt = q + " official"
        try:
            extra = self._chain.search(
                alt,
                max_results=min(14, pool),
                language=lang,
                date_restrict=date_r,
            )
        except Exception as exc:
            logger.debug("Cross-check search skipped: %s", exc)
            return results, False
        if not extra:
            return results, False
        self._metrics.record_cross_check()
        return self._merge_search_rows(results, extra), True

    @staticmethod
    def _is_fast(decision: SearchDecision, require_full_articles: bool) -> bool:
        """
        FAST path = AI classifier chose ``depth=fast`` AND the turn does not demand full
        articles / verified sources. A strict/verified turn always uses the DEEP pipeline.
        """
        if require_full_articles:
            return False
        return getattr(decision, "search_depth", SearchDepth.DEEP) == SearchDepth.FAST

    @staticmethod
    def _ensure_enrich_url_rows(
        results: list[SearchResult],
        urls: list[str],
    ) -> list[SearchResult]:
        """Ensure allowlisted URLs exist as rows so selective enrich can fetch them."""
        from .page_enrichment import _norm_url_key

        have = {_norm_url_key(r.url or "") for r in results}
        out = list(results)
        for raw in urls:
            u = (raw or "").strip()
            if not u.startswith("https://"):
                continue
            key = _norm_url_key(u)
            if not key or key in have:
                continue
            have.add(key)
            out.insert(
                0,
                SearchResult(
                    title=u,
                    url=u,
                    snippet="(selected for deep enrich)",
                    source="enrich_allowlist",
                    relevance=1.1,
                ),
            )
        return out

    def _pipeline_after_fetch(
        self,
        results: list[SearchResult],
        decision: SearchDecision,
        pool: int,
        lang: str,
        date_r: str,
        generate_fn: Optional[Callable[[str], str]],
        require_full_articles: bool = False,
        message: str = "",
        *,
        serp_draft: bool = False,
        enrich_urls: Optional[list[str]] = None,
    ) -> tuple[list[SearchResult], Optional[str]]:
        from finkey_search.snippet_sanitize import sanitize_search_results

        fast = self._is_fast(decision, require_full_articles) or bool(serp_draft)

        if fast:
            did_cross = False
        else:
            results, did_cross = self._maybe_cross_check_with_flag(
                results, decision, pool, lang, date_r
            )
        sanitize_search_results(results)
        if results:
            self._metrics.record_snippet_sanitize(len(results))

        self._metrics.record_rerank()
        # FAST: heuristic rerank only — an extra LLM call does not pay off for a quick fact.
        results = self._reranker.rerank(
            results,
            decision.search_query,
            max_keep=self._max_results,
            generate_fn=generate_fn,
            use_llm=self._rerank_llm and not fast,
        )

        allowlist = [u.strip() for u in (enrich_urls or []) if (u or "").strip()]
        if allowlist:
            results = self._ensure_enrich_url_rows(results, allowlist)

        # Research quality floor: even on "draft", enrich a few high-quality URLs
        # so the model is not synthesizing long reports from titles alone.
        research_floor = 0
        if serp_draft and not allowlist:
            try:
                from .research_orchestrator import is_deep_research
                from .url_quality import serp_url_quality_adjustment

                if is_deep_research(message or ""):
                    raw = (os.getenv("FINKEY_WEB_RESEARCH_DRAFT_ENRICH") or "5").strip()
                    try:
                        research_floor = max(0, min(int(raw), 8))
                    except ValueError:
                        research_floor = 5
                    if research_floor > 0:
                        from .url_quality import is_primary_source_url

                        ranked = sorted(
                            results,
                            key=lambda r: (
                                1 if is_primary_source_url(r.url or "") else 0,
                                serp_url_quality_adjustment(r.url or ""),
                                float(r.relevance or 0),
                            ),
                            reverse=True,
                        )
                        picks: list[str] = []
                        # Always try to keep ≥1 primary URL in the enrich floor.
                        for r in ranked:
                            u = (r.url or "").strip()
                            if not u.startswith("https://"):
                                continue
                            if is_blocked_for_operator_browse(u):
                                continue
                            if is_primary_source_url(u):
                                picks.append(u)
                                break
                        for r in ranked:
                            u = (r.url or "").strip()
                            if not u.startswith("https://"):
                                continue
                            if u in picks:
                                continue
                            if is_blocked_for_operator_browse(u):
                                continue
                            if serp_url_quality_adjustment(u) < -0.5:
                                continue
                            picks.append(u)
                            if len(picks) >= research_floor:
                                break
                        if picks:
                            allowlist = picks
                            results = self._ensure_enrich_url_rows(results, allowlist)
            except Exception:
                research_floor = 0

        enrich_budget = 0
        research_draft_enrich = bool(serp_draft and allowlist and research_floor > 0)
        if serp_draft and not allowlist:
            # Pure SERP draft (simple facts): titles/snippets only.
            enrich_budget = 0
        elif serp_draft and allowlist:
            # Selective enrich (user urls=… or research quality floor).
            enrich_budget = min(len(results), len(allowlist), 8)
        elif self._page_enrich and results:
            if allowlist:
                enrich_budget = min(len(results), max(len(allowlist), 1))
            elif require_full_articles:
                enrich_budget = len(results)
            elif fast:
                enrich_budget = min(self._page_enrich_n, _fast_enrich_cap())
            else:
                enrich_budget = adaptive_enrich_budget(results, self._page_enrich_n)

        if self._page_enrich and results and enrich_budget > 0:
            max_chars = 0 if require_full_articles else self._page_enrich_max_chars
            if research_draft_enrich:
                # Keep research-draft enrich cheap: HTTP + FAST deadline.
                max_chars = min(max_chars or 24_000, 24_000)
            # FAST + balanced DEEP without require_full: HTTP enrich first (ChatGPT-like latency).
            # Full article / verified mode keeps browser enrich.
            fetch_mode = (
                "http"
                if (fast or research_draft_enrich or not require_full_articles)
                else "browser"
            )
            depth = getattr(decision, "search_depth", SearchDepth.DEEP)
            if not isinstance(depth, SearchDepth):
                depth = SearchDepth.DEEP
            if require_full_articles and not research_draft_enrich:
                depth = SearchDepth.DEEP
            if research_draft_enrich:
                depth = SearchDepth.FAST
            enrich_results_with_structured_pages(
                results,
                max_pages=enrich_budget,
                url_policy=self._search_policy(),
                max_chars_per_page=max_chars,
                metrics=self._metrics,
                browser_timeout_ms=_fast_enrich_timeout_ms()
                if (fast or research_draft_enrich)
                else None,
                http_fallback=False if (fast or research_draft_enrich) else True,
                fetch_mode=fetch_mode,
                message=message or "",
                depth=depth,
                # Full-article mode keeps ceiling reads; wave early-stop still ok for DEEP bar.
                early_stop=True,
                enrich_url_allowlist=allowlist or None,
            )
            if require_full_articles:
                before = len(results)
                results = [
                    r for r in results
                    if (r.enriched_text or "").strip() and (r.url or "").startswith("https://")
                ]
                logger.info(
                    "Strict article mode: kept enriched=%d/%d rows",
                    len(results),
                    before,
                )

        verification_note: Optional[str] = None
        if did_cross:
            verification_note = (
                "Выполнен дополнительный поиск по уточняющей формулировке запроса "
                "(суффикс «official» к исходному запросу). "
                "Если числовые значения расходятся между доменами — перечисли оба и укажи источники."
            )

        return results, verification_note

    def _run_gap_queries(
        self,
        message: str,
        results: list[SearchResult],
        queries: list[str],
        *,
        pool: int,
        lang: str,
        date_r: str,
        require_full: bool,
        label: str,
    ) -> tuple[list[SearchResult], str]:
        from .evidence_gate import filter_place_relevant
        from .research_orchestrator import SubAgentPlan, merge_agent_results, run_search_agents

        plans = [
            SubAgentPlan(agent_id=f"g{i+1}", role="gap_fill", query=q)
            for i, q in enumerate(queries)
        ]
        agent_results = run_search_agents(
            plans,
            self._chain.search,
            max_results=max(8, min(pool, 16)),
            language=lang,
            date_restrict=date_r,
        )
        extra, gap_note = merge_agent_results(agent_results, stamp_agents=True)
        before_urls = {(r.url or "").split("#")[0].rstrip("/").lower() for r in results if r.url}
        results = self._merge_search_rows(results, extra)
        results, place_note2 = filter_place_relevant(message, results)
        after_urls = {(r.url or "").split("#")[0].rstrip("/").lower() for r in results if r.url}
        new_n = len(after_urls - before_urls)

        if new_n > 0 and self._page_enrich:
            from .page_enrichment import enrich_results_with_structured_pages

            need = [r for r in results if not (r.enriched_text or "").strip()]
            enrich_n = min(len(need), max(4, min(10, self._page_enrich_n)))
            if enrich_n > 0:
                results = need + [r for r in results if (r.enriched_text or "").strip()]
                enrich_results_with_structured_pages(
                    results,
                    max_pages=enrich_n,
                    url_policy=self._search_policy(),
                    max_chars_per_page=self._page_enrich_max_chars,
                    metrics=self._metrics,
                    http_fallback=True,
                    fetch_mode="http" if not require_full else "browser",
                    message=message or "",
                    depth=SearchDepth.DEEP,
                    early_stop=True,
                )

        note = f"{label}: +{new_n} URLs\n{gap_note}"
        if place_note2:
            note = note + "\n" + place_note2
        return results, note

    def _maybe_gap_followup(
        self,
        message: str,
        results: list[SearchResult],
        decision: SearchDecision,
        *,
        deep: bool,
        use_orch: bool,
        pool: int,
        lang: str,
        date_r: str,
        generate_fn: Optional[Callable[[str], str]],
        require_full: bool,
        ver_note: Optional[str],
        force: bool = False,
    ) -> tuple[list[SearchResult], Optional[str]]:
        """
        Reflection pass: if evidence is thin (or force), run follow-up SERP.
        Runs for easy facts too when numbers/place coverage is weak.
        """
        from .evidence_gate import assess_evidence, gap_followup_enabled, plan_gap_queries

        if not gap_followup_enabled():
            return results, ver_note

        report = assess_evidence(message, results)
        if not report.thin and not force:
            logger.info(
                "Evidence OK: enriched=%d numeric=%d place_hits=%d hosts=%d",
                report.enriched_chars,
                report.numeric_hits,
                report.place_token_hits,
                report.unique_hosts,
            )
            return results, ver_note

        queries = plan_gap_queries(
            message, report, generate_fn=generate_fn, force=force or report.thin
        )
        if not queries:
            return results, ver_note

        emit_web_progress(
            f"Веб-поиск: gaps={','.join(report.gaps) or 'critic'} → "
            f"follow-up {len(queries)} запрос(ов)…"
        )
        logger.info("Gap follow-up gaps=%s force=%s queries=%s", report.gaps, force, queries)

        results, note = self._run_gap_queries(
            message,
            results,
            queries,
            pool=pool,
            lang=lang,
            date_r=date_r,
            require_full=require_full,
            label=(
                f"Gap follow-up (ODR reflection): thin={report.thin} "
                f"gaps={report.gaps} force={force}"
            ),
        )
        ver_note = ((ver_note or "").rstrip() + "\n\n" + note).strip()
        return results, ver_note

    def _apply_numeric_critic(
        self,
        message: str,
        results: list[SearchResult],
        *,
        deep: bool,
        use_orch: bool,
        pool: int,
        lang: str,
        date_r: str,
        generate_fn: Optional[Callable[[str], str]],
        require_full: bool,
        ver_note: Optional[str],
    ) -> tuple[list[SearchResult], Optional[str]]:
        """Critic on figures; one more gap round if place-locked numbers missing."""
        from .numeric_critic import critic_enabled, run_numeric_critic

        if not critic_enabled():
            return results, ver_note
        critic = run_numeric_critic(message, results)
        if critic.note:
            ver_note = ((ver_note or "").rstrip() + "\n\n" + critic.note).strip()
        if critic.needs_more_search and (deep or use_orch or len(message or "") >= 40):
            results, ver_note = self._maybe_gap_followup(
                message,
                results,
                SearchDecision(should_search=True),
                deep=deep,
                use_orch=use_orch,
                pool=pool,
                lang=lang,
                date_r=date_r,
                generate_fn=generate_fn,
                require_full=require_full,
                ver_note=ver_note,
                force=True,
            )
            # Re-run critic after forced gap
            critic2 = run_numeric_critic(message, results)
            if critic2.note:
                ver_note = ((ver_note or "").rstrip() + "\n\n" + critic2.note).strip()
        return results, ver_note

    def run(
        self,
        message:      str,
        turn_number:  int              = 0,
        generate_fn:  Optional[Callable[[str], str]] = None,
        allow_llm_tier: bool           = True,
        conversation_messages: Optional[list[str]] = None,
        conversation_key: Optional[str] = None,
        require_full_articles: bool = False,
        force_search: bool = False,
        cache_scope: str = "",
        operator_generate_fn: Optional[Callable[[str], str]] = None,
        *,
        serp_draft: bool = False,
        enrich_urls: Optional[list[str]] = None,
        explicit_subqueries: Optional[list[str]] = None,
        search_depth: Optional[SearchDepth] = None,
        search_query_override: Optional[str] = None,
    ) -> SearchContext:
        from dataclasses import replace as _dc_replace

        decision = self._decide_search(
            message,
            turn_number,
            force_search=force_search,
            allow_llm_tier=allow_llm_tier,
            generate_fn=generate_fn,
            conversation_messages=conversation_messages,
        )
        # SERP string may differ from message: message = user goal (orch/deep/place),
        # override = concrete query the model asked to search.
        ov = (search_query_override or "").strip()
        if ov and ov != (decision.search_query or "").strip():
            decision = _dc_replace(decision, search_query=ov[:500])
        # Belt-and-suspenders: never send instruction essays to SERP backends.
        sq = (decision.search_query or "").strip()
        if len(sq) > 240 or (sq.count(" ") > 28):
            from .live_query import compact_serp_query

            clamped = compact_serp_query(sq, goal=message)
            if clamped and clamped != sq:
                decision = _dc_replace(decision, search_query=clamped[:240])
        if search_depth is not None:
            decision = _dc_replace(decision, search_depth=search_depth)
        elif serp_draft:
            decision = _dc_replace(decision, search_depth=SearchDepth.FAST)
        self._log_classifier_decision(decision)

        full_fetch = bool(require_full_articles or decision.verified_sources_required) and not serp_draft
        op_fn = operator_generate_fn or generate_fn
        enrich_url_list = [u.strip() for u in (enrich_urls or []) if (u or "").strip()]
        explicit_qs = [q.strip() for q in (explicit_subqueries or []) if (q or "").strip()]

        if not decision.should_search:
            self._metrics.record_classifier_skip()
            logger.info("Search skipped by classifier: %s", decision.reason)
            emit_pipeline_stage("web_classify", done=True)
            return self._skipped_context(decision)

        if decision.category == SearchCategory.BROWSER_ACTION:
            return self._run_browser_action(
                message=message,
                query=decision.search_query,
                generate_fn=op_fn,
                hint_url=decision.browser_action_url or "",
            )

        self._metrics.record_classifier_search()
        emit_web_progress("Решение классификатора: нужен веб-поиск по этому сообщению.")

        decision, visual_media_kind = self._apply_visual_media_discovery(decision, message)
        decision = self._apply_portal_query_rewrite(message, decision)
        from .live_query import shape_live_search_query
        from dataclasses import replace as _dc_replace

        shaped = shape_live_search_query(message, decision.search_query)
        if shaped and shaped != (decision.search_query or ""):
            decision = _dc_replace(decision, search_query=shaped)

        cache_key = decision.search_query.lower()[:80]
        lang = self._detect_language(message)
        date_r = self._pick_date_restrict(decision)

        if cache_key in self._cache and self._cache[cache_key].is_valid():
            logger.debug("Search cache hit: %s", cache_key)
            self._metrics.record_cache_hit()
            emit_web_progress("Веб-поиск: использую сохранённые результаты (кэш).")
            cached = self._cache[cache_key].ctx
            return replace(
                cached,
                verified_sources_required=decision.verified_sources_required,
            )

        rctx = self._try_redis_cache_get(cache_scope, cache_key, lang, date_r, decision)
        if rctx is not None:
            emit_web_progress("Веб-поиск: Redis — сохранённые результаты по этому запросу.")
            return replace(
                rctx,
                verified_sources_required=decision.verified_sources_required,
            )

        if not self._conversation_budget_ok(conversation_key):
            logger.warning(
                "Search skipped: per-conversation budget exhausted (%s)",
                (conversation_key or "")[:24],
            )
            return self._skipped_context(decision)

        if self._duplicate_query_cooldown(conversation_key, decision.search_query, turn_number):
            logger.info(
                "Search suppressed: same query fingerprint recently in conv=%r",
                (conversation_key or "")[:24],
            )
            emit_web_progress(
                "Веб-поиск: такой же запрос уже недавно искали в этом чате — пропуск повтора."
            )
            return self._skipped_context(decision)

        pool = min(50, max(self._max_results * 2, 24))
        self._metrics.record_search_start()

        qq = decision.search_query or ""
        qdisp = (qq[:118] + "…") if len(qq) > 118 else qq
        emit_pipeline_stage("web_search", f"Ищу: «{qdisp}»")
        emit_web_progress(f"Веб-поиск: запрос «{qdisp}»")

        from .research_orchestrator import is_deep_research, orchestrate_search, should_orchestrate

        use_orch = bool(explicit_qs) or should_orchestrate(message, decision)
        deep = is_deep_research(message) and not serp_draft
        if use_orch:
            emit_web_progress("Веб-поиск: координатор делит задачу на параллельных search-агентов…")

        results, orch_note, orch_plans = orchestrate_search(
            message,
            decision.search_query,
            self._chain.search,
            decision=decision,
            generate_fn=generate_fn if (use_orch and not explicit_qs) else None,
            max_results=pool if not deep else min(50, max(pool, 36)),
            language=lang,
            date_restrict=date_r,
            explicit_subqueries=explicit_qs or None,
        )
        if use_orch and len(orch_plans) > 1:
            emit_web_progress(
                f"Веб-поиск: {len(orch_plans)} агентов → {len(results)} ссылок "
                f"({', '.join(p.role for p in orch_plans[:5])})"
            )
            logger.info("Orch note:\n%s", orch_note)

        emit_web_progress(f"Веб-поиск: получено ссылок — {len(results)}.")

        if not results and self._enable_stealth:
            self._metrics.record_stealth_fallback()
            emit_web_progress("Веб-поиск: пробую дополнительный источник…")
            results = self._stealth_fallback_sync(decision.search_query)
            emit_web_progress(f"Веб-поиск: после доп. источника — {len(results)} ссылок.")

        results = self._merge_search_rows(results, [])

        from .evidence_gate import filter_place_relevant
        from .live_query import demote_stale_current_rows
        from .structured.github_repo import merge_github_structured_results

        results = demote_stale_current_rows(message, results)
        results, place_note = filter_place_relevant(message, results)
        if place_note:
            logger.info("%s", place_note)

        results = merge_github_structured_results(
            results,
            query=decision.search_query or "",
            message=message or "",
        )

        # Deep research: temporarily raise enrich budget (core quality, not per-topic hacks)
        enrich_boost = (not serp_draft) and (deep or (use_orch and len(orch_plans) > 1))
        prev_enrich_n = self._page_enrich_n
        prev_enrich_chars = self._page_enrich_max_chars
        if enrich_boost:
            self._page_enrich_n = max(prev_enrich_n, min(18, prev_enrich_n + 2 * max(1, len(orch_plans))))
            self._page_enrich_max_chars = max(prev_enrich_chars, min(160_000, prev_enrich_chars + 32_000))

        if serp_draft:
            emit_pipeline_stage("web_read", "Черновик SERP (без чтения страниц)")
            emit_web_progress("Веб-поиск: быстрый draft — только сниппеты и URL…")
        else:
            emit_pipeline_stage("web_read", f"Читаю источники ({len(results)})")
            emit_web_progress("Веб-поиск: извлекаю и проверяю содержимое страниц…")
        try:
            results, ver_note = self._pipeline_after_fetch(
                results,
                decision,
                pool,
                lang,
                date_r,
                generate_fn,
                full_fetch or deep,
                message=message,
                serp_draft=serp_draft,
                enrich_urls=enrich_url_list or None,
            )
            if not serp_draft:
                results, ver_note = self._maybe_gap_followup(
                    message,
                    results,
                    decision,
                    deep=deep,
                    use_orch=use_orch and len(orch_plans) > 1,
                    pool=pool,
                    lang=lang,
                    date_r=date_r,
                    generate_fn=generate_fn,
                    require_full=full_fetch or deep,
                    ver_note=ver_note,
                )
                results, ver_note = self._apply_numeric_critic(
                    message,
                    results,
                    deep=deep,
                    use_orch=use_orch and len(orch_plans) > 1,
                    pool=pool,
                    lang=lang,
                    date_r=date_r,
                    generate_fn=generate_fn,
                    require_full=full_fetch or deep,
                    ver_note=ver_note,
                )
            else:
                draft_note = (
                    "SERP DRAFT phase: titles/snippets/URLs only. "
                    "For hard research, follow up with depth=deep and urls=[best candidates]."
                )
                ver_note = ((ver_note or "").rstrip() + "\n" + draft_note).strip()
        finally:
            if enrich_boost:
                self._page_enrich_n = prev_enrich_n
                self._page_enrich_max_chars = prev_enrich_chars

        if place_note:
            ver_note = ((ver_note or "").rstrip() + "\n" + place_note).strip()
        if use_orch and len(orch_plans) > 1:
            ver_note = ((ver_note or "").rstrip() + "\n\n" + orch_note).strip()

        ctx = self._synthesizer.synthesize(
            results=results,
            query=decision.search_query,
            category=decision.category,
            verification_note=ver_note,
            visual_media_kind=visual_media_kind,
        )
        ctx.verified_sources_required = decision.verified_sources_required
        self._prepend_session_snippets(conversation_key, ctx)

        # The operator is a fallback, used only when SERP + enrichment gave no useful context.
        # DEEP used to run it always (up to 35+ LLM calls and a minute of latency);
        # to restore that behaviour set FINKEY_BROWSING_OPERATOR_ON_DEEP=1.
        if serp_draft:
            logger.info("Browsing operator skipped: serp_draft phase.")
        elif ctx.is_useful() and not (full_fetch and _operator_on_deep_always()):
            logger.info("Browsing operator skipped: useful SERP context already assembled.")
        else:
            self._enrich_with_browsing_operator(
                message, results, ctx, op_fn, media_kind=visual_media_kind
            )

        if ctx.is_useful():
            if self._cache_ttl > 0:
                ttl_eff = self._effective_cache_ttl(decision.category)
                self._cache[cache_key] = _CacheEntry(ctx, ttl=ttl_eff)
            self._try_redis_cache_set(
                cache_scope, cache_key, lang, date_r, decision.category, ctx
            )
            self._remember_query_fingerprint(conversation_key, decision.search_query, turn_number)
            self._bump_conversation_search(conversation_key)
            self._remember_session_snippets(conversation_key, ctx)

        logger.info(
            "Search done: query=%r category=%s results=%d useful=%s citation_urls=%d synth_chars=%d",
            decision.search_query,
            decision.category,
            len(results),
            ctx.is_useful(),
            len(ctx.citation_urls or []),
            len(ctx.synthesized or ""),
        )
        emit_pipeline_stage("web_search", done=True)
        if ctx.is_useful():
            emit_web_progress("Веб-поиск: контекст собран, формирую ответ.")
        else:
            emit_web_progress("Веб-поиск: полезных результатов мало, продолжаю без богатого веб-контекста.")
        return ctx


    async def run_async(
        self,
        message:      str,
        turn_number:  int              = 0,
        generate_fn:  Optional[Callable[[str], str]] = None,
        allow_llm_tier: bool           = True,
        conversation_messages: Optional[list[str]] = None,
        conversation_key: Optional[str] = None,
        require_full_articles: bool = False,
        force_search: bool = False,
        cache_scope: str = "",
        operator_generate_fn: Optional[Callable[[str], str]] = None,
    ) -> SearchContext:
        decision = self._decide_search(
            message,
            turn_number,
            force_search=force_search,
            allow_llm_tier=allow_llm_tier,
            generate_fn=generate_fn,
            conversation_messages=conversation_messages,
        )
        self._log_classifier_decision(decision)

        full_fetch = bool(require_full_articles or decision.verified_sources_required)
        op_fn = operator_generate_fn or generate_fn

        if not decision.should_search:
            self._metrics.record_classifier_skip()
            logger.info("Search skipped by classifier (async): %s", decision.reason)
            return self._skipped_context(decision)

        if decision.category == SearchCategory.BROWSER_ACTION:
            return self._run_browser_action(
                message=message,
                query=decision.search_query,
                generate_fn=op_fn,
            )

        self._metrics.record_classifier_search()

        decision, visual_media_kind = self._apply_visual_media_discovery(decision, message)
        decision = self._apply_portal_query_rewrite(message, decision)
        from .live_query import shape_live_search_query
        from dataclasses import replace as _dc_replace

        shaped = shape_live_search_query(message, decision.search_query)
        if shaped and shaped != (decision.search_query or ""):
            decision = _dc_replace(decision, search_query=shaped)

        cache_key = decision.search_query.lower()[:80]
        lang = self._detect_language(message)
        date_r = self._pick_date_restrict(decision)

        if cache_key in self._cache and self._cache[cache_key].is_valid():
            logger.debug("Search cache hit (async): %s", cache_key)
            self._metrics.record_cache_hit()
            cached = self._cache[cache_key].ctx
            return replace(
                cached,
                verified_sources_required=decision.verified_sources_required,
            )

        rctx = self._try_redis_cache_get(cache_scope, cache_key, lang, date_r, decision)
        if rctx is not None:
            emit_web_progress("Веб-поиск: Redis — сохранённые результаты по этому запросу.")
            return replace(
                rctx,
                verified_sources_required=decision.verified_sources_required,
            )

        if not self._conversation_budget_ok(conversation_key):
            logger.warning(
                "Search skipped (async): per-conversation budget exhausted (%s)",
                (conversation_key or "")[:24],
            )
            return self._skipped_context(decision)

        if self._duplicate_query_cooldown(conversation_key, decision.search_query, turn_number):
            logger.info(
                "Search suppressed (async): same query fingerprint conv=%r",
                (conversation_key or "")[:24],
            )
            return self._skipped_context(decision)

        pool   = min(50, max(self._max_results * 2, 24))

        self._metrics.record_search_start()
        loop = asyncio.get_running_loop()

        from .research_orchestrator import is_deep_research, orchestrate_search, should_orchestrate

        use_orch = should_orchestrate(message, decision)
        deep = is_deep_research(message)
        if use_orch:
            emit_web_progress("Веб-поиск: координатор делит задачу на параллельных search-агентов…")

        def _orch_call() -> tuple:
            return orchestrate_search(
                message,
                decision.search_query,
                self._chain.search,
                decision=decision,
                generate_fn=generate_fn if use_orch else None,
                max_results=pool if not deep else min(50, max(pool, 36)),
                language=lang,
                date_restrict=date_r,
            )

        results, orch_note, orch_plans = await loop.run_in_executor(None, _orch_call)
        if use_orch and len(orch_plans) > 1:
            emit_web_progress(
                f"Веб-поиск: {len(orch_plans)} агентов → {len(results)} ссылок"
            )

        if not results and self._enable_stealth:
            self._metrics.record_stealth_fallback()
            results = await self._stealth_fallback_async(decision.search_query)

        results = self._merge_search_rows(results, [])

        from .evidence_gate import filter_place_relevant
        from .live_query import demote_stale_current_rows

        results = demote_stale_current_rows(message, results)
        results, place_note = filter_place_relevant(message, results)
        if place_note:
            logger.info("%s", place_note)

        enrich_boost = deep or (use_orch and len(orch_plans) > 1)
        prev_enrich_n = self._page_enrich_n
        prev_enrich_chars = self._page_enrich_max_chars
        if enrich_boost:
            self._page_enrich_n = max(prev_enrich_n, min(18, prev_enrich_n + 2 * max(1, len(orch_plans))))
            self._page_enrich_max_chars = max(prev_enrich_chars, min(160_000, prev_enrich_chars + 32_000))

        def _pipe() -> tuple[list[SearchResult], Optional[str]]:
            return self._pipeline_after_fetch(
                results,
                decision,
                pool,
                lang,
                date_r,
                generate_fn,
                full_fetch or deep,
                message=message,
            )

        try:
            results, ver_note = await loop.run_in_executor(None, _pipe)

            def _gap() -> tuple[list[SearchResult], Optional[str]]:
                return self._maybe_gap_followup(
                    message,
                    results,
                    decision,
                    deep=deep,
                    use_orch=use_orch and len(orch_plans) > 1,
                    pool=pool,
                    lang=lang,
                    date_r=date_r,
                    generate_fn=generate_fn,
                    require_full=full_fetch or deep,
                    ver_note=ver_note,
                )

            results, ver_note = await loop.run_in_executor(None, _gap)

            def _critic() -> tuple[list[SearchResult], Optional[str]]:
                return self._apply_numeric_critic(
                    message,
                    results,
                    deep=deep,
                    use_orch=use_orch and len(orch_plans) > 1,
                    pool=pool,
                    lang=lang,
                    date_r=date_r,
                    generate_fn=generate_fn,
                    require_full=full_fetch or deep,
                    ver_note=ver_note,
                )

            results, ver_note = await loop.run_in_executor(None, _critic)
        finally:
            if enrich_boost:
                self._page_enrich_n = prev_enrich_n
                self._page_enrich_max_chars = prev_enrich_chars

        if place_note:
            ver_note = ((ver_note or "").rstrip() + "\n" + place_note).strip()
        if use_orch and len(orch_plans) > 1:
            ver_note = ((ver_note or "").rstrip() + "\n\n" + orch_note).strip()

        ctx = self._synthesizer.synthesize(
            results=results,
            query=decision.search_query,
            category=decision.category,
            verification_note=ver_note,
            visual_media_kind=visual_media_kind,
        )
        ctx.verified_sources_required = decision.verified_sources_required
        self._prepend_session_snippets(conversation_key, ctx)

        if ctx.is_useful() and not (full_fetch and _operator_on_deep_always()):
            logger.info("Browsing operator skipped (async): useful SERP context already assembled.")
        else:
            await loop.run_in_executor(
                None,
                lambda mk=visual_media_kind: self._enrich_with_browsing_operator(
                    message, results, ctx, op_fn, media_kind=mk
                ),
            )

        if ctx.is_useful():
            if self._cache_ttl > 0:
                ttl_eff = self._effective_cache_ttl(decision.category)
                self._cache[cache_key] = _CacheEntry(ctx, ttl=ttl_eff)
            self._try_redis_cache_set(
                cache_scope, cache_key, lang, date_r, decision.category, ctx
            )
            self._remember_query_fingerprint(conversation_key, decision.search_query, turn_number)
            self._bump_conversation_search(conversation_key)
            self._remember_session_snippets(conversation_key, ctx)

        logger.info(
            "Search done (async): query=%r category=%s results=%d useful=%s citation_urls=%d synth_chars=%d",
            decision.search_query,
            decision.category,
            len(results),
            ctx.is_useful(),
            len(ctx.citation_urls or []),
            len(ctx.synthesized or ""),
        )
        return ctx


    @staticmethod
    def _apply_visual_media_discovery(
        decision: SearchDecision, message: str
    ) -> tuple[SearchDecision, Optional[str]]:
        from .visual_media_intent import (
            augment_search_query_for_media,
            detect_visual_media_intent,
        )

        kind = detect_visual_media_intent(message)
        if not kind:
            return decision, None
        new_q = augment_search_query_for_media(decision.search_query, message, kind)
        if new_q != decision.search_query:
            return replace(decision, search_query=new_q), kind
        return decision, kind

    @staticmethod
    def _parse_operator_allowlist_env() -> Optional[list[str]]:
        raw = os.environ.get("FINKEY_BROWSER_ALLOWLIST_HOST_SUFFIXES", "").strip()
        if not raw:
            return None
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        return parts or None

    def _rank_operator_entry_urls(
        self,
        message: str,
        results: list[SearchResult],
        media_kind: Optional[str] = None,
        *,
        aux_text: str = "",
    ) -> list[str]:
        """Ordered list of eligible operator entry URLs (best first)."""
        try:
            from .portal_browse import resolve_cinema_portal_plan
        except ImportError:
            plan = None
        else:
            plan = resolve_cinema_portal_plan(message, aux_text=aux_text)

        policy = self._search_policy()
        ordered: list[str] = []
        seen: set[str] = set()

        def _norm(u: str) -> str:
            return u.split("?")[0].rstrip("/").lower()

        def _add(url: str) -> None:
            key = _norm(url)
            if key not in seen:
                seen.add(key)
                ordered.append(url)

        if plan is not None:
            url = plan.entry_url
            if not is_blocked_for_operator_browse(url, media_kind=media_kind):
                ok, reason = policy.validate(url)
                if ok:
                    logger.info(
                        "Browsing operator: portal canonical entry first (%s -> %r)",
                        plan.portal_host,
                        url[:140],
                    )
                    _add(url)
                else:
                    logger.warning("Portal entry URL blocked by policy: %s (%s)", url[:96], reason)

        ranked: list[tuple[float, str]] = []
        for r in results:
            url = (r.url or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            if is_blocked_for_operator_browse(url, media_kind=media_kind):
                logger.debug("Operator skip unsuitable host/path %s", url[:96])
                continue
            ok, reason = policy.validate(url)
            if not ok:
                logger.debug("Operator skip URL %s: %s", url[:80], reason)
                continue
            base_rel = float(getattr(r, "relevance", 0.5) or 0.5)
            pri = operator_entry_priority_score(
                url, base_relevance=base_rel, media_kind=media_kind
            )
            ranked.append((pri, url))
        ranked.sort(key=lambda x: x[0], reverse=True)
        for _score, url in ranked:
            _add(url)

        if not ordered:
            logger.info(
                "Browsing operator: no eligible entry URL after filters (raw hits=%d)",
                len(results),
            )
        elif ranked:
            logger.info(
                "Browsing operator: ranked %d eligible URL(s), top=%r",
                len(ordered),
                ordered[0][:140],
            )
        return ordered

    def _pick_operator_start_url(
        self,
        message: str,
        results: list[SearchResult],
        media_kind: Optional[str] = None,
        *,
        aux_text: str = "",
    ) -> Optional[str]:
        urls = self._rank_operator_entry_urls(
            message, results, media_kind=media_kind, aux_text=aux_text
        )
        return urls[0] if urls else None

    @staticmethod
    def _operator_summary_footer_only(summary: str) -> bool:
        low = summary.lower()
        footer_kw = (
            "may make mistakes", "might make mistakes", "can make mistakes",
            "может ошибаться", "қателесуі", "peut commettre", "kann fehler",
            "puede cometer", "cookie preferences", "cookie preferences",
        )
        return len(summary.strip()) < 320 and any(w in low for w in footer_kw)

    def _operator_result_insufficient(self, op_res: object) -> bool:
        """True when operator output is too weak to stop trying fallback URLs."""
        ok = bool(getattr(op_res, "ok", False))
        summary = (getattr(op_res, "summary", None) or "").strip()
        facts = getattr(op_res, "facts", None) or []

        if not ok:
            return True
        if facts and len(summary) >= 40:
            return False
        try:
            min_chars = int(os.environ.get("FINKEY_OPERATOR_MIN_SUMMARY_CHARS", "120"))
        except ValueError:
            min_chars = 120
        min_chars = max(40, min(min_chars, 2000))

        if len(summary) < min_chars:
            return True
        if self._operator_summary_footer_only(summary):
            return True

        low = summary.lower()[:800]
        weak_signals = (
            "404", "not found", "403", "forbidden", "access denied",
            "captcha", "cloudflare", "no results", "step budget exhausted",
            "url_loop_break", "auto-stop", "no_tables_found",
            "insufficient", "could not", "failed to",
        )
        if any(s in low for s in weak_signals) and len(summary) < 600:
            return True
        return False

    def _run_browser_action(
        self,
        message: str,
        query: str,
        generate_fn: Optional[Callable[[str], str]],
        hint_url: str = "",
    ) -> SearchContext:
        """
        Handle BROWSER_ACTION category: the user wants to interact with a website.
        Resolves the target URL, launches the operator, returns result as SearchContext.
        The result becomes the primary answer content (not just enrichment).
        """
        from datetime import date as _date

        self._metrics.record_classifier_search()
        emit_web_progress("Открываю браузер для выполнения задачи…")
        logger.info("Browser action task: query=%r message_preview=%r", query[:80], message[:80])

        try:
            from finkey_search.progress import emit_browse_progress
            emit_browse_progress(
                phase="browser_action:start",
                detail="Запускаю браузер…",
                url="",
            )
        except Exception:
            pass

        start_url: Optional[str] = None

        if hint_url and hint_url.startswith("http"):
            start_url = hint_url
            logger.info("Browser action URL from classifier: %r", start_url[:80])

        if not start_url:
            _url_re = re.compile(r"https?://[^\s\)\]\>\"\']{6,}")
            _m = _url_re.search(message)
            if _m:
                start_url = _m.group(0).rstrip(".,;:!?")

        if not start_url and query:
            _bare = query.strip().lstrip("https://").lstrip("http://").split("/")[0]
            if "." in _bare and len(_bare) > 4:
                start_url = f"https://{_bare}"

        if not start_url:
            try:
                search_q = f"official website {query}" if query else message[:120]
                hits = self._chain.search(
                    search_q, max_results=5, language="", date_restrict="",
                )
                policy = URLPolicy(block_private_and_loopback=True)
                for hit in hits:
                    ok, _ = policy.validate(hit.url)
                    if ok and hit.url.startswith("http"):
                        start_url = hit.url
                        logger.info("Browser action URL resolved via search: %r", start_url[:80])
                        break
            except Exception as exc:
                logger.warning("Browser action URL resolve search failed: %s", exc)

        if not start_url:
            logger.info("Browser action: could not resolve URL for query=%r", query)
            return SearchContext(
                found=False,
                query_used=query,
                category=SearchCategory.BROWSER_ACTION,
                synthesized="",
            )

        try:
            from finkey_search._browser.operator_agent import (
                operator_result_to_prompt_block,
                run_operator_sync,
            )
            from finkey_search._browser.contour import (
                ContourDriver,
                ClientSyncSession,
                get_contour_manager,
            )
            from finkey_search._browser.contour.schema import ContourMode
            from finkey_search.progress import _live_browser_enabled

            _live = _live_browser_enabled()
            _user_lang = self._detect_language(message)
            contour = get_contour_manager().open_session(
                conversation_id=None,
                message=message,
                language_hint=_user_lang,
                start_url=start_url,
            )
            if contour.mode == ContourMode.CLIENT_SYNC:
                emit_web_progress(
                    "Клиентский Chrome подключён — оператор будет работать в вашей сессии."
                )

            _oauth_extras = [
                "accounts.google.com",
                "login.microsoftonline.com",
                "github.com",
                "gitlab.com",
                "cloudflare.com",
                "recaptcha.net",
                "hcaptcha.com",
            ]

            try:
                max_steps = int(os.environ.get("FINKEY_BROWSER_ACTION_MAX_STEPS", "50"))
            except ValueError:
                max_steps = 50

            _lang_name = {
                "ru": "Russian", "kk": "Kazakh", "en": "English",
                "de": "German", "fr": "French", "es": "Spanish",
                "tr": "Turkish", "zh": "Chinese", "ar": "Arabic",
            }.get(_user_lang, "Russian")

            _url_hint = f"START_URL={start_url} — you are already navigating there.\n" if start_url else ""

            enriched_goal = (
                f"USER_LANGUAGE: {_lang_name}\n"
                f"{message.strip()}\n\n"
                f"{_url_hint}"
                "BROWSER_ACTION_RULES (always follow):\n"
                "• ALL speak messages and ask_user questions MUST be in USER_LANGUAGE above.\n"
                "• When the goal asks you to TYPE or SEND a specific message/text: type it EXACTLY as written in the goal — DO NOT translate, paraphrase, or modify it in any way. The text to send is the literal content of the goal, not a translation.\n"
                "• DO NOT navigate to the same URL repeatedly — if you are already on the right page, STAY and interact.\n"
                "• To send a message in a chat or search: find text input → type_in_input (with the EXACT text from goal) → press_enter → wait 10000ms → observe → if response still loading (lone dot •, spinner, 'generating') wait 8000ms more → observe again → repeat until FULL response text appears → finish.\n"
                "• A lone bullet point (•) OR empty response area means the AI is STILL generating — do NOT finish yet. Wait and observe again.\n"
                "• NEVER finish immediately after press_enter — the response has not loaded yet.\n"
                "• If the page shows only a disclaimer footer with no chat response, or the only visible text is the message you just sent — the response is STILL loading. Keep waiting.\n"
                "• If the page requires login/auth and no credentials are in the goal: use ask_user to request them, in USER_LANGUAGE.\n"
                "• After any form submission, ALWAYS wait at least 8000ms before observing the result.\n"
                "• finish() summary MUST contain the ACTUAL response/result text visible on screen — NOT your own message. Quote the AI reply word for word.\n"
            )

            client_session = None
            session_cfg = None
            if contour.mode == ContourMode.CLIENT_SYNC:
                driver = ContourDriver(
                    conversation_id=None,
                    language_hint=_user_lang,
                    start_url=start_url,
                )
                # Force mode from already-resolved contour info
                driver._info = contour
                meta = contour.metadata or {}
                client_session = ClientSyncSession(
                    driver,
                    goal_host=meta.get("goal_host") or None,
                    allow_navigate_off_goal=bool(meta.get("allow_navigate_off_goal")),
                    allowed_host_suffixes_extra=_oauth_extras,
                )
            else:
                session_cfg = get_contour_manager().operator_session_config(
                    contour,
                    headless=True,
                    take_screenshot=_live,
                )
                session_cfg.browser.timeout_ms = 30_000
                session_cfg.browser.navigation_wait_until = "domcontentloaded"
                session_cfg.browser.block_media = not _live
                session_cfg.spa_post_navigation_settle_ms = 300
                session_cfg.url_policy = self._search_policy()
                session_cfg.allowed_host_suffixes_extra = _oauth_extras

            op_res = run_operator_sync(
                goal=enriched_goal,
                start_url=start_url,
                generate_fn=generate_fn,
                config=session_cfg,
                max_steps=max_steps,
                session=client_session,
            )

            today = _date.today().isoformat()
            if op_res.ok and (op_res.summary or "").strip():
                block = operator_result_to_prompt_block(op_res)
                synthesized = (
                    f"[BROWSER_ACTION_RESULT — задача выполнена в браузере {today}]\n{block}"
                )
                logger.info(
                    "Browser action completed: url=%r steps=%d",
                    start_url[:80],
                    len(op_res.steps),
                )
                return SearchContext(
                    found=True,
                    query_used=query,
                    category=SearchCategory.BROWSER_ACTION,
                    citation_urls=[start_url],
                    synthesized=synthesized,
                    search_date=today,
                )
            else:
                logger.info("Browser action produced no result: err=%s", op_res.error)
                return SearchContext(
                    found=False,
                    query_used=query,
                    category=SearchCategory.BROWSER_ACTION,
                    synthesized="",
                )
        except Exception as exc:
            logger.warning("Browser action failed: %s", exc, exc_info=True)
            return SearchContext(
                found=False,
                query_used=query,
                category=SearchCategory.BROWSER_ACTION,
                synthesized="",
            )

    def _enrich_with_browsing_operator(
        self,
        message: str,
        results: list[SearchResult],
        ctx: SearchContext,
        generate_fn: Optional[Callable[[str], str]],
        media_kind: Optional[str] = None,
    ) -> None:
        if not self._enable_browsing_operator:
            return
        # SERP-hopping operator (random sites) — off by default; enable explicitly.
        serp_fb = (os.getenv("FINKEY_OPERATOR_SERP_FALLBACK", "0") or "0").strip().lower()
        if serp_fb not in ("1", "true", "yes", "on"):
            logger.debug("Browsing operator SERP fallback disabled (FINKEY_OPERATOR_SERP_FALLBACK=0)")
            return
        if generate_fn is None:
            logger.debug("Browsing operator skipped: no generate_fn")
            return

        try:
            from .portal_browse import build_portal_operator_goal, resolve_cinema_portal_plan
        except ImportError:
            portal_plan = None
            build_portal_operator_goal = None  # type: ignore[assignment]
        else:
            portal_plan = resolve_cinema_portal_plan(message, aux_text=ctx.query_used or "")

        entry_urls = self._rank_operator_entry_urls(
            message,
            results,
            media_kind=media_kind,
            aux_text=ctx.query_used or "",
        )
        if not entry_urls:
            logger.debug("Browsing operator skipped: no eligible result URL")
            return

        try:
            max_fallback = int(os.environ.get("FINKEY_OPERATOR_FALLBACK_MAX", "0"))
        except ValueError:
            max_fallback = 0
        max_fallback = max(0, min(max_fallback, 4))
        urls_to_try = entry_urls[: 1 + max_fallback]

        goal = (
            build_portal_operator_goal(message, portal_plan)
            if portal_plan is not None and callable(build_portal_operator_goal)
            else message.strip()
        )
        max_steps_eff = self._operator_max_steps
        if portal_plan is not None:
            try:
                ps = int(os.environ.get("FINKEY_PORTAL_OPERATOR_MAX_STEPS", "40"))
            except ValueError:
                ps = 28
            max_steps_eff = min(80, max(self._operator_max_steps, max(4, ps)))

        try:
            from finkey_search._browser.operator_agent import (
                operator_result_to_prompt_block,
                run_operator_sync,
            )
            from finkey_search._browser.operator_session import OperatorSessionConfig
            from finkey_search._browser.stealth_browser import BrowserConfig

            from finkey_search.progress import _live_browser_enabled
            _live = _live_browser_enabled()
            session_cfg = OperatorSessionConfig(
                browser=BrowserConfig(
                    headless=True,
                    scroll_after_load=False,
                    wait_after_load_ms=400,
                    timeout_ms=25_000,
                    navigation_wait_until="domcontentloaded",
                    take_screenshot=_live,
                    block_media=not _live,
                ),
                url_policy=self._search_policy(),
                spa_post_navigation_settle_ms=300,
            )

            for attempt_i, start_url in enumerate(urls_to_try):
                if attempt_i > 0:
                    emit_web_progress(
                        f"На первом сайте мало данных — проверяю следующий "
                        f"({attempt_i + 1}/{len(urls_to_try)})…"
                    )
                    logger.info(
                        "Browsing operator fallback attempt %d/%d url=%r",
                        attempt_i + 1,
                        len(urls_to_try),
                        start_url[:120],
                    )

                op_res = run_operator_sync(
                    goal=goal,
                    start_url=start_url,
                    generate_fn=generate_fn,
                    config=session_cfg,
                    max_steps=max_steps_eff,
                )
                if not self._operator_result_insufficient(op_res):
                    block = operator_result_to_prompt_block(op_res)
                    ctx.synthesized = (ctx.synthesized or "").rstrip() + "\n\n" + block
                    logger.info(
                        "Browsing operator enriched search start_url=%r steps=%d attempt=%d",
                        start_url[:120],
                        len(op_res.steps),
                        attempt_i + 1,
                    )
                    return

                logger.info(
                    "Browsing operator attempt %d/%d insufficient url=%r ok=%s summary_len=%d",
                    attempt_i + 1,
                    len(urls_to_try),
                    start_url[:120],
                    op_res.ok,
                    len((op_res.summary or "").strip()),
                )

            logger.debug(
                "Browsing operator: all %d URL attempt(s) insufficient",
                len(urls_to_try),
            )
        except Exception as exc:
            logger.warning("Browsing operator failed (non-fatal): %s", exc)

    def _detect_language(self, message: str) -> str:
        cyrillic_count = sum(1 for c in message if "\u0400" <= c <= "\u04FF")
        return "lang_ru" if cyrillic_count > len(message) * 0.2 else "lang_en"

    def _pick_date_restrict(self, decision: SearchDecision) -> str:
        recent_categories = {
            SearchCategory.CURRENCY_RATE,
            SearchCategory.CRYPTO_PRICE,
            SearchCategory.STOCK_PRICE,
            SearchCategory.INTEREST_RATE,
            SearchCategory.FINANCIAL_NEWS,
        }
        if decision.category in recent_categories:
            return "d7"
        return "m3"

    def reset_session_cache(self) -> None:
        self._cache.clear()
        self._conv_last_search_fp.clear()
        self._conv_search_counts.clear()


    def _stealth_fallback_sync(self, query: str) -> list[SearchResult]:
        try:
            from finkey_search._browser.stealth_browser import BrowserConfig, fetch_sync
            import urllib.parse

            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&num=5"
            config = BrowserConfig(headless=True, timeout_ms=20_000, take_screenshot=False)
            page_result = fetch_sync(search_url, config)
            if page_result.ok:
                return self._parse_google_html(page_result.clean_text(), query)
        except Exception as exc:
            logger.warning("Stealth fallback failed: %s", exc)
        return []

    async def _stealth_fallback_async(self, query: str) -> list[SearchResult]:
        try:
            from finkey_search._browser.stealth_browser import StealthBrowser, BrowserConfig
            import urllib.parse

            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&num=5"
            config = BrowserConfig(headless=True, timeout_ms=20_000)
            async with StealthBrowser(config) as browser:
                page_result = await browser.fetch(search_url)
                if page_result.ok:
                    return self._parse_google_html(page_result.clean_text(), query)
        except Exception as exc:
            logger.warning("Async stealth fallback failed: %s", exc)
        return []

    @staticmethod
    def _parse_google_html(text: str, query: str) -> list[SearchResult]:
        import re as _re

        snippets = _re.split(r"\s{3,}", text)
        results: list[SearchResult] = []
        for chunk in snippets[:10]:
            chunk = chunk.strip()
            if len(chunk) > 60 and len(chunk) < 600:
                results.append(
                    SearchResult(
                        title=chunk[:80],
                        snippet=chunk,
                        url="",
                        source="stealth_browser",
                    )
                )
        return results[:5]
