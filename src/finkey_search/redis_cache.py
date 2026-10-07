# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Cross-process Redis cache for ``WebSearchEngine`` (same REDIS_URL as memory/RAG).

Stores a compact JSON snapshot of ``SearchContext`` so replicas share SERP+enrich work.
Qdrant / PostgreSQL are not used here — semantic dedupe and durable web logs can layer on later.

Env:
  FINKEY_WEB_SEARCH_REDIS_CACHE   auto | 1 | 0   (auto: on when REDIS_URL is set)
  FINKEY_WEB_SEARCH_REDIS_TTL_SEC  TTL seconds (default 240; capped by per-category engine TTL)
  FINKEY_WEB_SEARCH_REDIS_MAX_BYTES max stored JSON size (default 900_000); payload is shrunk if needed
  FINKEY_WEB_SEARCH_REDIS_ENRICH_CAP max chars of enriched_text per row kept in cache (default 12000)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Optional

from .schema import SearchCategory, SearchContext, SearchResult

logger = logging.getLogger(__name__)

_VERSION = "v1"


def redis_cache_enabled() -> bool:
    raw = (os.getenv("FINKEY_WEB_SEARCH_REDIS_CACHE", "auto") or "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return bool(os.getenv("REDIS_URL", "").strip())
    return bool(os.getenv("REDIS_URL", "").strip())


def _ttl_default() -> int:
    try:
        return max(30, min(int(os.getenv("FINKEY_WEB_SEARCH_REDIS_TTL_SEC", "240")), 86_400))
    except ValueError:
        return 240


def _max_bytes() -> int:
    try:
        return max(50_000, min(int(os.getenv("FINKEY_WEB_SEARCH_REDIS_MAX_BYTES", "900000")), 2_000_000))
    except ValueError:
        return 900_000


def _enrich_cap() -> int:
    try:
        return max(0, min(int(os.getenv("FINKEY_WEB_SEARCH_REDIS_ENRICH_CAP", "12000")), 200_000))
    except ValueError:
        return 12_000


def make_redis_key(
    scope: str,
    *,
    query_key: str,
    lang: str,
    date_restrict: str,
    max_results: int,
) -> str:
    scope_s = (scope or "global").strip() or "global"
    scope_s = re.sub(r"[^\w\.\-:@/]+", "_", scope_s)[:120]
    h = hashlib.sha256(
        f"{_VERSION}|{query_key}|{lang}|{date_restrict}|{max_results}".encode("utf-8"),
    ).hexdigest()[:40]
    return f"finkey:web:{_VERSION}:{scope_s}:{h}"


def _result_to_dict(r: SearchResult, enrich_cap: int) -> dict[str, Any]:
    if enrich_cap <= 0:
        et_out: Optional[str] = None
    else:
        et = (r.enriched_text or "").strip()
        if len(et) > enrich_cap:
            et_out = et[:enrich_cap] + "\n…[truncated-for-cache]"
        else:
            et_out = et or None
    return {
        "title": r.title,
        "url": r.url,
        "snippet": r.snippet,
        "date": r.date,
        "source": r.source,
        "relevance": r.relevance,
        "enriched_text": et_out,
    }


def _dict_to_result(d: dict[str, Any]) -> SearchResult:
    return SearchResult(
        title=str(d.get("title") or ""),
        url=str(d.get("url") or ""),
        snippet=str(d.get("snippet") or ""),
        date=d.get("date"),
        source=str(d.get("source") or ""),
        relevance=float(d.get("relevance") or 1.0),
        enriched_text=d.get("enriched_text"),
    )


def context_to_payload(
    ctx: SearchContext,
    *,
    max_bytes: int,
    enrich_cap: int,
) -> str:
    cat = ctx.category.value if ctx.category else None
    if enrich_cap <= 0:
        enrich_cap = _enrich_cap()
    body: dict[str, Any] = {
        "v": _VERSION,
        "found": ctx.found,
        "query_used": ctx.query_used,
        "category": cat,
        "verified_sources_required": ctx.verified_sources_required,
        "citation_urls": list(ctx.citation_urls or []),
        "search_date": ctx.search_date,
        "total_results": int(ctx.total_results or 0),
        "synthesized": ctx.synthesized or "",
        "results": [_result_to_dict(r, enrich_cap) for r in (ctx.results or [])],
    }
    raw = json.dumps(body, ensure_ascii=False)
    if len(raw.encode("utf-8")) <= max_bytes:
        return raw
    body["results"] = [_result_to_dict(r, 0) for r in (ctx.results or [])]
    syn = body.get("synthesized") or ""
    for cut in (500_000, 300_000, 150_000, 80_000, 40_000, 20_000):
        body["synthesized"] = syn[:cut]
        raw = json.dumps(body, ensure_ascii=False)
        if len(raw.encode("utf-8")) <= max_bytes:
            return raw
    body["synthesized"] = syn[:12_000]
    return json.dumps(body, ensure_ascii=False)


def payload_to_context(data: dict[str, Any]) -> Optional[SearchContext]:
    try:
        if data.get("v") != _VERSION:
            return None
        cat_raw = data.get("category")
        cat: Optional[SearchCategory] = None
        if cat_raw:
            try:
                cat = SearchCategory(str(cat_raw))
            except ValueError:
                cat = None
        rows = data.get("results") or []
        if not isinstance(rows, list):
            return None
        results = [_dict_to_result(x) for x in rows if isinstance(x, dict)]
        return SearchContext(
            found=bool(data.get("found")),
            query_used=str(data.get("query_used") or ""),
            category=cat,
            verified_sources_required=bool(data.get("verified_sources_required")),
            citation_urls=list(data.get("citation_urls") or []),
            results=results,
            synthesized=str(data.get("synthesized") or ""),
            search_date=str(data.get("search_date") or ""),
            total_results=int(data.get("total_results") or 0),
        )
    except Exception as exc:
        logger.debug("web redis payload decode failed: %s", exc)
        return None


def redis_get_context(redis_client: Any, key: str) -> Optional[SearchContext]:
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(key)
    except Exception as exc:
        logger.debug("web redis get failed: %s", exc)
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return payload_to_context(data)


def redis_set_context(redis_client: Any, key: str, ctx: SearchContext, ttl_sec: int) -> None:
    if redis_client is None or ttl_sec <= 0:
        return
    try:
        blob = context_to_payload(ctx, max_bytes=_max_bytes(), enrich_cap=_enrich_cap())
        redis_client.setex(key, ttl_sec, blob)
    except Exception as exc:
        logger.debug("web redis set failed: %s", exc)


__all__ = [
    "redis_cache_enabled",
    "make_redis_key",
    "context_to_payload",
    "payload_to_context",
    "redis_get_context",
    "redis_set_context",
    "_ttl_default",
]
