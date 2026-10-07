# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Batch page fetch via crawl4ai ``AsyncWebCrawler.arun_many`` (optional dependency).

Purpose
-------
One asyncio pass over N URLs → per-URL ``{url, markdown, metadata}``. Designed as
a drop-in enrichment backend for the page-enrichment wave loop
(``page_enrichment._run_wave``) and for direct-URL fan-out (``url_direct_fetch``),
where today each URL is fetched on its own thread/browser context.

Gates & degradation
-------------------
• ``FINKEY_INTERNET_BATCH_BACKEND=crawl4ai|none`` (default ``none`` — opt-in;
  crawl4ai pulls playwright + heavy deps and is NOT in core requirements).
• Import is guarded by ``try/except ImportError``; when missing →
  ``{"ok": False, "error": "crawl4ai_unavailable"}``.
• Whole run is bounded by an ``asyncio.wait_for`` timeout; any exception is
  returned as ``{"ok": False, "error": ...}`` — this module never raises to
  the caller and never blocks the legacy path.

Caller contract: treat ``ok=False`` as "batch backend unusable, continue with
the existing per-URL fetch path".
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_BACKEND_ENV = "FINKEY_INTERNET_BATCH_BACKEND"


def batch_backend() -> str:
    raw = (os.getenv(_BACKEND_ENV, "none") or "none").strip().lower()
    return raw if raw in ("crawl4ai", "none") else "none"


def fetch_pages_batch(
    urls: list[str],
    *,
    timeout_s: float = 60.0,
    max_concurrency: int = 5,
    headless: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Batch-fetch ``urls`` with crawl4ai when the backend gate is on.

    Returns::

        {"ok": True,  "backend": "crawl4ai", "results": [{url, markdown, metadata, ...}, ...]}
        {"ok": False, "error": "crawl4ai_unavailable" | "batch_backend_disabled" | "no_urls" | "..."}
    """
    cleaned = [(u or "").strip() for u in (urls or []) if (u or "").strip()]
    if not cleaned:
        return {"ok": False, "error": "no_urls"}
    if batch_backend() != "crawl4ai":
        return {"ok": False, "error": "batch_backend_disabled"}
    try:
        import crawl4ai  # noqa: F401  — optional dep probe; real import inside runner
    except ImportError:
        logger.info("Batch fetch: crawl4ai not installed, caller should use legacy path")
        return {"ok": False, "error": "crawl4ai_unavailable"}

    _headless = headless if headless is not None else True

    async def _runner() -> list[dict[str, Any]]:
        from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            semaphore_count=max(1, min(int(max_concurrency), 10)),
        )
        out: list[dict[str, Any]] = []
        async with AsyncWebCrawler(headless=_headless, verbose=False) as crawler:
            dispatched = await crawler.arun_many(cleaned, config=config)
            for r in dispatched or []:
                md = getattr(r, "markdown", None)
                if md is not None and not isinstance(md, str):
                    md = getattr(md, "raw_markdown", None) or str(md)
                meta = getattr(r, "metadata", None)
                if not isinstance(meta, dict):
                    meta = {}
                out.append(
                    {
                        "url": getattr(r, "url", "") or "",
                        "markdown": (md or "").strip(),
                        "metadata": meta,
                        "success": bool(getattr(r, "success", True)),
                        "error": (getattr(r, "error_message", None) or "") or None,
                    }
                )
        return out

    try:
        results = asyncio.run(asyncio.wait_for(_runner(), timeout=timeout_s))
    except asyncio.TimeoutError:
        logger.info("Batch fetch timeout after %.1fs for %d urls", timeout_s, len(cleaned))
        return {"ok": False, "error": f"timeout_{int(timeout_s)}s"}
    except ImportError:
        return {"ok": False, "error": "crawl4ai_unavailable"}
    except Exception as exc:
        logger.info("Batch fetch failed: %s", exc)
        return {"ok": False, "error": f"{type(exc).__name__}:{str(exc)[:200]}"}

    return {"ok": True, "backend": "crawl4ai", "results": results}
