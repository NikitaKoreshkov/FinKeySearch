# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
WebSearchExecutor — executes the actual search via Google Custom Search API.

Primary:  Google Custom Search JSON API
          (fastest, most reliable, structured results)

The executor:
  - Sends query to Google CSE
  - Returns up to N ranked results
  - Extracts publication dates when available
  - Deduplicates by domain
  - Handles errors gracefully (returns empty on failure, never crashes FinKey)

Configuration (never commit secrets; use env or constructor).

  Preferred:
    GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX

  Legacy (same meaning as in the old FinKey repo — .env used GOOGLE_*):
    GOOGLE_API_KEY, GOOGLE_CSE_ID

  Resilience:
    FINKEY_SEARCH_CSE_HTTP_RETRIES (default 3, max 8)
    FINKEY_SEARCH_CSE_RETRY_BACKOFF_MS (default 350) — exponential backoff for 429 / 5xx
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Optional
from urllib.parse import urlparse

from .schema import SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_API_KEY = ""
_DEFAULT_CX      = ""
_GOOGLE_CSE_URL  = "https://www.googleapis.com/customsearch/v1"


def _env_cse_credentials() -> tuple[str, str]:
    """
    Resolve key + Search Engine ID from environment.
    New names win if set; else fall back to legacy FinKey (.env) names.
    """
    key = (
        os.getenv("GOOGLE_CSE_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    cx = (
        os.getenv("GOOGLE_CSE_CX", "").strip()
        or os.getenv("GOOGLE_CSE_ID", "").strip()
    )
    return key, cx


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url


def _extract_date_from_snippet(snippet: str) -> Optional[str]:
    """Try to find a date in the snippet text."""
    patterns = [
        r"\b(\d{1,2}\s+(?:янв|фев|мар|апр|мая|июн|июл|авг|сен|окт|ноя|дек)\w*\s+\d{4})\b",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{1,2}\.\d{2}\.\d{4})\b",
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b",
    ]
    for p in patterns:
        m = re.search(p, snippet, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


class WebSearchExecutor:
    """
    Executes Google Custom Search and returns structured results.

    Usage::
        executor = WebSearchExecutor()
        results = executor.search("курс доллара сегодня", max_results=5)
    """

    def __init__(
        self,
        api_key: str = "",
        cx:      str = "",
    ):
        env_key, env_cx = _env_cse_credentials()
        self._api_key = (api_key or env_key or _DEFAULT_API_KEY).strip()
        self._cx      = (cx      or env_cx or _DEFAULT_CX).strip()

    def _redact_for_log(self, text: str) -> str:
        out = str(text)
        k, c = self._api_key, self._cx
        if len(k) > 8:
            out = out.replace(k, "***API_KEY***")
        if len(c) > 4:
            out = out.replace(c, "***CX***")
        return out

    def search(
        self,
        query:       str,
        max_results: int  = 5,
        language:    str  = "lang_ru",
        date_restrict: str = "m3",
    ) -> list[SearchResult]:
        """
        Synchronous search. Returns up to max_results deduplicated results.
        Tries requests → httpx → stdlib urllib (with SSL fallback) in order.
        """
        if not str(self._api_key).strip() or not str(self._cx).strip():
            logger.warning(
                "Google CSE skipped: set GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX "
                "or legacy GOOGLE_API_KEY + GOOGLE_CSE_ID "
                "(see docs/GOOGLE_CSE_BROWSER_VERIFY.md)"
            )
            return []

        params = {
            "key":          self._api_key,
            "cx":           self._cx,
            "q":            query,
            "num":          min(max_results, 10),
            "lr":           language,
            "safe":         "active",
            "dateRestrict": date_restrict,
        }
        gl = os.getenv("FINKEY_SEARCH_GL", "").strip()
        cr = os.getenv("FINKEY_SEARCH_CR", "").strip()
        if gl:
            params["gl"] = gl
        if cr:
            params["cr"] = cr

        try:
            data = self._http_get(_GOOGLE_CSE_URL, params)
            return self._parse_response(data, max_results)
        except Exception as exc:
            logger.warning(
                "Google CSE search failed: %s",
                self._redact_for_log(exc),
            )
            return []

    def _cse_retry_params(self) -> tuple[int, float]:
        try:
            n = int(os.getenv("FINKEY_SEARCH_CSE_HTTP_RETRIES", "3"))
        except ValueError:
            n = 3
        try:
            ms = float(os.getenv("FINKEY_SEARCH_CSE_RETRY_BACKOFF_MS", "350"))
        except ValueError:
            ms = 350.0
        return max(1, min(n, 8)), max(0.05, ms / 1000.0)

    def _http_get(self, url: str, params: dict) -> dict:
        """HTTP GET with automatic SSL fix — tries requests first, then urllib."""
        import json as _json
        import urllib.parse

        full_url = url + "?" + urllib.parse.urlencode(params)
        headers  = {"User-Agent": "FinKeySearch/1.0"}
        retries, backoff_base = self._cse_retry_params()
        retry_status = frozenset({429, 500, 502, 503, 504})

        try:
            import requests as _req

            for attempt in range(retries):
                try:
                    resp = _req.get(full_url, headers=headers, timeout=12)
                    if resp.status_code in retry_status and attempt + 1 < retries:
                        time.sleep(backoff_base * (2**attempt))
                        continue
                    resp.raise_for_status()
                    return resp.json()
                except _req.HTTPError as exc:
                    code = getattr(exc.response, "status_code", None)
                    if code in retry_status and attempt + 1 < retries:
                        time.sleep(backoff_base * (2**attempt))
                        continue
                    raise
        except ImportError:
            pass

        try:
            import httpx

            for attempt in range(retries):
                try:
                    with httpx.Client(timeout=12) as client:
                        resp = client.get(full_url, headers=headers)
                        if resp.status_code in retry_status and attempt + 1 < retries:
                            time.sleep(backoff_base * (2**attempt))
                            continue
                        resp.raise_for_status()
                        return resp.json()
                except httpx.HTTPStatusError as exc:
                    code = exc.response.status_code if exc.response is not None else None
                    if code in retry_status and attempt + 1 < retries:
                        time.sleep(backoff_base * (2**attempt))
                        continue
                    raise
        except ImportError:
            pass

        import ssl, urllib.request
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        req = urllib.request.Request(full_url, headers=headers)
        with urllib.request.urlopen(req, timeout=6, context=ctx) as resp:
            return _json.loads(resp.read().decode())

    async def search_async(
        self,
        query:       str,
        max_results: int  = 5,
        language:    str  = "lang_ru",
        date_restrict: str = "m3",
    ) -> list[SearchResult]:
        """Async wrapper — runs the sync search in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(query, max_results, language, date_restrict),
        )

    def _parse_response(
        self,
        data: dict,
        max_results: int,
    ) -> list[SearchResult]:
        items = data.get("items", [])
        seen_domains: set[str] = set()
        results: list[SearchResult] = []

        for item in items:
            if len(results) >= max_results:
                break

            url     = item.get("link", "")
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            domain  = _extract_domain(url)

            if domain in seen_domains:
                continue
            seen_domains.add(domain)

            date = None
            pagemap = item.get("pagemap", {})
            metatags = pagemap.get("metatags", [{}])
            if metatags:
                meta = metatags[0]
                date = (
                    meta.get("article:published_time") or
                    meta.get("og:updated_time") or
                    meta.get("date")
                )
            if not date:
                date = _extract_date_from_snippet(snippet)

            snippet = re.sub(r'\s+', ' ', snippet).strip()

            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                date=date,
                source=domain,
                relevance=1.0 - (len(results) * 0.1),
            ))

        return results
