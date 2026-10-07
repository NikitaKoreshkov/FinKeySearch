# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Brave Search API backend.

Env:
  - BRAVE_API_KEY (required)
  - BRAVE_SEARCH_API_URL (optional)
  - BRAVE_MAX_RETRIES (default 2 — extra GET attempts after HTTP 429)
  - BRAVE_RETRY_BASE_SEC (default 0.75 — exponential backoff base)
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from urllib.parse import urlencode, urlparse

from finkey_search.schema import SearchResult

logger = logging.getLogger(__name__)

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url or ""


class BraveSearchBackend:
    def __init__(self, api_key: str = "", endpoint: str = "") -> None:
        self._key = (api_key or os.getenv("BRAVE_API_KEY", "").strip()).strip()
        self._endpoint = (endpoint or os.getenv("BRAVE_SEARCH_API_URL", "").strip() or _BRAVE_URL).strip()

    @property
    def name(self) -> str:
        return "brave"

    def configured(self) -> bool:
        return bool(self._key)

    def search(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        date_restrict: str,
    ) -> list[SearchResult]:
        if not self._key:
            return []

        q = (query or "").strip()
        if len(q) > 240:
            q = q[:240].rsplit(" ", 1)[0]
        if not q:
            return []

        params = {
            "q": q,
            "count": str(min(max(1, max_results), 20)),
            "safesearch": "moderate",
            "search_lang": "ru" if language == "lang_ru" else "en",
        }
        if date_restrict in ("d1", "w1", "d7", "m1"):
            params["freshness"] = {"d1": "pd", "w1": "pw", "d7": "pw", "m1": "pm"}[date_restrict]

        sep = "&" if "?" in self._endpoint else "?"
        full_url = self._endpoint + sep + urlencode(params)
        headers = {
            "X-Subscription-Token": self._key,
            "User-Agent": "FinKeyAI/1.0",
        }
        try:
            body = self._get_json(full_url, headers)
        except Exception as exc:
            # Re-raise rate limits so FallbackSearchChain trips the circuit breaker
            # instead of silently returning [] and letting fanout hammer Brave.
            if self._is_retryable_rate_limit(exc):
                raise
            logger.warning("Brave search failed: %s", exc)
            return []

        rows = ((body.get("web") or {}).get("results") or [])
        results: list[SearchResult] = []
        seen_domains: set[str] = set()

        for i, item in enumerate(rows):
            if len(results) >= max_results:
                break
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            snippet = (item.get("description") or "").strip()
            if len(snippet) < 20:
                continue
            domain = _extract_domain(url)
            if domain and domain in seen_domains:
                continue
            if domain:
                seen_domains.add(domain)

            snippet = re.sub(r"\s+", " ", snippet).strip()
            results.append(
                SearchResult(
                    title=title or domain or "result",
                    url=url,
                    snippet=snippet[:2000],
                    date=None,
                    source=domain or "brave",
                    relevance=max(0.0, 1.0 - (i * 0.08)),
                )
            )
        return results

    @staticmethod
    def _is_retryable_rate_limit(exc: BaseException) -> bool:
        code = getattr(exc, "code", None)
        if code == 429:
            return True
        resp = getattr(exc, "response", None)
        if resp is not None:
            sc = getattr(resp, "status_code", None)
            if sc == 429:
                return True
        msg = str(exc)
        return "429" in msg or "Too Many Requests" in msg

    def _get_json(self, full_url: str, headers: dict[str, str]) -> dict:
        try:
            max_retries = int(os.getenv("BRAVE_MAX_RETRIES", "2"))
        except ValueError:
            max_retries = 2
        try:
            base_sleep = float(os.getenv("BRAVE_RETRY_BASE_SEC", "0.75"))
        except ValueError:
            base_sleep = 0.75

        last_exc: BaseException | None = None
        for attempt in range(max_retries + 1):
            try:
                return self._get_json_once(full_url, headers)
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries or not self._is_retryable_rate_limit(exc):
                    raise
                delay = base_sleep * (2**attempt) + random.uniform(0, 0.35)
                logger.warning(
                    "Brave HTTP 429, backoff retry %s/%s in %.2fs",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _get_json_once(self, full_url: str, headers: dict[str, str]) -> dict:
        import urllib.request

        try:
            import requests as _rq

            resp = _rq.get(full_url, headers=headers, timeout=12)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            pass

        try:
            import httpx

            with httpx.Client(timeout=12) as client:
                r = client.get(full_url, headers=headers)
                r.raise_for_status()
                return r.json()
        except ImportError:
            pass

        req = urllib.request.Request(full_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode())
