# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Tavily Search API — optional second-hop retrieval when CSE is empty or rate-limited.

Env: ``TAVILY_API_KEY``. Docs: https://docs.tavily.com/documentation/api-reference/endpoint/search
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse

from finkey_search.schema import SearchResult

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url or ""


class TavilySearchBackend:
    def __init__(self, api_key: str = "") -> None:
        self._key = (api_key or os.getenv("TAVILY_API_KEY", "").strip()).strip()

    @property
    def name(self) -> str:
        return "tavily"

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
        del language, date_restrict
        if not self._key:
            return []

        payload = {
            "api_key":      self._key,
            "query":        query,
            "max_results":  min(max(1, max_results), 20),
            "search_depth": "advanced",
            "include_answer": False,
        }

        try:
            body = self._post_json(_TAVILY_URL, payload)
        except Exception as exc:
            logger.warning("Tavily search failed: %s", exc)
            return []

        raw_items = body.get("results") or []
        results: list[SearchResult] = []
        seen: set[str] = set()

        for i, item in enumerate(raw_items):
            if len(results) >= max_results:
                break
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            snippet = (item.get("content") or item.get("snippet") or "").strip()
            if not snippet and title:
                snippet = title
            if len(snippet) < 20:
                continue
            domain = _extract_domain(url)
            if domain in seen:
                continue
            seen.add(domain)

            score_raw = item.get("score")
            try:
                base_rel = float(score_raw) if score_raw is not None else 1.0 - i * 0.08
            except (TypeError, ValueError):
                base_rel = 1.0 - i * 0.08

            snippet = re.sub(r"\s+", " ", snippet).strip()

            results.append(
                SearchResult(
                    title=title or domain or "result",
                    url=url,
                    snippet=snippet[:2000],
                    date=None,
                    source=domain or "tavily",
                    relevance=max(0.0, min(1.0, base_rel)),
                )
            )

        return results

    def _post_json(self, url: str, payload: dict) -> dict:
        import urllib.error
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "FinKeySearch/1.0"},
            method="POST",
        )
        try:
            import requests as _rq

            resp = _rq.post(url, json=payload, headers={"User-Agent": "FinKeySearch/1.0"}, timeout=12)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            pass

        try:
            import httpx

            with httpx.Client(timeout=12) as client:
                r = client.post(url, json=payload, headers={"User-Agent": "FinKeySearch/1.0"})
                r.raise_for_status()
                return r.json()
        except ImportError:
            pass

        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode())
