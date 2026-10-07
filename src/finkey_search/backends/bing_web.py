# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Microsoft Bing Web Search API (v7) — программная выдача без HTML-скрейпа.

Ключ (любое имя из списка): ``BING_SEARCH_API_KEY``, ``AZURE_BING_SEARCH_KEY``,
``BING_SUBSCRIPTION_KEY``.

Опционально: ``BING_SEARCH_API_URL`` — полный URL эндпоинта (глобальный или Azure Cognitive Services).
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional
from urllib.parse import urlencode, urlparse

from finkey_search.schema import SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


def _subscription_key_from_env() -> str:
    return (
        os.getenv("BING_SEARCH_API_KEY", "").strip()
        or os.getenv("AZURE_BING_SEARCH_KEY", "").strip()
        or os.getenv("BING_SUBSCRIPTION_KEY", "").strip()
    )


def _endpoint_from_env() -> str:
    return (os.getenv("BING_SEARCH_API_URL", "").strip() or _DEFAULT_ENDPOINT).strip()


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url or ""


def _mkt_from_language(language: str) -> str:
    override = os.getenv("FINKEY_SEARCH_BING_MKT", "").strip()
    if override:
        return override
    lang = (language or "").strip().lower()
    if lang == "lang_ru":
        return "ru-RU"
    return "en-US"


def _freshness(date_restrict: str) -> Optional[str]:
    dr = (date_restrict or "m3").strip().lower()
    if dr == "d1":
        return "Day"
    if dr in ("w1", "d7"):
        return "Week"
    if dr in ("m1", "m3"):
        return "Month"
    return None


class BingWebSearchBackend:
    """Адаптер Bing Web Search API v7 → ``SearchResult``."""

    def __init__(self, subscription_key: str = "", endpoint: str = "") -> None:
        self._key = (subscription_key or _subscription_key_from_env()).strip()
        self._endpoint = (endpoint or _endpoint_from_env()).strip()

    @property
    def name(self) -> str:
        return "bing_web"

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
            logger.warning(
                "Bing Web Search skipped: set BING_SEARCH_API_KEY "
                "(or AZURE_BING_SEARCH_KEY / BING_SUBSCRIPTION_KEY)"
            )
            return []

        mkt = _mkt_from_language(language)
        freshness = _freshness(date_restrict)

        params: dict[str, str] = {
            "q": query,
            "count": str(min(max(max_results, 1), 50)),
            "mkt": mkt,
            "textDecorations": "false",
            "textFormat": "Raw",
            "safeSearch": "Moderate",
        }
        if freshness:
            params["freshness"] = freshness

        headers = {
            "User-Agent": "FinKeyAI/1.0",
            "Ocp-Apim-Subscription-Key": self._key,
        }

        sep = "&" if "?" in self._endpoint else "?"
        full_url = self._endpoint + sep + urlencode(params)

        try:
            body = self._http_get_json(full_url, headers)
            return self._parse_web(body, max_results)
        except Exception as exc:
            logger.warning("Bing Web Search failed: %s", self._redact_exc(exc))
            return []

    def _redact_exc(self, exc: BaseException) -> str:
        msg = str(exc)
        if len(self._key) > 8:
            msg = msg.replace(self._key, "***KEY***")
        return msg

    def _retry_params(self) -> tuple[int, float]:
        try:
            n = int(os.getenv("FINKEY_SEARCH_BING_HTTP_RETRIES", "3"))
        except ValueError:
            n = 3
        try:
            ms = float(os.getenv("FINKEY_SEARCH_BING_RETRY_BACKOFF_MS", "350"))
        except ValueError:
            ms = 350.0
        return max(1, min(n, 8)), max(0.05, ms / 1000.0)

    def _http_get_json(self, full_url: str, headers: dict[str, str]) -> dict:
        import json as _json

        retries, backoff_base = self._retry_params()
        retry_status = frozenset({429, 500, 502, 503, 504})

        try:
            import requests as _req

            for attempt in range(retries):
                try:
                    resp = _req.get(full_url, headers=headers, timeout=15)
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
                    with httpx.Client(timeout=15) as client:
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

        import ssl
        import urllib.request

        ctx = ssl.create_default_context()
        req = urllib.request.Request(full_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            return _json.loads(resp.read().decode())

    def _parse_web(self, data: dict, max_results: int) -> list[SearchResult]:
        wp = data.get("webPages") or {}
        items = wp.get("value") or []
        seen_domains: set[str] = set()
        results: list[SearchResult] = []

        for item in items:
            if len(results) >= max_results:
                break
            url = (item.get("url") or "").strip()
            title = (item.get("name") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            snippet = re.sub(r"\s+", " ", snippet).strip()
            domain = _extract_domain(url)
            if domain and domain in seen_domains:
                continue
            if domain:
                seen_domains.add(domain)
            elif not url:
                continue

            date = item.get("datePublished") or item.get("dateLastCrawled")

            results.append(
                SearchResult(
                    title=title or domain or "result",
                    url=url,
                    snippet=snippet,
                    date=str(date) if date else None,
                    source=domain,
                    relevance=1.0 - (len(results) * 0.1),
                )
            )

        return results
