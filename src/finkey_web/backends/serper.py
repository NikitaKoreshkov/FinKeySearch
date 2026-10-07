# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Serper.dev Google Search API backend.

Env:
  - SERPER_API_KEY (primary; also first entry if SERPER_API_KEYS unset)
  - SERPER_API_KEYS (optional comma/newline-separated pool — rotate on credits/auth fail)
  - SERPER_API_URL (optional, defaults to https://google.serper.dev/search)
  - SERPER_MAX_RETRIES (default 2 — extra POST attempts after HTTP 429)
  - SERPER_RETRY_BASE_SEC (default 0.5 — exponential backoff base)
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from urllib.parse import urlparse

from finkey_web.schema import SearchResult

logger = logging.getLogger(__name__)

_SERPER_URL = "https://google.serper.dev/search"

# Process-wide pool so every SerperSearchBackend shares rotation state.
_pool_lock = threading.Lock()
_pool_keys: list[str] = []
_pool_index: int = 0
_dead_keys: set[str] = set()


def _split_keys(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[\s,;]+", raw or ""):
        k = part.strip().strip('"').strip("'")
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def load_serper_api_keys(
    *,
    explicit: str = "",
    env: bool = True,
) -> list[str]:
    """
    Build ordered Serper key pool.

    Priority: ``explicit`` ctor arg → ``SERPER_API_KEYS`` → ``SERPER_API_KEY`` /
    ``FINKEY_SERPER_API_KEY``.
    """
    keys: list[str] = []
    if explicit.strip():
        keys.extend(_split_keys(explicit))
    if env:
        keys.extend(_split_keys(os.getenv("SERPER_API_KEYS", "") or ""))
        for single in (
            os.getenv("SERPER_API_KEY", "") or "",
            os.getenv("FINKEY_SERPER_API_KEY", "") or "",
        ):
            keys.extend(_split_keys(single))
    # Dedupe preserve order
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def reset_serper_key_pool_for_tests() -> None:
    """Test helper — clear process-wide rotation state."""
    global _pool_keys, _pool_index, _dead_keys
    with _pool_lock:
        _pool_keys = []
        _pool_index = 0
        _dead_keys = set()


def _ensure_pool(seed_keys: list[str]) -> None:
    global _pool_keys, _pool_index
    with _pool_lock:
        if not _pool_keys and seed_keys:
            _pool_keys = list(seed_keys)
            _pool_index = 0
        elif seed_keys:
            # Merge any newly configured keys without resetting dead set.
            for k in seed_keys:
                if k not in _pool_keys:
                    _pool_keys.append(k)


def _active_key() -> str:
    global _pool_index
    with _pool_lock:
        if not _pool_keys:
            return ""
        n = len(_pool_keys)
        for _ in range(n):
            k = _pool_keys[_pool_index % n]
            if k not in _dead_keys:
                return k
            _pool_index = (_pool_index + 1) % n
        return ""


def _mark_key_dead(key: str, *, reason: str) -> str:
    """Mark key exhausted and advance to next live key. Returns next key or ''."""
    global _pool_index
    with _pool_lock:
        if key:
            _dead_keys.add(key)
            logger.warning(
                "Serper key exhausted (%s…): %s — rotating (%d/%d dead)",
                key[:8],
                reason[:120],
                len(_dead_keys),
                len(_pool_keys),
            )
        if not _pool_keys:
            return ""
        n = len(_pool_keys)
        for _ in range(n):
            _pool_index = (_pool_index + 1) % n
            nxt = _pool_keys[_pool_index]
            if nxt not in _dead_keys:
                # Keep SERPER_API_KEY pointing at the live key for other readers.
                os.environ["SERPER_API_KEY"] = nxt
                return nxt
        return ""


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url or ""


class SerperSearchBackend:
    def __init__(self, api_key: str = "", endpoint: str = "") -> None:
        # Ctor key seeds the pool; env SERPER_API_KEYS adds the rest for rotation.
        seed = load_serper_api_keys(explicit=api_key or "", env=True)
        _ensure_pool(seed)
        self._key = _active_key() or (seed[0] if seed else "")
        self._endpoint = (endpoint or os.getenv("SERPER_API_URL", "").strip() or _SERPER_URL).strip()

    @property
    def name(self) -> str:
        return "serper"

    def configured(self) -> bool:
        return bool(_active_key() or self._key)

    def search(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        date_restrict: str,
    ) -> list[SearchResult]:
        q = (query or "").strip()
        if len(q) > 240:
            q = q[:240].rsplit(" ", 1)[0]
        if not q:
            return []

        payload = {
            "q": q,
            "num": min(max(1, max_results), 50),
            "hl": "ru" if language == "lang_ru" else "en",
            "autocorrect": True,
        }
        # Google qdr знает только d/w/m/y-гранулярность; d7 (≈ неделя) мапим в w1,
        # иначе фильтр свежести молча терялся на новостных/рыночных запросах.
        _date_map = {"d7": "w1", "m6": "m3"}
        date_norm = _date_map.get(date_restrict, date_restrict)
        if date_norm in ("d1", "w1", "m1", "m3"):
            payload["tbs"] = f"qdr:{date_norm}"

        # Try current key; on credits/auth failure rotate and retry within this call.
        body: dict | None = None
        last_exc: BaseException | None = None
        tried: set[str] = set()
        while True:
            key = _active_key() or self._key
            if not key or key in tried:
                break
            tried.add(key)
            self._key = key
            try:
                body = self._post_json(self._endpoint, payload, api_key=key)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if self._is_rotatable_client_error(exc):
                    nxt = _mark_key_dead(key, reason=str(exc))
                    if nxt:
                        continue
                    logger.warning("Serper key pool exhausted after %d key(s)", len(tried))
                    raise
                logger.warning("Serper search failed: %s", exc)
                return []

        if body is None:
            if last_exc is not None and self._is_rotatable_client_error(last_exc):
                raise last_exc
            return []

        rows = body.get("organic") or []
        results: list[SearchResult] = []
        seen_domains: set[str] = set()

        # Прямые ответы Google (answerBox / knowledgeGraph) — самый точный сигнал
        # для «быстрых фактов»; идут первыми с максимальной релевантностью.
        results.extend(self._direct_answer_results(body))

        for i, item in enumerate(rows):
            if len(results) >= max_results:
                break
            url = (item.get("link") or "").strip()
            title = (item.get("title") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            if len(snippet) < 20:
                continue
            domain = _extract_domain(url)
            if domain and domain in seen_domains:
                continue
            if domain:
                seen_domains.add(domain)

            snippet = re.sub(r"\s+", " ", snippet).strip()
            date = item.get("date")
            results.append(
                SearchResult(
                    title=title or domain or "result",
                    url=url,
                    snippet=snippet[:2000],
                    date=str(date) if date else None,
                    source=domain or "serper",
                    relevance=max(0.0, 1.0 - (i * 0.08)),
                )
            )

        return results

    @staticmethod
    def _is_rotatable_client_error(exc: BaseException) -> bool:
        """Credits / auth failures → rotate to next SERPER_API_KEYS entry."""
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None) if resp is not None else None
        if status is None:
            status = getattr(exc, "code", None)
        if status in (401, 402, 403):
            return True
        text = ""
        try:
            if resp is not None:
                text = (getattr(resp, "text", None) or "")[:500].lower()
        except Exception:
            text = ""
        if not text:
            text = str(exc).lower()
        if status == 400 and any(
            s in text
            for s in (
                "not enough credits",
                "insufficient credits",
                "quota",
                "payment",
                "invalid api key",
                "unauthorized",
            )
        ):
            return True
        return False

    # Back-compat alias used by older call sites / tests.
    _is_hard_client_error = _is_rotatable_client_error

    @staticmethod
    def _direct_answer_results(body: dict) -> list[SearchResult]:
        """
        Serper отдаёт готовые ответы Google помимо organic:
          • answerBox — snippet/answer (курс, погода, счёт, определение)
          • knowledgeGraph — карточка сущности (кто такой X, компания Y)
        Раньше это выбрасывалось — модель получала только сырые снипеты и
        ошибалась на простых фактах.
        """
        out: list[SearchResult] = []

        ab = body.get("answerBox")
        if isinstance(ab, dict):
            answer = str(ab.get("answer") or "").strip()
            snippet = str(ab.get("snippet") or "").strip()
            title = str(ab.get("title") or "").strip()
            url = str(ab.get("link") or "").strip()
            parts = [p for p in (title, answer, snippet) if p]
            blob = ". ".join(dict.fromkeys(parts))
            blob = re.sub(r"\s+", " ", blob).strip()
            if len(blob) >= 8:
                out.append(
                    SearchResult(
                        title=title or "Google answer",
                        url=url,
                        snippet=blob[:2400],
                        date=str(ab.get("date")) if ab.get("date") else None,
                        source=_extract_domain(url) or "google_answer_box",
                        relevance=1.0,
                    )
                )

        kg = body.get("knowledgeGraph")
        if isinstance(kg, dict):
            kg_title = str(kg.get("title") or "").strip()
            kg_type = str(kg.get("type") or "").strip()
            kg_desc = str(kg.get("description") or "").strip()
            url = str(kg.get("descriptionLink") or kg.get("website") or "").strip()
            attrs = kg.get("attributes")
            attr_lines: list[str] = []
            if isinstance(attrs, dict):
                for k, v in list(attrs.items())[:10]:
                    ks, vs = str(k).strip(), str(v).strip()
                    if ks and vs:
                        attr_lines.append(f"{ks}: {vs}")
            pieces = [p for p in (kg_type, kg_desc, "; ".join(attr_lines)) if p]
            blob = ". ".join(pieces)
            blob = re.sub(r"\s+", " ", blob).strip()
            if kg_title and len(blob) >= 8:
                out.append(
                    SearchResult(
                        title=kg_title,
                        url=url,
                        snippet=blob[:2400],
                        date=None,
                        source=_extract_domain(url) or "google_knowledge_graph",
                        relevance=0.98,
                    )
                )

        for item in (body.get("topStories") or [])[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("link") or "").strip()
            if not title or not url:
                continue
            src = str(item.get("source") or "").strip()
            date = str(item.get("date") or "").strip()
            out.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=f"{title} — {src}".strip(" —"),
                    date=date or None,
                    source=_extract_domain(url) or src or "top_stories",
                    relevance=0.9,
                )
            )

        return out

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
        return False

    def _post_json(self, url: str, payload: dict, *, api_key: str = "") -> dict:
        try:
            max_retries = int(os.getenv("SERPER_MAX_RETRIES", "2"))
        except ValueError:
            max_retries = 2
        try:
            base_sleep = float(os.getenv("SERPER_RETRY_BASE_SEC", "0.5"))
        except ValueError:
            base_sleep = 0.5

        key = (api_key or self._key or "").strip()
        last_exc: BaseException | None = None
        for attempt in range(max_retries + 1):
            try:
                return self._post_json_once(url, payload, api_key=key)
            except Exception as exc:
                last_exc = exc
                if self._is_rotatable_client_error(exc):
                    raise
                if attempt >= max_retries or not self._is_retryable_rate_limit(exc):
                    raise
                delay = base_sleep * (2**attempt) + random.uniform(0, 0.25)
                logger.warning(
                    "Serper HTTP 429, backoff retry %s/%s in %.2fs",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _post_json_once(self, url: str, payload: dict, *, api_key: str = "") -> dict:
        import urllib.request

        key = (api_key or self._key or "").strip()
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "X-API-KEY": key,
            "Content-Type": "application/json",
            "User-Agent": "FinKeyAI/1.0",
        }

        try:
            import requests as _rq

            resp = _rq.post(url, json=payload, headers=headers, timeout=12)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            pass

        try:
            import httpx

            with httpx.Client(timeout=12) as client:
                r = client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                return r.json()
        except ImportError:
            pass

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode())
