# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Lightweight HTTPS text extraction when Playwright/readability path is empty."""
from __future__ import annotations

import logging
import os
import re
import ssl
import urllib.error
import urllib.request
from urllib.parse import urljoin

from finkey_search.url_policy import url_blocked_for_fetch

logger = logging.getLogger(__name__)

_MAX_REDIRECT_HOPS = 6
_REDIRECT_CODES = (301, 302, 303, 307, 308)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 FinKeyAI/1.0"
)

_DEFAULT_ACCEPT_LANGUAGE = (
    "en-US;q=0.92,en;q=0.91,"
    "ru-RU;q=0.88,ru;q=0.87,"
    "kk-KZ;q=0.84,kk;q=0.83,"
    "de;q=0.80,fr;q=0.80,es;q=0.80,pt;q=0.80,"
    "zh-CN;q=0.80,zh;q=0.80,ja;q=0.80,ko;q=0.80,"
    "ar;q=0.80,tr;q=0.80,uk;q=0.80,uz;q=0.80,"
    "*;q=0.72"
)


def _accept_language_value() -> str:
    override = (os.getenv("FINKEY_HTTP_FETCH_ACCEPT_LANGUAGE") or "").strip()
    if override:
        return override
    return _DEFAULT_ACCEPT_LANGUAGE.replace("\n", "").replace(" ", "")


def html_fetch_headers() -> dict[str, str]:
    """Общие заголовки GET для HTML (можно переопределить только язык через env)."""
    return {
        "User-Agent":               _UA,
        "Accept":                   "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language":          _accept_language_value(),
        "Upgrade-Insecure-Requests": "1",
    }


def _clip_bytes(chunk: bytes, max_bytes: int) -> bytes:
    return chunk[:max_bytes]


class _SSRFGuardRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urllib-редиректы с SSRF-перепроверкой каждого хопа."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        reason = url_blocked_for_fetch(newurl)
        if reason:
            raise urllib.error.HTTPError(
                newurl, 403, f"redirect_blocked:{reason}", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_html_via_https(url: str, *, timeout_s: float = 12.0, max_bytes: int = 2_000_000) -> tuple[str, str]:
    """
    Returns ``(html_body, error_message)``.
    ``html_body`` is empty string on failure.

    Tries ``requests``, then ``httpx``, then stdlib urllib (sites often fail on urllib-only SSL).
    """
    if not url.startswith("https://"):
        return "", "non_https"
    ssrf = url_blocked_for_fetch(url)
    if ssrf:
        return "", f"ssrf_blocked:{ssrf}"

    errs: list[str] = []

    # Редиректы следуем вручную: каждый хоп перепроверяется SSRF-политикой,
    # иначе «хороший» URL может средиректить на 169.254.169.254/внутреннюю сеть.
    try:
        import requests as rq

        cur = url
        with rq.Session() as sess:
            for _hop in range(_MAX_REDIRECT_HOPS):
                r = sess.get(
                    cur,
                    headers=html_fetch_headers(),
                    timeout=timeout_s,
                    allow_redirects=False,
                )
                if r.status_code in _REDIRECT_CODES:
                    loc = (r.headers.get("Location") or "").strip()
                    if not loc:
                        return "", "redirect_without_location"
                    nxt = urljoin(cur, loc)
                    reason = url_blocked_for_fetch(nxt)
                    if reason:
                        return "", f"redirect_blocked:{reason}"
                    cur = nxt
                    continue
                r.raise_for_status()
                raw = _clip_bytes(r.content, max_bytes)
                return raw.decode("utf-8", errors="replace"), ""
            return "", "too_many_redirects"
    except ImportError:
        pass
    except Exception as exc:
        errs.append(f"requests:{type(exc).__name__}:{exc!s}"[:300])

    try:
        import httpx

        cur = url
        with httpx.Client(
            timeout=timeout_s,
            follow_redirects=False,
            headers=html_fetch_headers(),
        ) as client:
            for _hop in range(_MAX_REDIRECT_HOPS):
                r = client.get(cur)
                if r.status_code in _REDIRECT_CODES:
                    loc = (r.headers.get("Location") or "").strip()
                    if not loc:
                        return "", "redirect_without_location"
                    nxt = urljoin(cur, loc)
                    reason = url_blocked_for_fetch(nxt)
                    if reason:
                        return "", f"redirect_blocked:{reason}"
                    cur = nxt
                    continue
                r.raise_for_status()
                raw = _clip_bytes(r.content, max_bytes)
                return raw.decode("utf-8", errors="replace"), ""
            return "", "too_many_redirects"
    except ImportError:
        pass
    except Exception as exc:
        errs.append(f"httpx:{type(exc).__name__}:{exc!s}"[:300])

    h_urllib = {
        **html_fetch_headers(),
        "Accept-Encoding": "identity",
    }
    req = urllib.request.Request(url, headers=h_urllib, method="GET")
    ctx = ssl.create_default_context()
    opener = urllib.request.build_opener(
        _SSRFGuardRedirectHandler(),
        urllib.request.HTTPSHandler(context=ctx),
    )
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = _clip_bytes(resp.read(max_bytes + 2_000), max_bytes)
        return raw.decode("utf-8", errors="replace"), ""
    except urllib.error.HTTPError as exc:
        return "", f"http_{exc.code}"
    except Exception as exc:
        tag = type(exc).__name__
        errs.append(f"urllib:{tag}:{exc!s}"[:300])
        return "", errs[-1][:400] if errs else tag


_GAP_MARK = (
    "\n … ─── [пропущен фрагмент страницы: сохранены начало и конец текста] ─── … \n"
)


def excerpt_head_tail(blob: str, lim: int) -> str:
    """Не отрезать только «шапку»: при лимите даём начало + конец документа."""
    if lim <= 0 or len(blob) <= lim:
        return blob
    mark = _GAP_MARK
    budget = lim - len(mark)
    if budget < 400:
        return blob[:lim]
    h = budget // 2
    return blob[:h] + mark + blob[-(budget - h):]


def _page_extractor_pref() -> str:
    """FINKEY_PAGE_EXTRACTOR=trafilatura|legacy (default: trafilatura-when-imported)."""
    raw = (os.getenv("FINKEY_PAGE_EXTRACTOR", "trafilatura") or "").strip().lower()
    return raw if raw in ("trafilatura", "legacy") else "trafilatura"


_trafilatura_import_failed = False


def _trafilatura_extract(html: str) -> str | None:
    """
    Optional-dependency extractor. Returns cleaned text, or ``None`` on
    ImportError / no-content / any internal failure — the caller then falls
    back to the legacy regex strip. Never raises.
    """
    global _trafilatura_import_failed
    if _trafilatura_import_failed:
        return None
    try:
        import trafilatura  # optional dep — guarded, not in core requirements
    except ImportError:
        _trafilatura_import_failed = True
        return None
    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
    except Exception as exc:
        logger.debug("trafilatura extract failed: %s", exc)
        return None
    if not text or not text.strip():
        return None
    return text.strip()


def html_to_readable_text(html: str, *, max_chars: int) -> str:
    """Strip scripts/styles/tags — затем при лимите сохраняем начало и конец страницы.

    Extraction preference (``FINKEY_PAGE_EXTRACTOR``):
      • ``trafilatura`` (default) — main-article extraction whenever the
        optional package is importable; falls back to the legacy regex strip
        below whenever trafilatura returns None (missing dep, no content,
        internal error).
      • ``legacy`` — always the regex strip.
    """
    if not html.strip():
        return ""
    blob = ""
    if _page_extractor_pref() == "trafilatura":
        blob = _trafilatura_extract(html) or ""
    if not blob:
        blob = html
        blob = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", blob)
        blob = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", blob)
        blob = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", blob)
        blob = re.sub(r"<[^>]+>", " ", blob)
        blob = re.sub(r"\s+", " ", blob).strip()
    try:
        hard_strip = int(os.getenv("FINKEY_HTTP_FETCH_STRIP_HARD_CAP", "900000"))
    except ValueError:
        hard_strip = 900000
    hard_strip = max(60_000, min(hard_strip, 1_800_000))
    if len(blob) > hard_strip:
        blob = excerpt_head_tail(blob, hard_strip)
    if max_chars > 0:
        blob = excerpt_head_tail(blob, max_chars)
    return blob


def _default_http_page_chars() -> int:
    """Сколько символов сохранять после strip (см. excerpt_head_tail при переполнении)."""
    try:
        v = int(os.getenv("FINKEY_WEB_PAGE_ENRICH_MAX_CHARS", "500000"))
    except ValueError:
        v = 500_000
    return max(4_000, min(v, 500_000))


def fetch_page_text_http(
    url: str,
    *,
    timeout_s: float = 12.0,
    max_chars: int | None = None,
) -> tuple[str, str]:
    """HTTPS GET → текст: по умолчанию большой лимит; при переполнении — начало+конец страницы."""
    html, err = fetch_html_via_https(url, timeout_s=timeout_s)
    if err:
        logger.info("HTTP fetch failed url=%s err=%s", url[:120], err)
        return "", err or "fetch_failed"
    mc = _default_http_page_chars() if max_chars is None else max_chars
    text = html_to_readable_text(html, max_chars=max(1, mc))
    if len(text) < 40:
        return "", "thin_body"
    return text, ""
