# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Live-query shaping — structural (not topic allowlists).

When the user asks for *current* facts (weather now, live FX, spot price),
bias SERP toward "today/current/live" and demote climate/historical pages.
"""
from __future__ import annotations

import re

from .schema import SearchResult

# Model often pastes the full user essay into ``query``; Serper/Brave choke on that.
_ESSAY_QUERY_RE = re.compile(
    r"(проанализируй|проанализировать|собери\s+актуальн|на\s+основе\s+свежих|"
    r"обязательно\s+опирайся|структурируй|дай\s+прогноз|"
    r"\banalyze\b|\bresearch\b|\bbased\s+on\s+(fresh|recent|live)\b|"
    r"\bmust\s+cite\b|\bgive\s+a\s+forecast\b)",
    re.I,
)


def compact_serp_query(query: str, *, goal: str = "") -> str:
    """
    Keep SERP strings short. Full user goals stay in ``message`` for orch/deep-research;
    backends want a compact query (≤~180–240 chars), not an instruction essay.
    """
    from .classifier import _optimize_query
    from .schema import SearchCategory

    q = (query or "").strip()
    g = (goal or "").strip()
    if q and len(q) <= 180 and q.count(" ") <= 24 and not _ESSAY_QUERY_RE.search(q):
        return re.sub(r"\s+", " ", q).strip()[:240]
    base = q if q and (not g or len(q) <= len(g)) else g
    if not base:
        return ""
    # Drop instruction verbs so SERP is noun/topic-shaped.
    cleaned = re.sub(
        r"^(проанализируй|проанализировать|собери|собрать|найди|подбери|сделай|"
        r"investigate|analyze|research|gather|compare|find)\s+",
        "",
        base,
        flags=re.I,
    ).strip()
    # Cut trailing deliverable clauses ("и дай прогноз…", "на основе…").
    cleaned = re.split(
        r",?\s*(и\s+на\s+основе|и\s+дай|дай\s+прогноз|обязательно|структурируй|"
        r"based\s+on|give\s+a\s+forecast|must\s+cite)\b",
        cleaned,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" ,;:")
    if len(cleaned) < 8:
        cleaned = base
    compact = (_optimize_query(cleaned, SearchCategory.GENERAL) or cleaned).strip()
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) > 240:
        compact = compact[:240].rsplit(" ", 1)[0]
    return compact or base[:240]

_CURRENT_WEATHER_RE = re.compile(
    r"(погод\w*|weather|температур\w*|temperature|forecast)\b",
    re.I,
)
_NOW_RE = re.compile(
    r"(сейчас|current|right\s+now|today|сегодня|live|актуальн)",
    re.I,
)
_FX_RE = re.compile(
    r"(курс|exchange\s*rate|fx\b|eur\s*/?\s*usd|usd\s*/?\s*rub|доллар\w*\s+к\s+руб|"
    r"ключев\w*\s+ставк)",
    re.I,
)
_SPOT_RE = re.compile(
    r"(сколько\s+стоит|price\s+of|spot\s+price|котировк|brent|bitcoin|btc\b)",
    re.I,
)

_STALE_WEATHER_RE = re.compile(
    r"(climate|climatic|monthly\s+average|average\s+weather|historical\s+weather|"
    r"среднегодов|климат\w*|средн\w*\s+(месяч|температур)|normals?|"
    r"pogodaiklimat|climate-data|climate\.|yearly\s+weather)",
    re.I,
)
_STALE_FX_RE = re.compile(
    r"(historical\s+rates?|history\s+of|за\s+10\s+лет|архив\s+курс|chart\s+history)",
    re.I,
)
_LIVE_HINT_RE = re.compile(
    r"(today|current|now|live|сейчас|почас|hourly|accuweather|wunderground|"
    r"timeanddate|weather\.com|/weather/|meteoblue|msn\.com/weather)",
    re.I,
)


def is_current_fact_ask(message: str) -> bool:
    msg = message or ""
    if _CURRENT_WEATHER_RE.search(msg):
        return True
    if _FX_RE.search(msg):
        return True
    if _SPOT_RE.search(msg) and (_NOW_RE.search(msg) or len(msg) < 80):
        return True
    if _NOW_RE.search(msg) and len(msg) < 120:
        return True
    return False


def shape_live_search_query(message: str, query: str) -> str:
    """Append live/current hints when the ask is about now — no entity lists."""
    q = (query or message or "").strip()
    msg = message or ""
    low = q.lower()
    if _CURRENT_WEATHER_RE.search(msg):
        if not re.search(r"(current|today|сейчас|live|right\s+now)", low):
            q = f"{q} current weather today temperature".strip()
        # Prefer Latin place spelling when message is Cyrillic-only weather ask
        # by keeping original + English "weather today" suffix (already added).
    elif _FX_RE.search(msg):
        if not re.search(r"(current|live|today|сейчас|spot)", low):
            q = f"{q} current live rate today official".strip()
    elif _SPOT_RE.search(msg) and len(msg) < 100:
        if not re.search(r"(current|live|today|сейчас|spot|price)", low):
            q = f"{q} current price today".strip()
    return re.sub(r"\s+", " ", q).strip()[:240]


def demote_stale_current_rows(message: str, results: list[SearchResult]) -> list[SearchResult]:
    """
    Push climate/historical pages below live pages for current-fact asks.
    Prefer rows with live markers in title/url/snippet.
    """
    if not results or not is_current_fact_ask(message):
        return results
    weather = bool(_CURRENT_WEATHER_RE.search(message or ""))
    fx = bool(_FX_RE.search(message or ""))

    def _blob(r: SearchResult) -> str:
        return f"{r.title or ''} {r.url or ''} {r.snippet or ''}"

    def _stale(r: SearchResult) -> bool:
        blob = _blob(r)
        if weather and _STALE_WEATHER_RE.search(blob):
            return True
        if fx and _STALE_FX_RE.search(blob):
            return True
        return False

    def _live(r: SearchResult) -> bool:
        return bool(_LIVE_HINT_RE.search(_blob(r)))

    live = [r for r in results if _live(r) and not _stale(r)]
    mid = [r for r in results if not _stale(r) and r not in live]
    stale = [r for r in results if _stale(r)]
    if live or mid:
        return live + mid + stale
    return results


__all__ = [
    "compact_serp_query",
    "is_current_fact_ask",
    "shape_live_search_query",
    "demote_stale_current_rows",
]
