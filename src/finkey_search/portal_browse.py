# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Structured routing for official ticketing portals — not prompt prose, but URLs + goals.

When the user clearly means Ticketon (or other known portals) we construct a deep-link
URL directly to the city+date listing, so BrowsingOperator starts at the exact right page
instead of navigating from the homepage.

Environment:
  • ``FINKEY_PORTAL_OPERATOR_MAX_STEPS`` — floor for operator steps on portal tasks (default ``28``).
  • ``FINKEY_PORTAL_LOCAL_TZ`` — IANA timezone for DATE_TARGET_ISO (default ``Asia/Almaty``).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # pragma: no cover


@dataclass(frozen=True)
class CinemaPortalBrowsePlan:
    portal_host: str
    entry_url: str
    city_hint: Optional[str]
    city_slug: Optional[str]
    date_kind: str


_TICKETON = re.compile(
    r"(?i)(?:https?://)?(?:www\.)?ticketon\.kz\b|\bticketon\b|тикетон|тикитон|тикитом",
)
_FILM_SCHEDULE = re.compile(
    r"(?i)(фильм|фильмы|кино|сеанс|сеансы|афиш|билет|билеты|показ|показы"
    r"|кинотеатр|расписан|что\s+покажут|что\s+идёт|что\s+идет)",
)
_DATE_HINT = re.compile(r"(?i)(завтра|послезавтра|сегодня)")
_OPEN_WEB = re.compile(r"(?i)(открой|открыть|зайди|зайти|перейди|перейти|посмотри|посмотреть)")

_CITY_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"(?i)\bалматы\b|\bалмат[аы]\b"),               "Алматы",           "almaty"),
    (re.compile(r"(?i)\bастана\b|\bнур[- ]султан\b|\bnur[- ]sultan\b"), "Астана",   "astana"),
    (re.compile(r"(?i)\bшымкент\b"),                             "Шымкент",          "shymkent"),
    (re.compile(r"(?i)\bкараганда\b"),                           "Караганда",        "karaganda"),
    (re.compile(r"(?i)\bактобе\b"),                              "Актобе",           "aktobe"),
    (re.compile(r"(?i)\bпавлодар\b"),                            "Павлодар",         "pavlodar"),
    (re.compile(r"(?i)\bтараз\b"),                               "Тараз",            "taraz"),
    (re.compile(r"(?i)\bатырау\b"),                              "Атырау",           "atyrau"),
    (re.compile(r"(?i)\bкостанай\b"),                            "Костанай",         "kostanay"),
    (re.compile(r"(?i)\bуст[- ]каменогорск\b|\bоскемен\b"),     "Усть-Каменогорск", "ust-kamenogorsk"),
    (re.compile(r"(?i)\bактау\b"),                               "Актау",            "aktau"),
    (re.compile(r"(?i)\bуральск\b"),                             "Уральск",          "uralsk"),
    (re.compile(r"(?i)\bпетропавловск\b"),                       "Петропавловск",    "petropavlovsk"),
    (re.compile(r"(?i)\bкокшетау\b"),                            "Кокшетау",         "kokshetau"),
    (re.compile(r"(?i)\bтуркестан\b"),                           "Туркестан",        "turkestan"),
    (re.compile(r"(?i)\bсемей\b"),                               "Семей",            "semey"),
]


def _extract_city(text: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (label, url_slug) or (None, None)."""
    for pat, label, slug in _CITY_PATTERNS:
        if pat.search(text):
            return label, slug
    return None, None


def _date_kind(text: str) -> str:
    t = text.lower()
    if "послезавтра" in t:
        return "day_after"
    if "завтра" in t:
        return "tomorrow"
    if "сегодня" in t:
        return "today"
    return "unspecified"


def _local_today() -> "datetime.date":
    tz_name = (os.environ.get("FINKEY_PORTAL_LOCAL_TZ") or "Asia/Almaty").strip() or "Asia/Almaty"
    if ZoneInfo is None:
        return datetime.now().date()
    try:
        return datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        return datetime.now().date()


def format_target_date_iso(kind: str) -> str:
    """Calendar date for UI filters (local TZ)."""
    base = _local_today()
    if kind == "tomorrow":
        d = base + timedelta(days=1)
    elif kind == "day_after":
        d = base + timedelta(days=2)
    elif kind == "today":
        d = base
    else:
        d = base + timedelta(days=1)
    return d.isoformat()


def _build_ticketon_url(city_slug: Optional[str], date_iso: str) -> str:
    """
    Construct a deep-link URL for Ticketon cinema listing.
    e.g. https://ticketon.kz/astana/cinema?date_from=2026-05-15
    Falls back to homepage if city is unknown.
    """
    if city_slug:
        return f"https://ticketon.kz/{city_slug}/cinema?date_from={date_iso}"
    return "https://ticketon.kz/"


def resolve_cinema_portal_plan(
    user_message: str, *, aux_text: str = ""
) -> Optional[CinemaPortalBrowsePlan]:
    """aux_text may contain the classifier/SERP query (e.g. ``site:ticketon.kz``)."""
    msg = (user_message or "").strip()
    blob_for_brand = f"{msg}\n{(aux_text or '').strip()}".strip()
    if len(blob_for_brand) < 4:
        return None
    if not _TICKETON.search(blob_for_brand):
        return None

    film = bool(_FILM_SCHEDULE.search(msg))
    dt = bool(_DATE_HINT.search(msg))
    city_label, city_slug = _extract_city(msg)
    openv = bool(_OPEN_WEB.search(msg))

    wants_listing = film or (dt and city_label is not None) or (openv and film)
    if not wants_listing:
        return None

    dk = _date_kind(msg)
    if dk == "unspecified":
        dk = "today"   # default to today so we get current schedule, not tomorrow

    date_iso = format_target_date_iso(dk)
    entry_url = _build_ticketon_url(city_slug, date_iso)

    return CinemaPortalBrowsePlan(
        portal_host="ticketon.kz",
        entry_url=entry_url,
        city_hint=city_label,
        city_slug=city_slug,
        date_kind=dk,
    )


def augment_classifier_query_for_portals(user_message: str, search_query: str) -> str:
    """Bias SERP retrieval toward the official domain when a portal crawl is intended."""
    if resolve_cinema_portal_plan(user_message, aux_text=search_query) is None:
        return search_query
    q = (search_query or "").strip()
    low = q.lower()
    if "ticketon.kz" in low or "site:ticketon" in low:
        return search_query
    return f"{q} site:ticketon.kz".strip()


def build_portal_operator_goal(user_message: str, plan: CinemaPortalBrowsePlan) -> str:
    city = plan.city_hint or ""
    date_iso = format_target_date_iso(plan.date_kind)
    city_line = (
        f"CITY_TARGET={city} (slug={plan.city_slug})"
        if city
        else "CITY_TARGET=(city unknown — navigate to the city listing using site UI)"
    )
    um = user_message.strip()
    return (
        "PORTAL_TASK=ticketon.kz_afisha\n"
        f"DATE_TARGET_ISO={date_iso}\n"
        f"{city_line}\n"
        f"START_URL_ALREADY_CORRECT=true  ← you are already on the right listing page\n"
        "OBJECTIVE: The browser has already opened the Ticketon cinema listing for "
        f"{city or 'the requested city'} on {date_iso}. "
        "Your job is to EXTRACT the data that is already visible:\n"
        "  1. Dismiss any cookie/promo popup immediately (click ×, 'Закрыть', or JS remove).\n"
        "  2. The films are shown as CARDS/POSTERS — there are NO HTML tables, so extract_table "
        "     will return no_tables_found. That is expected. Read from MAIN_CONTENT in observe.\n"
        "  3. scroll DOWN 4–6 times to reveal ALL films. After each scroll, observe and note "
        "     film title + price from MAIN_CONTENT.\n"
        "  4. DO NOT re-navigate to the homepage or a different city.\n"
        f"  5. If a date filter bar is visible, verify selected date is {date_iso}; "
        f"     if not, click the correct date tile first.\n"
        "  6. finish() with a COMPLETE list: every film title, showtimes, ticket price — no invented data.\n"
        f"USER_MESSAGE:\n{um}"
    )
