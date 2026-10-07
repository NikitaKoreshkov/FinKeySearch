# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
URL quality signals for web search SERP ranking and BrowsingOperator entry URL.

Operators waste steps on social/auth-wall pages; SERP snippets from those domains
often pollute grounding.  No site-specific hacks beyond host/path classes.

When ``media_kind`` is set (``video`` / ``image`` / ``mixed``), YouTube and stock /
mood-board hosts are allowed for the operator and boosted in SERP scoring.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

_OPERATOR_ALWAYS_BLOCKED_SUFFIXES: tuple[str, ...] = (
    "instagram.com",
    "tiktok.com",
    "tiktokcdn.com",
    "pinimg.com",
    "snapchat.com",
    "threads.net",
    "fb.watch",
    "ytimg.com",
)

_PINTEREST_SUFFIXES: tuple[str, ...] = (
    "pinterest.com",
    "pinterest.ru",
)

_VIDEO_HOST_SUFFIXES: tuple[str, ...] = (
    "youtube.com",
    "youtu.be",
)

_MEDIA_OPERATOR_VIDEO_SUFFIXES: tuple[str, ...] = (
    "youtube.com",
    "youtu.be",
    "m.youtube.com",
    "rutube.ru",
    "vimeo.com",
)

_MEDIA_OPERATOR_IMAGE_SUFFIXES: tuple[str, ...] = (
    "pinterest.com",
    "pinterest.ru",
    "freepik.com",
    "unsplash.com",
    "pixabay.com",
    "shutterstock.com",
    "depositphotos.com",
    "istockphoto.com",
    "stock.adobe.com",
    "gettyimages.com",
)

_MEDIA_STOCK_IMAGE_BOOST_SUFFIXES: tuple[str, ...] = tuple(
    s for s in _MEDIA_OPERATOR_IMAGE_SUFFIXES if "pinterest" not in s
)

_OPERATOR_DISCOURAGED_SUFFIXES: tuple[str, ...] = (
    "facebook.com",
    "m.facebook.com",
    "l.facebook.com",
    "reddit.com",
    "redd.it",
    "twitter.com",
    "x.com",
    "mobile.twitter.com",
    "t.me",
    "telegram.me",
    "linkedin.com",
    "lnk.bio",
    "linktr.ee",
)

_LOGIN_WALL_MARKERS: tuple[str, ...] = (
    "accounts/login",
    "/oauth/",
    "login?next=",
    "/signin?",
    "/signup?",
    "/i/flow/login",
    "/authorization",
)

# Path patterns that are almost never the answer page (wrong-asset converters, tracking).
# Not a domain allowlist — structural noise only. The answer model judges source quality.
_STRUCTURAL_NOISE_PATH_RE = re.compile(
    r"/(converter|convert)/(shib|doge|pepe|meme)\b|"
    r"/i/flow/login|/accounts/login",
    re.I,
)

# Thin aggregator / SEO farm path markers (structural — not topic allowlists).
_THIN_AGGREGATOR_PATH_RE = re.compile(
    r"/(tag|tags|category|categories|author|authors|user|users|search|s|hashtag)/|"
    r"/(amp)/|"
    r"/page/\d+",
    re.I,
)

# Official / primary-source path shapes (investor relations, press, stats).
# Avoid bare "/report/" — SEO market teaser pages use that heavily.
_PRIMARY_PATH_RE = re.compile(
    r"/(investor(?:-relations)?|investors|ir|press(?:-releases?)?|newsroom|"
    r"media(?:-center)?|about(?:-us)?/news|filings?|sec-filings?|"
    r"statistics?|stats|datasets?|publications?|"
    r"annual[-_]?reports?|financial[-_]?reports?|earnings)/",
    re.I,
)

_OFFICIAL_STATS_HOST_RE = re.compile(
    r"(^|\.)("
    # Structural: major national / multilateral stats & filings hosts worldwide
    # (not a topic or country allowlist for SERP routing).
    r"sec\.gov|census\.gov|bls\.gov|data\.gov|"
    r"worldbank\.org|imf\.org|oecd\.org|who\.int|un\.org|"
    r"gov\.uk|ons\.gov\.uk|data\.gov\.uk|"
    r"europa\.eu|ec\.europa\.eu|eurostat\.ec\.europa\.eu|"
    r"destatis\.de|insee\.fr|istat\.it|abs\.gov\.au|statcan\.gc\.ca|"
    r"stat\.gov\.cn|e\-stat\.go\.jp|mospi\.gov\.in|"
    r"gov\.kz|stat\.gov\.kz|egov\.kz|"
    r"rosstat\.gov\.ru|gks\.ru"
    r")$",
    re.I,
)

# Thin SEO "market report" teaser pages — structural, not brand allowlists.
_SEO_MARKET_REPORT_PATH_RE = re.compile(
    r"/(market[-_]?(size|share|report|outlook|forecast|analysis)|"
    r"industry[-_]?(report|analysis|outlook)|research[-_]?report|"
    r"press-releases?/[^/]+$)",
    re.I,
)

_IR_HOST_RE = re.compile(
    r"^(investor|investors|ir)\.|(\.|^)(investor|investors|ir)\.",
    re.I,
)


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().split(":")[0]
    except Exception:
        return ""


def _path_query(url: str) -> str:
    try:
        p = urlparse(url)
        return f"{p.path.lower()}?{p.query.lower()}"
    except Exception:
        return ""


def _suffix_match(host: str, suffixes: tuple[str, ...]) -> bool:
    host = host.lower().strip(".")
    if not host:
        return False
    for suf in suffixes:
        if host == suf or host.endswith("." + suf):
            return True
    return False


def _media_allows_operator_host(host: str, media_kind: str) -> bool:
    if not media_kind:
        return False
    if media_kind == "video":
        return _suffix_match(host, _MEDIA_OPERATOR_VIDEO_SUFFIXES)
    if media_kind == "image":
        return _suffix_match(host, _MEDIA_OPERATOR_IMAGE_SUFFIXES)
    if media_kind == "mixed":
        return _suffix_match(
            host,
            _MEDIA_OPERATOR_VIDEO_SUFFIXES + _MEDIA_OPERATOR_IMAGE_SUFFIXES,
        )
    return False


def is_login_wall_url(url: str) -> bool:
    pq = _path_query(url)
    return any(m in pq for m in _LOGIN_WALL_MARKERS)


def is_blocked_for_operator_browse(url: str, media_kind: Optional[str] = None) -> bool:
    if not url.startswith(("http://", "https://")):
        return True
    host = _host(url)
    if not host:
        return True
    if _suffix_match(host, _OPERATOR_ALWAYS_BLOCKED_SUFFIXES):
        return True
    if _suffix_match(host, _PINTEREST_SUFFIXES):
        if _media_allows_operator_host(host, media_kind or ""):
            pass
        else:
            return True
    if _suffix_match(host, _VIDEO_HOST_SUFFIXES):
        if _media_allows_operator_host(host, media_kind or ""):
            pass
        else:
            return True
    if is_login_wall_url(url):
        return True
    return False


def _is_gov_or_edu_host(host: str) -> bool:
    h = (host or "").lower().strip(".")
    if not h:
        return False
    # TLD / public-suffix shapes: .gov, .gov.xx, .edu, .edu.xx, .ac.xx
    labels = h.split(".")
    if len(labels) >= 2 and labels[-1] in ("gov", "edu"):
        return True
    if len(labels) >= 3 and labels[-2] in ("gov", "edu", "ac"):
        return True
    return False


def is_primary_source_url(url: str) -> bool:
    """
    Structural primary-source shape for enrich/synth priority.
    Not a brand allowlist: TLD/path/host-shape only.
    """
    if not (url or "").startswith(("http://", "https://")):
        return False
    host = _host(url)
    path = _path_query(url)
    if _is_gov_or_edu_host(host):
        return True
    if _OFFICIAL_STATS_HOST_RE.search(host):
        return True
    if host == "github.com" or host.endswith(".github.com"):
        return True
    if _IR_HOST_RE.search(host):
        return True
    if _PRIMARY_PATH_RE.search(path):
        return True
    return False


def serp_url_quality_adjustment(url: str, media_kind: Optional[str] = None) -> float:
    """
    Structural SERP adjust only — NOT editorial / topic trust.

    Boost primary-source *shapes* (.gov/.edu, IR/press/stats paths, known stats hosts).
    Demote pages that cannot be read (login walls, shortlinks, feed-only social)
    or that match clear wrong-asset / thin-aggregator paths.
    Do NOT boost named news brands or topic allowlists.
    """
    if not url.startswith(("http://", "https://")):
        return -0.5
    host = _host(url)
    path = _path_query(url)
    adj = 0.0
    if _suffix_match(host, _OPERATOR_ALWAYS_BLOCKED_SUFFIXES):
        adj -= 2.5
    elif _suffix_match(host, _PINTEREST_SUFFIXES):
        if media_kind in ("image", "mixed"):
            adj += 0.5  # intent match when user asked for images
        else:
            adj -= 2.8
    elif _suffix_match(host, _VIDEO_HOST_SUFFIXES):
        if media_kind in ("video", "mixed"):
            adj += 0.8  # intent match when user asked for video
        else:
            adj -= 1.05
    elif _suffix_match(host, _OPERATOR_DISCOURAGED_SUFFIXES):
        # Harder demote: social feeds rarely carry extractable primary facts.
        adj -= 1.35
    if media_kind in ("image", "mixed") and _suffix_match(host, _MEDIA_STOCK_IMAGE_BOOST_SUFFIXES):
        adj += 0.4
    if is_login_wall_url(url):
        adj -= 1.1
    if _STRUCTURAL_NOISE_PATH_RE.search(path):
        adj -= 2.0
    if _THIN_AGGREGATOR_PATH_RE.search(path):
        adj -= 0.55
    if _SEO_MARKET_REPORT_PATH_RE.search(path) and not is_primary_source_url(url):
        # Paywalled teaser / SEO market-size pages — weak extractable grounding.
        adj -= 0.75
    if host in ("t.co", "bit.ly", "goo.gl", "ow.ly", "tinyurl.com"):
        adj -= 1.4

    # Primary-source structural boosts (no topic brand lists).
    if _is_gov_or_edu_host(host):
        adj += 1.15
    if _OFFICIAL_STATS_HOST_RE.search(host):
        adj += 0.75
    if _IR_HOST_RE.search(host):
        adj += 0.9
    if _PRIMARY_PATH_RE.search(path):
        adj += 0.65
    if host == "github.com" or host.endswith(".github.com"):
        # Repo/API pages are high-signal for OSS metrics (synth also has structured API).
        adj += 0.55
    return adj


def operator_entry_priority_score(
    url: str,
    *,
    base_relevance: float = 0.5,
    media_kind: Optional[str] = None,
) -> float:
    """
    Higher = better first hop for BrowsingOperator. Blocked URLs must never reach here.
    """
    if is_blocked_for_operator_browse(url, media_kind=media_kind):
        return float("-inf")
    host = _host(url)
    score = float(base_relevance) + serp_url_quality_adjustment(url, media_kind=media_kind)
    path = urlparse(url).path.lower()
    if host and re.search(r"/(news|world|politics|business|live|articles?)/", path):
        score += 0.15
    return score
