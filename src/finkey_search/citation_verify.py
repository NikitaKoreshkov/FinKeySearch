# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Post-generation checks: model answer must mention allowed citation URLs / hosts.

Used by the host app when ``verified_sources_required`` is True for the turn.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse


def _host_norm(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def answer_references_allowed_web_sources(answer: str, citation_urls: list[str]) -> bool:
    """
    Returns True when ``answer`` contains at least one full allowed URL substring,
    a matching host substring, or an http(s) URL whose host overlaps allowed hosts.
    """
    ans = answer or ""
    if not citation_urls:
        return False

    allowed_hosts: set[str] = set()
    for u in citation_urls:
        u = (u or "").strip()
        if not u:
            continue
        if u in ans:
            return True
        hn = _host_norm(u)
        if hn:
            allowed_hosts.add(hn)
            if hn in ans.lower():
                return True

    if not allowed_hosts:
        return False

    for m in re.finditer(r"https?://[^\s\)\]\"\'>]+", ans, flags=re.IGNORECASE):
        raw = m.group(0).rstrip(".,;:\"'")
        cand = _host_norm(raw.split("?", 1)[0])
        if cand and cand in allowed_hosts:
            return True

    return False
