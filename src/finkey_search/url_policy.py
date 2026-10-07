# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
URL policy for browser automation — production-style guardrails.

Blocks:
  • non-http(s) schemes
  • localhost / loopback hostnames
  • link-local and private IPv4/IPv6 literals in the host
  • AWS-style metadata endpoint (169.254.169.254)
  • hostnames, резолвящиеся в приватные/loopback адреса (DNS rebinding,
    nip.io-стиль; отключается через FINKEY_URL_POLICY_DNS_CHECK=0)

Optional allowlist: only hosts ending with one of the given suffixes (e.g. ".wikipedia.org").
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


_METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata",
    }
)


def _ip_blocked(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _host_blocked_literal(host: str) -> bool:
    h = host.lower().rstrip(".")
    if not h:
        return True
    if h in _METADATA_HOSTS:
        return True
    if h == "localhost" or h.endswith(".localhost"):
        return True
    return _ip_blocked(h)


def _dns_check_enabled() -> bool:
    raw = (os.getenv("FINKEY_URL_POLICY_DNS_CHECK", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


_DNS_CACHE: dict[str, tuple[float, bool]] = {}
_DNS_CACHE_LOCK = threading.Lock()
_DNS_CACHE_TTL = 300.0
_DNS_CACHE_CAP = 2048


def _host_resolves_to_blocked(host: str) -> bool:
    """True if any A/AAAA record for the host is private / loopback / link-local.

    Closes the DNS-rebinding bypass of literal-IP checks (nip.io style hosts →
    169.254.169.254). An unresolvable host is not blocked — the request fails on
    its own. Results are cached for 5 minutes.
    """
    h = host.lower().rstrip(".")
    now = time.time()
    with _DNS_CACHE_LOCK:
        hit = _DNS_CACHE.get(h)
        if hit and hit[0] > now:
            return hit[1]
    try:
        infos = socket.getaddrinfo(h, None, proto=socket.IPPROTO_TCP)
        blocked = any(_ip_blocked(str(info[4][0])) for info in infos)
    except (socket.gaierror, OSError):
        blocked = False
    with _DNS_CACHE_LOCK:
        if len(_DNS_CACHE) >= _DNS_CACHE_CAP:
            _DNS_CACHE.clear()
        _DNS_CACHE[h] = (now + _DNS_CACHE_TTL, blocked)
    return blocked


def url_blocked_for_fetch(url: str) -> str:
    """SSRF check for HTTP fetchers: '' when allowed, otherwise a reason code.

    Applied to the original URL and to every redirect hop.
    """
    raw = (url or "").strip()
    if not raw:
        return "empty_url"
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return f"bad_scheme:{scheme or 'none'}"
    host = (parsed.hostname or "").lower()
    if not host:
        return "no_host"
    if _host_blocked_literal(host):
        return "blocked_host_private_or_loopback"
    if _dns_check_enabled() and _host_resolves_to_blocked(host):
        return "blocked_host_resolves_private"
    return ""


@dataclass
class URLPolicy:
    """Host-level rules applied before every navigation."""

    allowed_host_suffixes: Optional[list[str]] = None
    """If set, host must match at least one suffix (case-insensitive), e.g. ['.wikipedia.org', 'example.com']."""

    block_private_and_loopback: bool = True

    blocked_path_substrings: Optional[tuple[str, ...]] = (
        "/login", "/signin", "/signup", "/oauth", "/authorize",
        "/checkout", "/payment", "/billing", "/wallet",
        "/account/delete", "/password", "/reset-password",
    )

    def validate(self, url: str) -> tuple[bool, str]:
        """
        Returns (ok, reason). reason is empty when ok is True.
        """
        raw = (url or "").strip()
        if not raw:
            return False, "empty_url"
        if raw.startswith("//"):
            raw = "https:" + raw
        parsed = urlparse(raw)
        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            return False, f"bad_scheme:{scheme or 'none'}"
        host = (parsed.hostname or "").lower()
        if not host:
            return False, "no_host"

        if self.block_private_and_loopback and _host_blocked_literal(host):
            return False, "blocked_host_private_or_loopback"

        if self.block_private_and_loopback and _dns_check_enabled() and _host_resolves_to_blocked(host):
            return False, "blocked_host_resolves_private"

        if self.allowed_host_suffixes:
            ok = False
            for suf in self.allowed_host_suffixes:
                s = suf.lower().strip().lstrip(".")
                if host == s or host.endswith("." + s):
                    ok = True
                    break
            if not ok:
                return False, "host_not_in_allowlist"

        path = (parsed.path or "").lower()
        if self.blocked_path_substrings:
            for frag in self.blocked_path_substrings:
                if frag and frag.lower() in path:
                    return False, f"blocked_path:{frag}"

        return True, ""


def normalize_http_url(url: str) -> Optional[str]:
    """Return normalized https? URL string or None."""
    u = (url or "").strip()
    if u.startswith("//"):
        u = "https:" + u
    if not re.match(r"^https?://", u, re.I):
        return None
    return u
