# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Lazy singleton Redis client for internet web-search cache (shares REDIS_URL with memory)."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_client: Optional[Any] = None
_failed = False


def get_shared_redis_optional() -> Optional[Any]:
    """Return decode_responses=True redis client or None."""
    global _client, _failed
    with _lock:
        if _failed:
            return None
        if _client is not None:
            return _client
        url = (os.getenv("REDIS_URL") or "").strip()
        if not url:
            _failed = True
            return None
        try:
            import redis as redis_lib

            c = redis_lib.from_url(url, decode_responses=True)
            c.ping()
            _client = c
            logger.info("Web search Redis cache: connected")
            return _client
        except Exception as exc:
            logger.warning("Web search Redis unavailable: %s", exc)
            _failed = True
            return None


def reset_shared_redis_client_for_tests() -> None:
    global _client, _failed
    with _lock:
        _client = None
        _failed = False
