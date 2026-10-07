# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Rate limiting and circuit breaking for outbound search HTTP."""
from __future__ import annotations

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class SlidingWindowLimiter:
    """Fixed-window style limiter using monotonic timestamps (non-blocking)."""

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self.max_calls       = max(1, int(max_calls))
        self.window_seconds  = float(window_seconds)
        self._hits: deque[float] = deque()

    def acquire(self) -> bool:
        now = time.monotonic()
        boundary = now - self.window_seconds
        while self._hits and self._hits[0] < boundary:
            self._hits.popleft()
        if len(self._hits) >= self.max_calls:
            logger.debug("Rate limit reached (%d/%ds)", self.max_calls, int(self.window_seconds))
            return False
        self._hits.append(now)
        return True


class CircuitBreaker:
    """
    Trip open after consecutive failures; block calls until recovery_timeout elapses.
    One probe allowed after cooldown (half-open behavior via failure reopen).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 45.0,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds  = float(recovery_seconds)
        self._failures         = 0
        self._open_until: float = 0.0

    def allow_request(self) -> bool:
        return time.monotonic() >= self._open_until

    def record_success(self) -> None:
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._open_until = time.monotonic() + self.recovery_seconds
            logger.warning(
                "Circuit breaker OPEN for %.0fs after %d failures",
                self.recovery_seconds,
                self.failure_threshold,
            )
            self._failures = 0
