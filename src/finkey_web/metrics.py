# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Lightweight counters for internet retrieval — logs + programmatic snapshots."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InternetMetricsSnapshot:
    classifier_skips:       int = 0
    classifier_searches:    int = 0
    cache_hits:             int = 0
    redis_cache_hits:       int = 0
    searches_started:       int = 0
    backend_calls:          dict[str, int] = field(default_factory=dict)
    backend_rows:           dict[str, int] = field(default_factory=dict)
    backend_errors:         dict[str, int] = field(default_factory=dict)
    rerank_passes:          int = 0
    cross_check_runs:       int = 0
    snippet_sanitized:      int = 0
    page_enrich_attempts:   int = 0
    page_enrich_success:    int = 0
    page_enrich_structured_miss: int = 0
    page_enrich_http_fallback_ok: int = 0
    stealth_fallback_hits:  int = 0


class InternetMetrics:
    """Thread-safe counters for observability hooks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap = InternetMetricsSnapshot()
        self._backend_calls: dict[str, int] = {}
        self._backend_rows: dict[str, int] = {}
        self._backend_errors: dict[str, int] = {}

    @staticmethod
    def _otel_err() -> None:
        try:
            from finkey_web.telemetry import otel_note_backend_error
            otel_note_backend_error()
        except Exception:
            pass

    @staticmethod
    def _otel_skip() -> None:
        try:
            from finkey_web.telemetry import otel_note_classifier_skip
            otel_note_classifier_skip()
        except Exception:
            pass

    @staticmethod
    def _otel_search() -> None:
        try:
            from finkey_web.telemetry import otel_note_search_start
            otel_note_search_start()
        except Exception:
            pass

    def snapshot(self) -> InternetMetricsSnapshot:
        with self._lock:
            return InternetMetricsSnapshot(
                classifier_skips      = self._snap.classifier_skips,
                classifier_searches   = self._snap.classifier_searches,
                cache_hits            = self._snap.cache_hits,
                redis_cache_hits      = self._snap.redis_cache_hits,
                searches_started      = self._snap.searches_started,
                backend_calls         = dict(self._backend_calls),
                backend_rows          = dict(self._backend_rows),
                backend_errors        = dict(self._backend_errors),
                rerank_passes         = self._snap.rerank_passes,
                cross_check_runs      = self._snap.cross_check_runs,
                snippet_sanitized     = self._snap.snippet_sanitized,
                page_enrich_attempts  = self._snap.page_enrich_attempts,
                page_enrich_success   = self._snap.page_enrich_success,
                page_enrich_structured_miss=self._snap.page_enrich_structured_miss,
                page_enrich_http_fallback_ok=self._snap.page_enrich_http_fallback_ok,
                stealth_fallback_hits = self._snap.stealth_fallback_hits,
            )

    def record_classifier_skip(self) -> None:
        with self._lock:
            self._snap.classifier_skips += 1
        self._otel_skip()

    def record_classifier_search(self) -> None:
        with self._lock:
            self._snap.classifier_searches += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self._snap.cache_hits += 1

    def record_redis_cache_hit(self) -> None:
        with self._lock:
            self._snap.redis_cache_hits += 1

    def record_search_start(self) -> None:
        with self._lock:
            self._snap.searches_started += 1
        self._otel_search()

    def record_backend_call(self, backend_name: str) -> None:
        with self._lock:
            self._backend_calls[backend_name] = self._backend_calls.get(backend_name, 0) + 1

    def record_backend_success(self, backend_name: str, rows: int) -> None:
        with self._lock:
            self._backend_rows[backend_name] = self._backend_rows.get(backend_name, 0) + rows

    def record_backend_error(self, backend_name: str) -> None:
        with self._lock:
            self._backend_errors[backend_name] = self._backend_errors.get(backend_name, 0) + 1
        self._otel_err()

    def record_cross_check(self) -> None:
        with self._lock:
            self._snap.cross_check_runs += 1

    def record_snippet_sanitize(self, count: int = 1) -> None:
        with self._lock:
            self._snap.snippet_sanitized += max(0, count)

    def record_rerank(self) -> None:
        with self._lock:
            self._snap.rerank_passes += 1

    def record_page_enrich_attempt(self) -> None:
        with self._lock:
            self._snap.page_enrich_attempts += 1

    def record_page_enrich_success(self) -> None:
        with self._lock:
            self._snap.page_enrich_success += 1

    def record_page_enrich_structured_miss(self) -> None:
        with self._lock:
            self._snap.page_enrich_structured_miss += 1

    def record_page_enrich_http_fallback_ok(self) -> None:
        with self._lock:
            self._snap.page_enrich_http_fallback_ok += 1

    def record_stealth_fallback(self) -> None:
        with self._lock:
            self._snap.stealth_fallback_hits += 1

    def as_dict(self) -> dict[str, Any]:
        s = self.snapshot()
        return {
            "classifier_skips":      s.classifier_skips,
            "classifier_searches":   s.classifier_searches,
            "cache_hits":            s.cache_hits,
            "redis_cache_hits":      s.redis_cache_hits,
            "searches_started":      s.searches_started,
            "backend_calls":         s.backend_calls,
            "backend_rows":          s.backend_rows,
            "backend_errors":        s.backend_errors,
            "rerank_passes":         s.rerank_passes,
            "cross_check_runs":      s.cross_check_runs,
            "snippet_sanitized":     s.snippet_sanitized,
            "page_enrich_attempts":  s.page_enrich_attempts,
            "page_enrich_success":   s.page_enrich_success,
            "page_enrich_structured_miss": s.page_enrich_structured_miss,
            "page_enrich_http_fallback_ok": s.page_enrich_http_fallback_ok,
            "stealth_fallback_hits": s.stealth_fallback_hits,
        }

    def to_prometheus_text(self) -> str:
        """Minimal text exposition for scraping (counters as gauges snapshot)."""
        s = self.snapshot()
        lines = [
            f"finkey_internet_classifier_skips_total {s.classifier_skips}",
            f"finkey_internet_classifier_searches_total {s.classifier_searches}",
            f"finkey_internet_cache_hits_total {s.cache_hits}",
            f"finkey_internet_redis_cache_hits_total {s.redis_cache_hits}",
            f"finkey_internet_searches_started_total {s.searches_started}",
            f"finkey_internet_rerank_passes_total {s.rerank_passes}",
            f"finkey_internet_cross_check_runs_total {s.cross_check_runs}",
            f"finkey_internet_snippet_sanitized_total {s.snippet_sanitized}",
            f"finkey_internet_page_enrich_attempts_total {s.page_enrich_attempts}",
            f"finkey_internet_page_enrich_success_total {s.page_enrich_success}",
            f"finkey_internet_page_enrich_structured_miss_total {s.page_enrich_structured_miss}",
            f"finkey_internet_page_http_fallback_ok_total {s.page_enrich_http_fallback_ok}",
            f"finkey_internet_stealth_fallback_total {s.stealth_fallback_hits}",
        ]
        for name, val in sorted(s.backend_calls.items()):
            safe = name.replace("-", "_").replace(".", "_")
            lines.append(f'finkey_internet_backend_calls_total{{backend="{safe}"}} {val}')
        for name, val in sorted(s.backend_errors.items()):
            safe = name.replace("-", "_").replace(".", "_")
            lines.append(f'finkey_internet_backend_errors_total{{backend="{safe}"}} {val}')
        return "\n".join(lines) + "\n"
