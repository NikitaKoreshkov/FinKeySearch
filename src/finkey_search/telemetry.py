# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Optional OpenTelemetry wiring — no hard dependency.

If ``opentelemetry-api`` is installed and env ``OTEL_METRICS_EXPORTER`` or
``FINKEY_OTEL_METRICS=1`` is set, counters are updated alongside ``InternetMetrics``.
Otherwise this module is a no-op.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from finkey_search.metrics import InternetMetrics

logger = logging.getLogger(__name__)

_otel_ready: Optional[bool] = None
_ctr_searches = _ctr_skips = _ctr_errors = None


def _otel_enabled() -> bool:
    return os.environ.get("FINKEY_OTEL_METRICS", "").strip().lower() in ("1", "true", "yes", "on")


def _lazy_init():
    global _otel_ready, _ctr_searches, _ctr_skips, _ctr_errors
    if _otel_ready is not None:
        return
    _otel_ready = False
    if not _otel_enabled():
        return
    try:
        from opentelemetry import metrics as otel_metrics

        meter = otel_metrics.get_meter("finkey.internet", "1.0.0")
        _ctr_searches = meter.create_counter(
            "finkey_internet_search_started",
            description="Internet search pipeline started",
        )
        _ctr_skips = meter.create_counter(
            "finkey_internet_classifier_skips",
            description="Classifier decided not to search",
        )
        _ctr_errors = meter.create_counter(
            "finkey_internet_backend_errors",
            description="Search backend failures",
        )
        _otel_ready = True
    except Exception as exc:
        logger.debug("OpenTelemetry metrics unavailable: %s", exc)
        _otel_ready = False


def otel_note_search_start() -> None:
    _lazy_init()
    if _otel_ready and _ctr_searches:
        _ctr_searches.add(1)


def otel_note_classifier_skip() -> None:
    _lazy_init()
    if _otel_ready and _ctr_skips:
        _ctr_skips.add(1)


def otel_note_backend_error() -> None:
    _lazy_init()
    if _otel_ready and _ctr_errors:
        _ctr_errors.add(1)


def push_internet_snapshot_to_otel(metrics: "InternetMetrics") -> None:
    """Optional hook: flush dict as log line for scrapers without Prometheus."""
    if not _otel_enabled():
        return
    try:
        logger.info("internet_metrics %s", metrics.as_dict())
    except Exception:
        pass
