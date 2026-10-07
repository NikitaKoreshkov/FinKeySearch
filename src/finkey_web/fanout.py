# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Back-compat shim — implementation lives in ``research_orchestrator``.

Prefer: ``from finkey_web.research_orchestrator import orchestrate_search``
"""
from __future__ import annotations

from .research_orchestrator import (
    build_fanout_queries,
    fanout_enabled,
    orchestrate_search,
    orchestrator_enabled,
    parallel_search,
    should_orchestrate,
)

__all__ = [
    "fanout_enabled",
    "build_fanout_queries",
    "parallel_search",
    "orchestrate_search",
    "orchestrator_enabled",
    "should_orchestrate",
]
