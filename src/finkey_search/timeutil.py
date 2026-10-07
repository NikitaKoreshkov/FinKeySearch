# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Local clock helper: the classifier must never judge freshness blind."""
from __future__ import annotations

import os
from datetime import datetime


def current_datetime() -> datetime:
    tz_name = (os.getenv("FINKEY_TZ") or "").strip()
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(tz_name))
        except Exception:  # noqa: BLE001 — bad tz name falls back to system zone
            pass
    return datetime.now().astimezone()
