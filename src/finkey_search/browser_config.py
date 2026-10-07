# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Per-page fetch budget handed to a structured page fetcher.

The package ships the shape, not the engine: ``finkey_search._browser.stealth_browser``
may provide ``fetch_structured_sync(url, cfg)``, and any host-supplied callable that
accepts this config works too. With no fetcher installed, enrichment stays HTTP-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_NAV_ALLOWED = frozenset({"commit", "domcontentloaded", "load", "networkidle"})


@dataclass(frozen=True)
class BrowserConfig:
    headless: bool = True
    timeout_ms: int = 25_000
    wait_after_load_ms: int = 1_500
    max_retries: int = 2
    take_screenshot: bool = False
    scroll_after_load: bool = True
    block_media: bool = True
    navigation_wait_until: str = "load"
    identity_key: Optional[str] = None
    language_hint: Optional[str] = None
    user_data_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.navigation_wait_until not in _NAV_ALLOWED:
            raise ValueError(
                f"navigation_wait_until must be one of {sorted(_NAV_ALLOWED)}"
            )
