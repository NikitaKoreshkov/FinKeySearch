# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Chain multiple search backends with per-backend guards."""
from __future__ import annotations

import contextvars
import logging
import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Optional

from finkey_search.schema import SearchResult

from .protocol import SearchBackend

logger = logging.getLogger(__name__)


def _hedge_delay_s() -> float:
    """
    Через сколько секунд запускать запасные бэкенды параллельно, если
    приоритетный ещё не ответил. 0 — выключить хеджирование (строгая
    последовательная цепочка, как раньше).
    """
    raw = (os.getenv("FINKEY_SEARCH_HEDGE_MS", "2000") or "").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 2000
    return max(0, min(v, 15_000)) / 1000.0


@dataclass
class GuardedBackend:
    inner: SearchBackend
    allow: Callable[[], bool]
    """Return False to skip this backend (rate limit / circuit open)."""

    on_call: Optional[Callable[[str], None]] = None
    on_success: Optional[Callable[[str, int], None]] = None
    on_failure: Optional[Callable[[str, Exception], None]] = None


class FallbackSearchChain:
    """
    Try backends in order until one returns non-empty results.
    Empty list from a backend does **not** count as hard failure — next backend runs.

    Hedged mode (default, ``FINKEY_SEARCH_HEDGE_MS``>0): приоритетный бэкенд
    стартует сразу; если он не ответил за грейс-период (429-ретраи, сеть),
    остальные запускаются параллельно, и берётся лучший по приоритету среди
    непустых. Последовательная деградация «Serper висит 6с → потом Brave ещё
    3с» превращается в гонку с общим временем ~max, а не sum.
    """

    def __init__(self, layers: list[GuardedBackend]) -> None:
        self._layers = layers

    @property
    def name(self) -> str:
        return "fallback_chain"

    def search(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        date_restrict: str,
    ) -> list[SearchResult]:
        allowed = [l for l in self._layers if l.allow()]
        for skipped in (l for l in self._layers if l not in allowed):
            logger.debug("Search backend skipped by guard: %s", skipped.inner.name)
        if not allowed:
            return []

        hedge = _hedge_delay_s()
        if hedge <= 0 or len(allowed) == 1:
            return self._search_serial(
                allowed, query,
                max_results=max_results, language=language, date_restrict=date_restrict,
            )
        return self._search_hedged(
            allowed, query, hedge,
            max_results=max_results, language=language, date_restrict=date_restrict,
        )

    def _call_layer(
        self,
        layer: GuardedBackend,
        query: str,
        *,
        max_results: int,
        language: str,
        date_restrict: str,
    ) -> list[SearchResult]:
        if layer.on_call:
            layer.on_call(layer.inner.name)
        try:
            rows = layer.inner.search(
                query,
                max_results=max_results,
                language=language,
                date_restrict=date_restrict,
            )
        except Exception as exc:
            logger.warning("Search backend error [%s]: %s", layer.inner.name, exc)
            if layer.on_failure:
                layer.on_failure(layer.inner.name, exc)
            return []
        if rows:
            if layer.on_success:
                layer.on_success(layer.inner.name, len(rows))
            logger.info("Search backend hit: %s (%d rows)", layer.inner.name, len(rows))
        return rows or []

    def _search_serial(
        self,
        layers: list[GuardedBackend],
        query: str,
        *,
        max_results: int,
        language: str,
        date_restrict: str,
    ) -> list[SearchResult]:
        for layer in layers:
            rows = self._call_layer(
                layer, query,
                max_results=max_results, language=language, date_restrict=date_restrict,
            )
            if rows:
                return rows
        return []

    def _search_hedged(
        self,
        layers: list[GuardedBackend],
        query: str,
        hedge_s: float,
        *,
        max_results: int,
        language: str,
        date_restrict: str,
    ) -> list[SearchResult]:
        results_by_priority: dict[int, list[SearchResult]] = {}
        pool = ThreadPoolExecutor(
            max_workers=len(layers), thread_name_prefix="serp-hedge"
        )

        def _submit(i: int) -> Future:
            ctx = contextvars.copy_context()
            return pool.submit(
                ctx.run,
                self._call_layer,
                layers[i],
                query,
                max_results=max_results,
                language=language,
                date_restrict=date_restrict,
            )

        futures: dict[Future, int] = {_submit(0): 0}
        hedged = False

        try:
            while futures:
                done, _pending = wait(
                    futures, timeout=None if hedged else hedge_s,
                    return_when=FIRST_COMPLETED,
                )

                if not done and not hedged:
                    # Приоритетный не успел за грейс — запускаем запасные параллельно.
                    hedged = True
                    for i in range(1, len(layers)):
                        futures[_submit(i)] = i
                    logger.info(
                        "Search hedge fired after %.1fs: launched %d backup backend(s)",
                        hedge_s, len(layers) - 1,
                    )
                    continue

                for fut in done:
                    prio = futures.pop(fut)
                    try:
                        rows = fut.result()
                    except Exception:
                        rows = []
                    if rows:
                        results_by_priority[prio] = rows

                # Топ-приоритет ответил непустым — берём сразу.
                if 0 in results_by_priority:
                    break
                # Приоритетный завершился пусто: если есть любой готовый непустой — берём
                # лучший из готовых; иначе ждём остальных (или хеджируем немедленно).
                if all(p != 0 for p in futures.values()) and results_by_priority:
                    break
                if not futures and not hedged and len(layers) > 1:
                    hedged = True
                    for i in range(1, len(layers)):
                        futures[_submit(i)] = i
        finally:
            # Не ждём отставшие бэкенды: победитель уже есть, потоки доработают в фоне.
            pool.shutdown(wait=False, cancel_futures=True)

        if not results_by_priority:
            return []
        best = min(results_by_priority)
        return results_by_priority[best]
