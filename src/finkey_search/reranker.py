# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Re-order search hits before synthesis — heuristic + optional LLM gate."""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from finkey_search.schema import SearchResult
from finkey_search.synthesizer import _score_result_uncapped as heuristic_score

logger = logging.getLogger(__name__)


_LLM_RERANK_PROMPT = '''You pick the best web snippets for answering the user query.
Reply ONLY JSON: {{"keep": [<0-based indices>]}}
Keep at most {max_keep} indices, ordered best-first. Prefer factual, dated, on-topic snippets.

Query: "{query}"

Snippets:
{numbered}
'''


class SearchReranker:
    def __init__(self, llm_min_pool: int = 6) -> None:
        """If raw rows >= llm_min_pool and LLM enabled, consider LLM trimming."""
        self.llm_min_pool = llm_min_pool

    def rerank(
        self,
        results: list[SearchResult],
        query: str,
        *,
        max_keep: int,
        generate_fn: Optional[Callable[[str], str]] = None,
        use_llm: bool = False,
    ) -> list[SearchResult]:
        if not results:
            return []

        scored = sorted(
            results,
            key=lambda r: heuristic_score(r, query),
            reverse=True,
        )

        if (
            use_llm
            and generate_fn is not None
            and len(scored) >= self.llm_min_pool
        ):
            trimmed = self._llm_pick_indices(scored, query, max_keep, generate_fn)
            if trimmed:
                return trimmed

        return scored[:max_keep]

    def _llm_pick_indices(
        self,
        ordered: list[SearchResult],
        query: str,
        max_keep: int,
        generate_fn: Callable[[str], str],
    ) -> list[SearchResult]:
        lines = []
        for i, r in enumerate(ordered[:24]):
            blob = f"{r.title[:120]} | {(r.snippet or '')[:280]}"
            lines.append(f"{i}. {blob}")
        numbered = "\n".join(lines)
        prompt = _LLM_RERANK_PROMPT.format(
            query=query.replace('"', "'")[:400],
            numbered=numbered,
            max_keep=max_keep,
        )
        try:
            raw = generate_fn(prompt).strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return []
            data = json.loads(m.group())
            idxs = data.get("keep") or []
            if not isinstance(idxs, list):
                return []
            out: list[SearchResult] = []
            seen: set[int] = set()
            for x in idxs:
                try:
                    j = int(x)
                except (TypeError, ValueError):
                    continue
                if j < 0 or j >= len(ordered) or j in seen:
                    continue
                seen.add(j)
                out.append(ordered[j])
                if len(out) >= max_keep:
                    break
            return out if out else []
        except Exception as exc:
            logger.debug("LLM rerank skipped: %s", exc)
            return []
