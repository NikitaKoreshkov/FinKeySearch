# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""OpenAI-format tool definitions for native ``web_search`` on the answer stream."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional

from finkey_search.live_query import compact_serp_query

logger = logging.getLogger(__name__)


def _web_search_tool_max_chars() -> int:
    raw = (os.getenv("FINKEY_WEB_SEARCH_TOOL_MAX_CHARS") or "120000").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 120_000
    return max(24_000, min(v, 400_000))


WEB_SEARCH_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Live internet search (real-time). The test for calling it: could the true "
                "answer have changed, or first come into existence, after your training data "
                "ended? If yes — or if you are simply not certain — call this. Judge the "
                "substance of the question, not its wording: it does not need words like "
                "'today', 'current' or 'latest' to be time-sensitive. "
                "Call it especially when the question concerns a date after your training "
                "cutoff but on or before today's date (given in the system prompt): having no "
                "record of such an event means you missed it, NOT that it has yet to happen — "
                "never answer that an already-past date is still in the future. "
                "Also call it when you do not know, do not recognise a name or term, would "
                "guess or refuse, or when the answer hinges on a current number, price, rate, "
                "score, schedule, office-holder, released version or legal rule. "
                "And call it immediately whenever the user asks you to search, verify or check "
                "online. Do NOT claim you lack internet — this tool IS live access. "
                "Skip it for greetings, small talk, pure math/reasoning, rewriting text the "
                "user supplied, and genuinely timeless concepts you already know well. "
                "Hard multi-facet research: first call with depth=fast (optional subqueries), "
                "then depth=deep with urls=[best candidates]. Use mode=verify to check "
                "numeric claims in a draft against evidence. Prefer GitHub API facts in "
                "results for stars/forks — do not sum predecessor repos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in the user's language (or English).",
                    },
                    "require_sources": {
                        "type": "boolean",
                        "description": "True when the user asked for links/sources/citations.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["fast", "deep"],
                        "description": (
                            "fast = SERP draft (snippets+URLs, no page enrich); "
                            "deep = fuller page enrichment (optionally limited by urls)."
                        ),
                    },
                    "subqueries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional parallel sub-queries (max 5) for multi-facet research. "
                            "Skips the planner LLM and fans out immediately."
                        ),
                    },
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional allowlist of https URLs to enrich deeply "
                            "(second-phase deep pass after a fast draft)."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["search", "verify"],
                        "description": (
                            "search = retrieve (default). verify = check draft_answer "
                            "claims against prior evidence / citation_urls."
                        ),
                    },
                    "draft_answer": {
                        "type": "string",
                        "description": "Required for mode=verify: the draft text to fact-check.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "Optional evidence blob for mode=verify "
                            "(defaults to empty — pass prior synthesized text when available)."
                        ),
                    },
                    "citation_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional citation URL allowlist for mode=verify.",
                    },
                    "focus_claims": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subset of claim strings to verify.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

WEB_SEARCH_TOOL_NAMES: frozenset[str] = frozenset(
    t["function"]["name"] for t in WEB_SEARCH_TOOLS
)


def _as_str_list(raw: Any, *, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        s = str(item or "").strip()
        if s:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _run_verify_mode(args: dict[str, Any]) -> str:
    from finkey_search.claim_verify import verify_claims_against_evidence

    draft = (args.get("draft_answer") or args.get("query") or "").strip()
    if not draft:
        return json.dumps(
            {"ok": False, "phase": "verify", "error": "draft_answer_required"},
            ensure_ascii=False,
        )
    evidence = (args.get("evidence") or "").strip()
    citations = _as_str_list(args.get("citation_urls"), limit=24)
    focus = _as_str_list(args.get("focus_claims"), limit=24)
    require_sources = bool(args.get("require_sources"))
    report = verify_claims_against_evidence(
        draft,
        evidence,
        citation_urls=citations,
        require_citations=require_sources or bool(citations),
        focus_claims=focus or None,
    )
    payload = report.to_dict()
    payload["phase"] = "verify"
    payload["synthesized"] = (
        "[CLAIM VERIFY]\n"
        + report.note
        + (
            "\nUnsupported: " + ", ".join(c.raw for c in report.unsupported)
            if report.unsupported
            else ""
        )
        + (
            "\nConflicts: " + ", ".join(c.raw for c in report.conflicts)
            if report.conflicts
            else ""
        )
        + (
            "\nGap queries: " + " | ".join(report.gap_queries)
            if report.gap_queries
            else ""
        )
    )
    payload["citation_urls"] = citations
    payload["query_used"] = "verify"
    payload["useful"] = True
    payload["error"] = None if report.ok else "claims_need_repair"
    return json.dumps(payload, ensure_ascii=False)


def run_web_search_tool(
    name: str,
    arguments: Optional[dict[str, Any]],
    *,
    engine: Any,
    generate_fn: Optional[Callable[[str], str]] = None,
    conversation_key: str = "",
    cache_scope: str = "",
    turn_number: int = 1,
    user_goal: str = "",
) -> str:
    """Execute ``web_search`` via ``WebSearchEngine.run``; return JSON for the tool message."""
    if name != "web_search":
        return json.dumps({"ok": False, "error": f"unknown_tool:{name}"}, ensure_ascii=False)

    args = arguments if isinstance(arguments, dict) else {}
    mode = str(args.get("mode") or "search").strip().lower()
    if mode == "verify":
        return _run_verify_mode(args)

    query = (args.get("query") or "").strip()
    if not query:
        return json.dumps({"ok": False, "error": "query_required"}, ensure_ascii=False)

    require_sources = bool(args.get("require_sources"))
    depth = str(args.get("depth") or "fast").strip().lower()
    serp_draft = depth != "deep" and not require_sources
    require_full = depth == "deep" or require_sources
    subqueries = _as_str_list(args.get("subqueries"), limit=5)
    urls = _as_str_list(args.get("urls"), limit=12)

    from finkey_search.schema import SearchDepth

    search_depth = SearchDepth.DEEP if require_full else SearchDepth.FAST
    # ChatGPT-like: orchestrator / deep-research / place-lock see the full user goal;
    # SERP always uses a compact query (never the full research essay).
    goal = (user_goal or "").strip() or query
    serp_q = compact_serp_query(query, goal=goal)
    compact_subs = [
        compact_serp_query(sq, goal=goal) for sq in subqueries if (sq or "").strip()
    ]
    compact_subs = [s for s in compact_subs if s][:5]

    try:
        ctx = engine.run(
            message=goal,
            turn_number=max(1, int(turn_number or 1)),
            generate_fn=generate_fn,
            allow_llm_tier=False,  # model already decided to search
            conversation_messages=None,
            conversation_key=conversation_key or None,
            require_full_articles=require_full and not serp_draft,
            force_search=True,
            cache_scope=cache_scope or "",
            serp_draft=serp_draft,
            enrich_urls=urls or None,
            explicit_subqueries=compact_subs or None,
            search_depth=search_depth,
            search_query_override=serp_q or None,
        )
    except Exception as exc:
        logger.exception("web_search tool failed")
        return json.dumps(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]},
            ensure_ascii=False,
        )

    useful = bool(getattr(ctx, "is_useful", lambda: False)())
    synthesized = (getattr(ctx, "synthesized", None) or "").strip()
    cap = _web_search_tool_max_chars()
    if len(synthesized) > cap:
        synthesized = synthesized[:cap] + "\n…"

    urls_out = list(getattr(ctx, "citation_urls", None) or [])[:24]
    # Prefer result URLs for draft follow-up even if citation list is short.
    candidate_urls = list(urls_out)
    for r in list(getattr(ctx, "results", None) or [])[:16]:
        u = (getattr(r, "url", None) or "").strip()
        if u.startswith("https://") and u not in candidate_urls:
            candidate_urls.append(u)
        if len(candidate_urls) >= 16:
            break

    cat = getattr(ctx, "category", None)
    cat_val = getattr(cat, "value", cat) if cat is not None else None
    phase = "draft" if serp_draft else "deep"

    return json.dumps(
        {
            "ok": useful,
            "phase": phase,
            "query_used": getattr(ctx, "query_used", query) or query,
            "useful": useful,
            "verified_sources_required": bool(
                getattr(ctx, "verified_sources_required", False) or require_sources
            ),
            "citation_urls": urls_out,
            "candidate_urls": candidate_urls,
            "category": cat_val,
            "search_date": getattr(ctx, "search_date", "") or "",
            "total_results": int(getattr(ctx, "total_results", 0) or 0),
            "synthesized": synthesized,
            "error": None if useful else "no_useful_results",
        },
        ensure_ascii=False,
    )
