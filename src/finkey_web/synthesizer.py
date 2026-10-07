# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
ResultSynthesizer — turns raw search results into clean knowledge blocks.

Raw search results are messy — duplicate info, irrelevant sentences,
marketing copy, navigation fragments.

The synthesizer:
  1. Scores results by relevance to the original query
  2. Extracts the most factual sentences
  3. Groups by type (price/rate vs news vs general)
  4. Produces a compact, dated "knowledge block" for the system prompt
  5. Marks each fact with its source (domain name)

Domain-agnostic: works for any kind of real-time data query.
The AI uses this naturally — no "I searched Google", just knows.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

from .schema import SearchCategory, SearchContext, SearchResult
from .url_quality import serp_url_quality_adjustment




def _synth_relevance_floor() -> float:
    raw = (os.getenv("FINKEY_SYNTH_RELEVANCE_FLOOR") or "0.06").strip()
    try:
        v = float(raw)
    except ValueError:
        v = 0.06
    return max(0.02, min(v, 0.45))


def _synth_max_rows() -> int:
    raw = (os.getenv("FINKEY_SYNTH_MAX_SOURCES_PER_BLOCK") or "28").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 28
    return max(6, min(v, 48))


def _synth_snippet_preview_chars() -> int:
    raw = (os.getenv("FINKEY_SYNTH_SNIPPET_CHARS") or "24000").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 24_000
    return max(800, min(v, 120_000))


def _synth_enriched_display_floor() -> int:
    """Minimum excerpt target for enriched rows (``_display_snippet`` lower bound)."""
    raw = (os.getenv("FINKEY_SYNTH_ENRICHED_DISPLAY_FLOOR") or "24000").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 24_000
    return max(400, min(v, 200_000))


def _synth_min_rows_when_all_low() -> int:
    raw = (os.getenv("FINKEY_SYNTH_FALLBACK_MIN_ROWS") or "14").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 14
    return max(4, min(v, _synth_max_rows()))



_NOISE_PHRASES = [
    "cookie", "cookies", "privacy policy", "terms of service",
    "subscribe", "подписаться", "реклама", "advertisement",
    "войти", "login", "sign in", "register", "зарегистрироваться",
    "©", "all rights reserved",
]

def _enriched_snippet_cap() -> int:
    raw = os.getenv("FINKEY_SYNTH_ENRICHED_SNIPPET_CHARS") or os.getenv(
        "FINKEY_WEB_PAGE_ENRICH_MAX_CHARS",
        "500000",
    )
    try:
        v = int(raw)
    except ValueError:
        v = 500_000
    return max(4_000, min(v, 500_000))


def _display_snippet(r: SearchResult, limit: int) -> str:
    blob = (r.enriched_text or r.snippet or "").strip()
    if (r.enriched_text or "").strip():
        cap = _enriched_snippet_cap()
        lim = max(limit, min(cap, len(blob)))
        return blob[:lim]
    return blob[:limit]


def _citation_suffix(r: SearchResult) -> str:
    u = (r.url or "").strip()
    if u.startswith("https://"):
        return f"\n    URL: {u}"
    if u.startswith("http://"):
        return f"\n    URL: {u}"
    return ""


def _score_result_uncapped(
    result: SearchResult,
    query: str,
    visual_media_kind: Optional[str] = None,
) -> float:
    """Ranking signal — may exceed 1.0 so ties break deterministically."""
    text_blob = (result.enriched_text or result.snippet or "")
    score = float(result.relevance)
    lower_snippet = text_blob.lower()
    lower_query = query.lower()

    query_words = [w for w in lower_query.split() if len(w) > 3]
    hits = sum(1 for w in query_words if w in lower_snippet)
    score += min(hits * 0.1, 0.4)

    if re.search(r"\d[\d\.,]+", text_blob):
        score += 0.15

    if result.date:
        score += 0.1

    if any(noise in lower_snippet for noise in _NOISE_PHRASES):
        score -= 0.3

    # No trusted-domain boosts. Credibility is the answer model's job given full context.
    u = (result.url or "").strip()
    if u:
        score += serp_url_quality_adjustment(u, media_kind=visual_media_kind)

    # Structured API rows (GitHub stars etc.) outrank blog roundups.
    if (result.source or "") == "github_api" or (result.enriched_text or "").startswith(
        "[STRUCTURED:github_api]"
    ):
        score += 1.5

    return max(0.0, score)


def _score_result(
    result: SearchResult,
    query: str,
    visual_media_kind: Optional[str] = None,
) -> float:
    """0–1 relevance stored on the dataclass (display / filtering)."""
    return max(
        0.0,
        min(_score_result_uncapped(result, query, visual_media_kind), 1.0),
    )



def _format_currency(results: list[SearchResult], query: str) -> str:
    cap = _synth_max_rows()
    floor_e = _synth_enriched_display_floor()
    snip_n = _synth_snippet_preview_chars()
    lines = ["💱 Курсы валют:"]
    for r in results[:cap]:
        snippet = _display_snippet(r, floor_e if r.enriched_text else snip_n)
        date_str = f" ({r.source}" + (f", {r.date}" if r.date else "") + ")"
        lines.append(f"  • {snippet}{date_str}{_citation_suffix(r)}")
    return "\n".join(lines)


def _format_price(results: list[SearchResult], query: str, icon: str = "📈") -> str:
    cap = _synth_max_rows()
    floor_e = _synth_enriched_display_floor()
    snip_n = _synth_snippet_preview_chars()
    lines = [f"{icon} Цены/котировки:"]
    for r in results[:cap]:
        snippet = _display_snippet(r, floor_e if r.enriched_text else snip_n)
        date_str = f" ({r.source}" + (f", {r.date}" if r.date else "") + ")"
        lines.append(f"  • {snippet}{date_str}{_citation_suffix(r)}")
    return "\n".join(lines)


def _format_rate(results: list[SearchResult], query: str) -> str:
    cap = _synth_max_rows()
    floor_e = _synth_enriched_display_floor()
    snip_n = _synth_snippet_preview_chars()
    lines = ["🏦 Процентные ставки:"]
    for r in results[:cap]:
        snippet = _display_snippet(r, floor_e if r.enriched_text else snip_n)
        date_str = f" ({r.source}" + (f", {r.date}" if r.date else "") + ")"
        lines.append(f"  • {snippet}{date_str}{_citation_suffix(r)}")
    return "\n".join(lines)


def _format_news(results: list[SearchResult], query: str) -> str:
    cap = _synth_max_rows()
    floor_e = _synth_enriched_display_floor()
    snip_n = _synth_snippet_preview_chars()
    lines = ["📰 Актуальные события:"]
    for r in results[:cap]:
        title = r.title[:160]
        snippet = _display_snippet(r, floor_e if r.enriched_text else snip_n)
        date_str = f" ({r.source}" + (f", {r.date}" if r.date else "") + ")"
        lines.append(f"  • **{title}**{date_str}")
        lines.append(f"    {snippet}{_citation_suffix(r)}")
    return "\n".join(lines)


def _format_general(results: list[SearchResult], query: str) -> str:
    cap = _synth_max_rows()
    floor_e = _synth_enriched_display_floor()
    snip_n = _synth_snippet_preview_chars()
    lines = ["🔍 Актуальная информация:"]
    for r in results[:cap]:
        snippet = _display_snippet(r, floor_e if r.enriched_text else snip_n)
        date_str = f" ({r.source}" + (f", {r.date}" if r.date else "") + ")"
        lines.append(f"  • {snippet}{date_str}{_citation_suffix(r)}")
    return "\n".join(lines)


_FORMATTERS = {
    SearchCategory.CURRENCY_RATE:   _format_currency,
    SearchCategory.CRYPTO_PRICE:    lambda r, q: _format_price(r, q, "🪙"),
    SearchCategory.STOCK_PRICE:     lambda r, q: _format_price(r, q, "📈"),
    SearchCategory.INTEREST_RATE:   _format_rate,
    SearchCategory.FINANCIAL_NEWS:  _format_news,
    SearchCategory.ECONOMIC_DATA:   _format_general,
    SearchCategory.COMPANY_INFO:    _format_news,
    SearchCategory.REGULATORY:      _format_general,
    SearchCategory.PRODUCT_RATES:   _format_rate,
    SearchCategory.GENERAL_FINANCE: _format_general,
    SearchCategory.GENERAL:         _format_general,
}



class ResultSynthesizer:
    """
    Converts raw search results into a clean knowledge block for the prompt.
    """

    def synthesize(
        self,
        results:   list[SearchResult],
        query:     str,
        category:  Optional[SearchCategory] = None,
        verification_note: Optional[str] = None,
        visual_media_kind: Optional[str] = None,
    ) -> SearchContext:

        today = datetime.now().strftime("%d %B %Y")

        if not results:
            return SearchContext(
                found=False,
                query_used=query,
                category=category,
                synthesized="",
                search_date=today,
            )

        scored = sorted(
            results,
            key=lambda r: _score_result_uncapped(r, query, visual_media_kind),
            reverse=True,
        )

        for r in scored:
            r.relevance = _score_result(r, query, visual_media_kind)

        floor = _synth_relevance_floor()
        good = [r for r in scored if r.relevance >= floor]
        if not good:
            take = min(_synth_min_rows_when_all_low(), len(scored))
            good = scored[:take] if take else scored[:3]

        try:
            cite_cap = max(16, min(int(os.getenv("FINKEY_SYNTH_MAX_CITATION_URLS", "48")), 96))
        except ValueError:
            cite_cap = 48
        citation_urls = list(
            dict.fromkeys(
                (r.url or "").strip()
                for r in good
                if (r.url or "").strip().startswith(("https://", "http://"))
            )
        )[:cite_cap]

        cat_eff = category or SearchCategory.GENERAL
        formatter = _FORMATTERS.get(cat_eff, _format_general)
        content = formatter(good, query)

        sources = list(dict.fromkeys(r.source for r in good if r.source))
        try:
            src_cap = max(8, min(int(os.getenv("FINKEY_SYNTH_SOURCES_LINE_MAX", "24")), 40))
        except ValueError:
            src_cap = 24
        sources_line = " | ".join(sources[:src_cap])

        verify = ""
        if verification_note:
            verify = f"\n⚖️ Source cross-check:\n{verification_note.strip()}\n"

        structured = ""
        try:
            struct_rows = [
                r
                for r in good
                if (r.source or "") == "github_api"
                or (r.enriched_text or "").startswith("[STRUCTURED:github_api]")
            ]
            if struct_rows:
                slines = [
                    "🔒 STRUCTURED FACTS (authoritative — prefer over blog roundups):",
                ]
                for r in struct_rows[:8]:
                    blob = (r.enriched_text or r.snippet or "").strip()
                    slines.append(f"  • {blob[:900]}")
                slines.append(
                    "  For GitHub stars/forks/license: use ONLY these STRUCTURED rows. "
                    "Never sum predecessor repos (e.g. AutoGen + Agent Framework) into one star count."
                )
                structured = "\n".join(slines) + "\n"
        except Exception:
            structured = ""

        primary_block = ""
        try:
            from .url_quality import is_primary_source_url

            prim_rows = [
                r
                for r in good
                if is_primary_source_url(getattr(r, "url", "") or "")
                or (r.source or "") == "github_api"
                or (r.enriched_text or "").startswith("[STRUCTURED:github_api]")
            ]
            if prim_rows:
                plines = [
                    "🏛️ PRIMARY SOURCES (prefer numbers/dates/names from these over SEO blogs):",
                ]
                for r in prim_rows[:10]:
                    url = (getattr(r, "url", "") or "").strip()
                    title = (getattr(r, "title", "") or "").strip()[:120]
                    snip = (r.snippet or r.enriched_text or "")[:280].replace("\n", " ")
                    plines.append(f"  • {title} | {url}\n    {snip}")
                plines.append(
                    "  If a blog disagrees with a PRIMARY row on a hard figure, follow PRIMARY."
                )
                primary_block = "\n".join(plines) + "\n"
        except Exception:
            primary_block = ""

        grounded = ""
        try:
            from .evidence_gate import extract_place_tokens
            from .numeric_critic import extract_figures

            places = extract_place_tokens(query)
            # Prefer figures from non-social hosts when scoring context.
            figs = extract_figures(query, good)
            trusted = [f for f in figs if f.place_ok][:16]
            # Never fall back to off-place junk when a place was asked
            if not trusted and not places:
                trusted = figs[:12]
            # Drop facebook/twitter-sourced grounded rows when better domains exist
            junk_dom = {"facebook.com", "m.facebook.com", "twitter.com", "x.com", "t.me"}
            non_junk = [f for f in trusted if f.domain not in junk_dom]
            if non_junk:
                trusted = non_junk[:16]
            if trusted:
                glines = [
                    "📌 GROUNDED FIGURES (precise claims MUST use only these rows):",
                ]
                for f in trusted:
                    glines.append(
                        f"  • {f.raw} ({f.domain}) — {f.context[:110]}"
                    )
                glines.append(
                    "  Use a figure ONLY if its context names the asked place/entity. "
                    "If context says Manhattan/Flatiron while the ask is Brooklyn — skip it. "
                    "If a neighborhood/product/price is not listed — say missing; do NOT invent. "
                    "Prefer .gov / official / primary filings over social/blog mirrors."
                )
                grounded = "\n".join(glines) + "\n"
            elif places:
                grounded = (
                    "📌 GROUNDED FIGURES: none with local place-lock for "
                    f"{places}. Do not invent prices; say what is missing.\n"
                )
        except Exception:
            grounded = ""

        # When grounded sheet exists, keep excerpts shorter so the model anchors on it.
        content_out = content
        if (grounded or structured or primary_block) and len(content) > 24_000:
            content_out = content[:24_000] + "\n…[excerpts truncated; use STRUCTURED/PRIMARY/GROUNDED FIGURES + Sources]…\n"

        # Demote social hosts in the Sources line for display clarity.
        sources_clean = [
            s
            for s in sources
            if s
            and s
            not in {
                "facebook.com",
                "m.facebook.com",
                "twitter.com",
                "x.com",
                "t.me",
                "reddit.com",
            }
        ] or sources
        sources_line = " | ".join(sources_clean[:src_cap])

        block = (
            f"\n\n{'═'*52}\n"
            f"LIVE WEB DATA  (search: \"{query}\")\n"
            f"📅 Snapshot date: {today}\n"
            f"{'─'*52}\n"
            f"{structured}"
            f"{primary_block}"
            f"{grounded}"
            f"{content_out}\n"
            f"{verify}"
            f"{'─'*52}\n"
            f"Sources (domains): {sources_line}\n"
            f"{'═'*52}\n\n"
            f"HARD RULES FOR USING THIS BLOCK:\n"
            f"• Prefer primary / official / STRUCTURED facts over social posts and SEO blogs. "
            f"Skip facebook/twitter/telegram as evidence for hard numbers unless nothing else exists.\n"
            f"• Write a clean structured deliverable: short sections, no filler, no repeated words/phrases. "
            f"If you catch yourself looping the same word — stop and rewrite that sentence once.\n"
            f"• You judge source quality yourself from the excerpts (primary docs, dated articles, thin SEO pages).\n"
            f"• Geographic / entity lock: if the user asks about place/entity X, use numbers only from excerpts that "
            f"clearly refer to X. Do not transplant figures from another city, country, company, job title, or product. "
            f"A national average is not a city figure unless the excerpt says so.\n"
            f"• Outlier hygiene: if one excerpt shows an extreme number (e.g. 10× the cluster of other sources), "
            f"treat it as suspect — report the cluster range first and label the outlier separately with its domain.\n"
            f"• Cite ONLY domains/URLs that appear in this block (Sources line or `URL:` lines). "
            f"Never invent Bloomberg/TradingView/etc. if they are not listed above.\n"
            f"• Every concrete number you state must appear verbatim (or as an obvious rounding) in the excerpts; "
            f"do not import figures from memory. Neighborhood/product names likewise — only if named above.\n"
            f"• Current-fact asks (weather now, live FX, spot price): prefer excerpts marked current/today/live; "
            f"ignore climate normals / long-run averages unless the user asked for climate.\n"
            f"• If the user names a specific authority, brand, or jurisdiction, use **only** excerpt lines that clearly "
            f"belong to that entity; do not transplant figures from another regulator/country/company. "
            f"If numbers disagree across lines — report both with domains.\n"
            f"• State facts and figures **only** when they clearly follow from the text above; "
            f"attach the source domain in parentheses for each number (e.g., 450 ₸ (example.com)).\n"
            f"• If sources contradict each other — say so and give both values with domains.\n"
            f"• Do not blend unrelated excerpts into one narrative without a labeled source for each fact "
            f"(names, dates, figures).\n"
            f"• Do not make specific claims about tournament schedules, fighters, team rosters, etc., "
            f"unless they appear in the excerpts — generalize cautiously or say \"not in sources\".\n"
            f"• If data are insufficient for a precise answer — say exactly what is missing; do not invent.\n"
            f"• Avoid classroom phrases like \"I searched the web\" — just answer from substance.\n"
            f"• If this block is non-empty and relevant — **do not** refuse with \"training ends in 2026\", "
            f"\"no info for that date\", \"that date is in my future\", or \"no internet/real-time access\": "
            f"search already ran; ground answers in the text above and domains. Refusal is allowed only when excerpts clearly "
            f"do not cover the question.\n"
        )

        return SearchContext(
            found=True,
            query_used=query,
            category=category,
            citation_urls=citation_urls,
            results=good,
            synthesized=block,
            search_date=today,
            total_results=len(results),
        )
