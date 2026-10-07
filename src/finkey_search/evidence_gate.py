# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Evidence gate — reflection after SERP+enrich: score what we have, plan what's missing.

Domain-agnostic: no topic/role/currency denylists. Places come from locative
prepositions; content nouns are the rest of the question minus function words.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional
from urllib.parse import urlparse

from .schema import SearchDepth, SearchResult

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(
    r"(?:[$€£¥₽₸]\s?\d[\d\s.,]*|\d[\d\s.,]*\s?(?:€|EUR|USD|\$|₽|₸|zł|£|"
    r"%|k\b|тыс|млн|млрд|usd|eur))",
    re.I,
)

# Plain floats for cross-page agreement (temps, rates, scores) — not topic-bound.
_PLAIN_NUM_RE = re.compile(r"(?<![\w./])(-?\d{1,4}(?:[.,]\d{1,2})?)(?![\w./])")

# Structural research shape (verbs/length), not entity allowlists.
_RESEARCH_SHAPE_RE = re.compile(
    r"(проанализируй|проанализировать|анализ\b|собери\b|собрать\b|"
    r"сравни\b|compare\b|research\b|investigate|comprehensive|"
    r"обзор\s+рынка|deep\s*dive|gather\s+all)",
    re.I,
)

# Closed-class grammar only — not topics, roles, currencies, or domains.
_FUNCTION_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "as", "at", "by", "for",
        "from", "into", "of", "on", "to", "with", "vs", "versus", "than",
        "this", "that", "these", "those", "what", "which", "when", "where",
        "how", "why", "who", "whom", "all", "any", "some", "not", "no",
        "и", "в", "во", "на", "по", "для", "из", "к", "ко", "о", "об", "обо",
        "от", "до", "при", "без", "над", "под", "перед", "через", "между",
        "как", "что", "это", "или", "но", "же", "ли", "бы", "не", "ни",
        "уже", "ещё", "еще", "только", "также", "тоже", "очень", "есть",
        "быть", "был", "была", "были", "будет",
        "всем", "всей", "всех", "всему", "всё", "все", "этот", "эта", "эти",
    }
)

_INSTR_PREFIX_RE = re.compile(
    r"^(?:проанализируй|проанализировать|собери|собрать|найди|подбери|сделай|"
    r"investigate|analyze|analyse|research|gather|compare|find|please)\s+",
    re.I,
)

_PREP_PLACE_RE = re.compile(
    r"(?:\bв\b|\bin\b|\bat\b|\bдля\b|\bпо\b|\bof\b|\bрегионе\b|\bрайоне\b|"
    r"\bcity\s+of\b|\bcountry\s+of\b)\s+"
    r"([A-Za-zА-Яа-яЁёÁÉÍÓÚÄÖÜÑáéíóúäöüñ][A-Za-zА-Яа-яЁёÁÉÍÓÚÄÖÜÑáéíóúäöüñ-]{1,})",
    re.I,
)


@dataclass
class EvidenceReport:
    enriched_chars: int = 0
    snippet_chars: int = 0
    numeric_hits: int = 0
    place_token_hits: int = 0
    place_tokens: list[str] = field(default_factory=list)
    unique_hosts: int = 0
    row_count: int = 0
    enriched_rows: int = 0
    thin: bool = False
    gaps: list[str] = field(default_factory=list)
    summary_for_planner: str = ""


@dataclass
class AnswerReadyReport:
    """Early-stop signal during page enrich (ChatGPT-like: stop when enough)."""
    ready: bool = False
    reason: str = ""
    enriched_rows: int = 0
    enriched_chars: int = 0
    unique_hosts: int = 0
    has_numbers: bool = False
    numeric_conflict: bool = False
    place_ok: bool = True
    effective_depth: str = "deep"


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def gap_followup_enabled() -> bool:
    raw = (os.getenv("FINKEY_WEB_GAP_FOLLOWUP", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def extract_place_tokens(message: str, *, max_n: int = 8) -> list[str]:
    """
    Places = tokens after locative prepositions only.
    No Capitalized scooping, no topic/role denylists.
    """
    found: list[str] = []
    seen: set[str] = set()
    for m in _PREP_PLACE_RE.finditer(message or ""):
        t = (m.group(1) or "").strip(" -.,;:")
        key = t.lower()
        if len(key) < 2 or key in _FUNCTION_WORDS or key in seen:
            continue
        # Structural: ALLCAPS ≤4 (EUR) is not a place name
        if t.isupper() and len(t) <= 4:
            continue
        seen.add(key)
        found.append(t)
    return found[:max_n]


def extract_content_nouns(message: str, *, max_n: int = 10) -> list[str]:
    """Question content tokens minus places and function words."""
    raw = _INSTR_PREFIX_RE.sub("", (message or "").strip())
    places = {p.lower() for p in extract_place_tokens(message)}
    out: list[str] = []
    seen: set[str] = set()
    for w in re.findall(r"[A-Za-zА-Яа-яЁё]{4,}", raw):
        key = w.lower()
        if key in _FUNCTION_WORDS or key in places or key in seen:
            continue
        if w.isupper() and len(w) <= 4:
            continue
        seen.add(key)
        out.append(key)
    return out[:max_n]


def assess_evidence(message: str, results: list[SearchResult]) -> EvidenceReport:
    place = extract_place_tokens(message)[:4]
    place_l = [p.lower() for p in place]
    enriched_chars = 0
    snippet_chars = 0
    numeric = 0
    place_hits = 0
    hosts: set[str] = set()
    enriched_rows = 0
    title_bits: list[str] = []

    for r in results or []:
        blob = (r.enriched_text or "").strip()
        snip = (r.snippet or "").strip()
        title = (r.title or "").strip()
        url = (r.url or "").strip()
        if blob:
            enriched_rows += 1
            enriched_chars += len(blob)
            text = f"{title}\n{url}\n{blob}"
        else:
            text = f"{title}\n{url}\n{snip}"
            snippet_chars += len(snip)
        numeric += len(_NUM_RE.findall(text[:12_000]))
        low = text.lower()
        if place_l and any(p in low for p in place_l):
            place_hits += 1
        try:
            h = urlparse(r.url or "").netloc.lower().replace("www.", "")
            if h:
                hosts.add(h)
        except Exception:
            pass
        if title:
            title_bits.append(title[:80])

    report = EvidenceReport(
        enriched_chars=enriched_chars,
        snippet_chars=snippet_chars,
        numeric_hits=numeric,
        place_token_hits=place_hits,
        place_tokens=place,
        unique_hosts=len(hosts),
        row_count=len(results or []),
        enriched_rows=enriched_rows,
        summary_for_planner="; ".join(title_bits[:12]),
    )

    min_enriched = _env_int("FINKEY_WEB_EVIDENCE_MIN_ENRICHED_CHARS", 2500)
    min_numeric = _env_int("FINKEY_WEB_EVIDENCE_MIN_NUMERIC", 3)
    min_hosts = _env_int("FINKEY_WEB_EVIDENCE_MIN_HOSTS", 3)
    min_place_ratio = _env_float("FINKEY_WEB_EVIDENCE_MIN_PLACE_RATIO", 0.35)

    gaps: list[str] = []
    if report.row_count < 4:
        gaps.append("too_few_results")
    if report.enriched_chars < min_enriched and report.snippet_chars < min_enriched * 2:
        gaps.append("thin_page_text")
    if report.numeric_hits < min_numeric:
        gaps.append("missing_numbers")
    if place and report.row_count > 0:
        ratio = report.place_token_hits / max(1, report.row_count)
        if ratio < min_place_ratio:
            gaps.append("weak_place_overlap")
    if report.unique_hosts < min_hosts:
        gaps.append("too_few_hosts")

    hard = {"thin_page_text", "missing_numbers", "weak_place_overlap", "too_few_results"}
    report.gaps = gaps
    report.thin = bool(set(gaps) & hard) or len(gaps) >= 2
    return report


def _page_plain_numbers(text: str, *, limit: int = 24) -> list[float]:
    out: list[float] = []
    for m in _PLAIN_NUM_RE.findall((text or "")[:8_000]):
        try:
            v = float(str(m).replace(",", "."))
        except ValueError:
            continue
        if abs(v) > 1e7:
            continue
        out.append(v)
        if len(out) >= limit:
            break
    return out


def _numbers_close(a: float, b: float) -> bool:
    diff = abs(a - b)
    if diff <= 1.0:
        return True
    scale = max(abs(a), abs(b), 1e-9)
    return (diff / scale) <= 0.08


def _share_number(a: list[float], b: list[float]) -> bool:
    for x in a[:16]:
        for y in b[:16]:
            if _numbers_close(x, y):
                return True
    return False


def numeric_conflict_across_enriched(results: list[SearchResult]) -> tuple[bool, bool]:
    """
    Returns ``(has_numbers, conflict)`` using enriched bodies only.
    Conflict = ≥2 enriched pages with numbers and no shared value within tolerance.
    Domain-agnostic (no topic lists).
    """
    page_nums: list[list[float]] = []
    for r in results or []:
        blob = (r.enriched_text or "").strip()
        if not blob:
            continue
        nums = _page_plain_numbers(blob)
        if nums:
            page_nums.append(nums)
    if not page_nums:
        return False, False
    if len(page_nums) == 1:
        return True, False
    for i, a in enumerate(page_nums):
        for b in page_nums[i + 1 :]:
            if _share_number(a, b):
                return True, False
    return True, True


def _effective_answer_depth(message: str, depth: SearchDepth) -> SearchDepth:
    """
    Classifier depth is primary; long / research-shaped asks escalate to DEEP
    for early-stop thresholds (still no topic allowlists).
    """
    d = depth if isinstance(depth, SearchDepth) else SearchDepth.DEEP
    if d != SearchDepth.FAST:
        return SearchDepth.DEEP
    msg = (message or "").strip()
    if len(msg) >= 160:
        return SearchDepth.DEEP
    if _RESEARCH_SHAPE_RE.search(msg):
        return SearchDepth.DEEP
    return SearchDepth.FAST


def answer_ready(
    message: str,
    results: list[SearchResult],
    *,
    depth: SearchDepth = SearchDepth.DEEP,
) -> AnswerReadyReport:
    """
    Domain-agnostic sufficiency for wave enrich early-stop.

    FAST: stop after 1–2 solid enriched pages when facts agree.
    DEEP: stop only when research evidence gate is not thin and numbers agree.
    """
    eff = _effective_answer_depth(message, depth)
    place = extract_place_tokens(message or "")[:4]
    place_l = [p.lower() for p in place]
    enriched_rows = 0
    enriched_chars = 0
    hosts: set[str] = set()
    place_hits = 0

    for r in results or []:
        blob = (r.enriched_text or "").strip()
        if not blob:
            continue
        enriched_rows += 1
        enriched_chars += len(blob)
        low = f"{r.title or ''}\n{r.url or ''}\n{blob[:4000]}".lower()
        if place_l and any(p in low for p in place_l):
            place_hits += 1
        try:
            h = urlparse(r.url or "").netloc.lower().replace("www.", "")
            if h:
                hosts.add(h)
        except Exception:
            pass

    has_numbers, conflict = numeric_conflict_across_enriched(results)
    place_ok = (not place_l) or place_hits >= 1
    report = AnswerReadyReport(
        ready=False,
        enriched_rows=enriched_rows,
        enriched_chars=enriched_chars,
        unique_hosts=len(hosts),
        has_numbers=has_numbers,
        numeric_conflict=conflict,
        place_ok=place_ok,
        effective_depth=eff.value,
    )

    if enriched_rows <= 0:
        report.reason = "no_enriched"
        return report
    if conflict:
        report.reason = "numeric_conflict"
        return report
    if not place_ok:
        report.reason = "weak_place"
        return report

    if eff == SearchDepth.FAST:
        short = len((message or "").strip()) < 100
        solid = has_numbers or enriched_chars >= 600
        if enriched_rows >= 2 and solid:
            report.ready = True
            report.reason = "fast_multi_agree"
            return report
        if enriched_rows >= 1 and has_numbers and place_ok and (short or enriched_chars >= 400):
            report.ready = True
            report.reason = "fast_single_solid"
            return report
        report.reason = "fast_need_more"
        return report

    # DEEP: reuse research thin-bar (stricter); early-stop only when not thin.
    ev = assess_evidence(message, results)
    if ev.thin:
        report.reason = "deep_thin:" + ",".join(ev.gaps[:4])
        return report
    if enriched_rows < 3:
        report.reason = "deep_need_hosts"
        return report
    report.ready = True
    report.reason = "deep_sufficient"
    return report


def filter_place_relevant(
    message: str,
    results: list[SearchResult],
    *,
    min_keep: int = 4,
) -> tuple[list[SearchResult], str]:
    """
    Prefer rows where place appears in title/url/snippet (not only buried body)
    and content nouns from the question also appear.
    """
    places = extract_place_tokens(message)[:4]
    nouns = extract_content_nouns(message)
    if (not places and not nouns) or not results:
        return results, ""
    place_l = [p.lower() for p in places]
    noun_l = nouns[:8]

    def _primary(r: SearchResult) -> str:
        return f"{r.title or ''}\n{r.url or ''}\n{r.snippet or ''}".lower()

    def _full(r: SearchResult) -> str:
        return _primary(r) + "\n" + ((r.enriched_text or "")[:4000]).lower()

    def _score(r: SearchResult) -> tuple[int, int, float]:
        prim = _primary(r)
        full = _full(r)
        p_hit = 1 if (not place_l or any(p in prim for p in place_l)) else 0
        n_hits = sum(1 for n in noun_l if n in full) if noun_l else 1
        if noun_l and n_hits == 0:
            p_hit = 0
        return (p_hit, n_hits, float(r.relevance or 0))

    ranked = sorted(results, key=_score, reverse=True)
    matched = [r for r in ranked if _score(r)[0] == 1 and _score(r)[1] >= 1]
    note = (
        f"Place/noun filter: places={places} nouns={noun_l[:5]} "
        f"matched={len(matched)}/{len(results)}"
    )
    if matched:
        return matched, note + f" (matched only, kept={len(matched)})"
    return ranked[: max(min_keep, 8)], note + " (weak match — relevance-sorted top)"


_GAP_PLANNER_PROMPT = """You are a research critic writing NEW web search queries
(reflection pass: fill the gaps the first pass left open).

The first pass left gaps. Propose 3–5 concrete SERP queries that will fill them.

Return ONE JSON object only:
{{"queries":["search query 1","search query 2","search query 3"]}}

Hard rules:
- Every query MUST include the place/entity tokens when listed.
- Mix: at least one query in the user's language, at least one precise English query.
- Prefer queries that ask for numbers/ranges/dates when gaps include missing_numbers.
- Use year {year} when the question is about current data.
- Do NOT invent URLs. Search queries only. No browsers.
- Do NOT copy first-pass titles verbatim as queries.
- Today: {today}

USER_QUESTION:
{message}

GAPS: {gaps}
PLACE_TOKENS: {places}
CONTENT_NOUNS: {nouns}
FIRST_PASS_TITLES: {titles}
SNIPPET_SAMPLES: {snippets}
"""


def _parse_gap_queries(raw: str) -> list[str]:
    if not raw:
        return []
    m = re.search(r"\{[\s\S]*\}", raw.strip())
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    qs = data.get("queries")
    if not isinstance(qs, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for q in qs:
        qn = re.sub(r"\s+", " ", str(q or "").strip())
        key = qn.lower()
        if len(qn) < 8 or key in seen:
            continue
        seen.add(key)
        out.append(qn[:240])
    return out[:5]


def heuristic_gap_queries(message: str, report: EvidenceReport) -> list[str]:
    """Follow-ups from gaps + tokens already in the question — no topic lists."""
    topic = _INSTR_PREFIX_RE.sub("", (message or "").strip())
    topic = re.split(r"[:：]", topic, maxsplit=1)[0]
    topic = re.sub(r"\s+", " ", topic).strip(" ,;")[:140]
    year = date.today().year
    place = " ".join(report.place_tokens[:3])
    base = f"{place} {topic}".strip() if place else topic
    out: list[str] = []
    gaps = set(report.gaps)

    if "missing_numbers" in gaps or "thin_page_text" in gaps or not gaps:
        out.append(f"{base} statistics data numbers prices rates {year}")
        out.append(f"{base} {year}")
    if "weak_place_overlap" in gaps and place:
        nouns = extract_content_nouns(message)
        noun_s = " ".join(nouns[:5])
        out.append(f"{place} {noun_s} {year}".strip())
        out.append(f"{place} {noun_s} average median range {year}".strip())
        out.append(f"{place} {topic[:80]}")
    if "too_few_hosts" in gaps or "too_few_results" in gaps:
        out.append(f"{base} report analysis overview {year}")
    if place:
        nouns = extract_content_nouns(message)
        if nouns:
            out.append(f"{place} {' '.join(nouns[:4])} {year}")

    if re.search(r"[А-Яа-яЁё]", message or "") and place:
        out.append(f"{place} data {year}")
    elif place:
        out.append(f"{place} {year} official statistics")

    seen: set[str] = set()
    final: list[str] = []
    for q in out:
        qn = re.sub(r"\s+", " ", q).strip()
        key = qn.lower()
        if len(qn) < 8 or key in seen:
            continue
        seen.add(key)
        final.append(qn[:240])
    return final[:5]


def plan_gap_queries(
    message: str,
    report: EvidenceReport,
    *,
    generate_fn: Optional[Callable[[str], str]] = None,
    force: bool = False,
) -> list[str]:
    if not gap_followup_enabled():
        return []
    if not report.thin and not force:
        return []

    fn = generate_fn
    if fn is None:
        try:
            from .research_orchestrator import _builtin_planner_generate_fn

            fn = _builtin_planner_generate_fn()
        except Exception:
            fn = None

    snippets = ""
    # Titles already on report; keep prompt compact
    if fn:
        nouns = extract_content_nouns(message)
        prompt = _GAP_PLANNER_PROMPT.format(
            today=date.today().isoformat(),
            year=date.today().year,
            message=(message or "")[:2000],
            gaps=", ".join(report.gaps) or "thin_evidence",
            places=", ".join(report.place_tokens) or "(none)",
            nouns=", ".join(nouns[:8]) or "(none)",
            titles=(report.summary_for_planner or "")[:1000],
            snippets=snippets or "(n/a)",
        )
        try:
            raw = fn(prompt)
            qs = _parse_gap_queries(raw)
            if qs:
                logger.info("Gap follow-up LLM queries=%s", qs)
                return qs
        except Exception as exc:
            logger.debug("Gap planner LLM failed: %s", exc)

    qs = heuristic_gap_queries(message, report)
    logger.info("Gap follow-up heuristic queries=%s gaps=%s", qs, report.gaps)
    return qs


__all__ = [
    "EvidenceReport",
    "AnswerReadyReport",
    "assess_evidence",
    "answer_ready",
    "numeric_conflict_across_enriched",
    "extract_place_tokens",
    "extract_content_nouns",
    "filter_place_relevant",
    "gap_followup_enabled",
    "plan_gap_queries",
    "heuristic_gap_queries",
]
