# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Research Orchestrator — a domain-agnostic deep-research fan-out.

Pipeline:
  Coordinator → should we parallelize?
  Planner     → LLM JSON preferred; else GENERIC research lenses (not topic allowlists)
  Agents      → parallel SERP
  Merge       → dedupe + provenance for the answer model

No bitcoin/iran/gold hardcodes. No trusted-domain ranking.
Never spawns parallel browsers.

Env:
  FINKEY_WEB_FANOUT=1
  FINKEY_WEB_ORCH_MAX_AGENTS=5
  FINKEY_WEB_ORCH_LLM_PLAN=1
  FINKEY_WEB_ORCH_BUILTIN_LLM=1   # Cerebras/OpenRouter planner if generate_fn missing
"""
from __future__ import annotations

import json
import logging
import os
import re
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

from .schema import SearchDecision, SearchDepth, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class SubAgentPlan:
    agent_id: str
    role: str
    query: str


@dataclass
class SubAgentResult:
    plan: SubAgentPlan
    results: list[SearchResult] = field(default_factory=list)
    error: str = ""
    elapsed_ms: float = 0.0


def orchestrator_enabled() -> bool:
    raw = (os.getenv("FINKEY_WEB_FANOUT", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _max_agents() -> int:
    try:
        n = int(os.getenv("FINKEY_WEB_ORCH_MAX_AGENTS", "5"))
    except ValueError:
        n = 5
    return max(2, min(n, 8))


def _llm_plan_enabled() -> bool:
    raw = (os.getenv("FINKEY_WEB_ORCH_LLM_PLAN", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _builtin_llm_enabled() -> bool:
    raw = (os.getenv("FINKEY_WEB_ORCH_BUILTIN_LLM", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


# Structural research intent — verbs/shapes, NOT entity names
_RESEARCH_INTENT_RE = re.compile(
    r"(проанализируй|проанализировать|анализ\b|собери\b|собрать\b|"
    r"найди\s+лучш|лучшее\s+жиль|цена\s*[-/]?\s*качество|price\s*[-/]?\s*quality|"
    r"обзор\s+рынка|весь\s+рынок|рынок\s+\w+|market\s+analysis|deep\s*dive|"
    r"research\b|gather\s+all|all\s+data|все\s+данн|"
    r"по\s+всему\s+региону|во\s+вс[её]м\s+регионе|neighborhoods|районы|"
    r"сравни\s+.+\s+и\s+|compare\s+.+\s+(and|vs)|"
    r"рейтинг|ранжир|подбери|рекоменд|investigate|comprehensive)",
    re.I,
)

_SIMPLE_FACT_RE = re.compile(
    r"(сколько\s+(сейчас\s+)?(стоит|курс)|какая\s+сейчас\s+(цена|погода)|"
    r"кто\s+сейчас\s+(ceo|директор|президент\s+\w+)|"
    r"what('s| is)\s+the\s+(price|rate|weather)|price\s+of\s+\w+\s*$)",
    re.I,
)

_COMPARE_RE = re.compile(r"\b(сравни|compare|vs\.?|versus|против)\b", re.I)


def is_deep_research(message: str) -> bool:
    """Open-ended multi-aspect research (housing market, industry scan, etc.)."""
    msg = (message or "").strip()
    if len(msg) < 40:
        return False
    if _RESEARCH_INTENT_RE.search(msg):
        return True
    # Long multi-clause ask with several imperatives
    if len(msg) >= 160 and msg.count(",") + msg.count(".") + msg.count("?") >= 2:
        if re.search(r"(анализ|данные|рынок|регион|лучш|market|data|best)", msg, re.I):
            return True
    return False


def should_orchestrate(message: str, decision: Optional[SearchDecision] = None) -> bool:
    """
    Parallel search agents for deep/multi-aspect questions only.
    Domain-agnostic: no entity allowlists.
    """
    if not orchestrator_enabled():
        return False
    msg = (message or "").strip()
    if not msg:
        return False

    if _SIMPLE_FACT_RE.search(msg) and not _COMPARE_RE.search(msg) and not is_deep_research(msg):
        return False

    if is_deep_research(msg):
        return True
    if _COMPARE_RE.search(msg) and len(msg) >= 40:
        return True

    if decision is not None:
        depth = getattr(decision, "search_depth", SearchDepth.DEEP)
        if depth == SearchDepth.DEEP and len(msg) >= 120:
            return True
        if getattr(decision, "verified_sources_required", False) and len(msg) >= 80:
            return True
    return False


def _topic_core(message: str, primary: str) -> str:
    """Strip instruction verbs; keep the subject of research (place/market), not the whole ask."""
    raw = (primary or message or "").strip()
    raw = re.sub(
        r"^(проанализируй|проанализировать|собери|собрать|найди|подбери|сделай|"
        r"investigate|analyze|research|gather|compare|find)\s+",
        "",
        raw,
        flags=re.I,
    )
    # Drop trailing instruction clauses after colon / "и на основе" / "структурируй"
    raw = re.split(r"[:：]", raw, maxsplit=1)[0]
    raw = re.split(
        r",?\s*(и\s+на\s+основе|и\s+сделай|структурируй|только\s+по\s+источникам|"
        r"предложи|with\s+justification|based\s+on\s+sources)\b",
        raw,
        maxsplit=1,
        flags=re.I,
    )[0]
    raw = re.sub(r"\s+", " ", raw).strip(" ,;")
    # Prefer a compact subject (location + market) under ~120 chars
    if len(raw) > 120:
        raw = raw[:120].rsplit(" ", 1)[0]
    return (raw or (message or "")[:120]).strip()


_PLANNER_PROMPT = """You are the research coordinator.
Decompose the user question into independent web-search sub-agents.

Return ONE JSON object only:
{{"parallel":true|false,"agents":[{{"id":"a1","role":"short_role","query":"concrete SERP query"}}]}}

Rules:
- parallel=false + 1 agent for a single fact (one price, weather, one CEO name).
- parallel=true with 3–{max_n} agents for market analysis, "gather all data", compare options,
  find best price/quality, regional scans, multi-part research.
- Cover DISTINCT lenses adapted to THIS question's nouns (salaries ≠ housing ≠ coworking ≠ SaaS).
  Examples of role shapes (pick what fits): overview, prices_data, best_value, segments,
  context_risks, comparisons — do NOT force "neighborhoods/rent" unless the user asked about housing/areas.
- Every query MUST include the place/entity from the question when present.
- Mix languages: at least one query in the user's language AND one precise English query when useful.
- Prefer concrete SERP phrasing with year when data is time-sensitive. Today: {today}.
- Do NOT invent URLs. Search queries only — no browsers.
- Roles: short English slugs from the task — NOT a fixed list of tickers or countries.

USER_QUESTION:
{message}

PRIMARY_HINT:
{primary}
"""


def _parse_planner_json(raw: str) -> Optional[list[SubAgentPlan]]:
    if not raw:
        return None
    m = re.search(r"\{[\s\S]*\}", raw.strip())
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    agents = data.get("agents")
    if not isinstance(agents, list) or not agents:
        return None
    out: list[SubAgentPlan] = []
    for i, a in enumerate(agents[: _max_agents()]):
        if not isinstance(a, dict):
            continue
        q = str(a.get("query") or "").strip()
        if len(q) < 6:
            continue
        rid = str(a.get("id") or f"a{i+1}").strip()[:16]
        role = str(a.get("role") or f"agent_{i+1}").strip()[:48]
        out.append(SubAgentPlan(agent_id=rid, role=role, query=q[:240]))
    return out or None


def plan_with_llm(
    message: str,
    primary: str,
    generate_fn: Callable[[str], str],
) -> Optional[list[SubAgentPlan]]:
    prompt = _PLANNER_PROMPT.format(
        max_n=_max_agents(),
        today=date.today().isoformat(),
        message=(message or "")[:2500],
        primary=(primary or "")[:500],
    )
    try:
        raw = generate_fn(prompt)
        plans = _parse_planner_json(raw)
        if plans:
            logger.info("Orch LLM planner agents=%d roles=%s", len(plans), [p.role for p in plans])
            return plans
    except Exception as exc:
        logger.debug("Orch LLM planner failed: %s", exc)
    return None


_HAS_PRICE_RE = re.compile(
    r"(цена|цен[ыу]|стоимост|price|cost|rent|аренд|зарплат|salary|wage|ставк|"
    r"rate|fee|тариф|сколько\s+стоит)",
    re.I,
)
_HAS_BEST_RE = re.compile(
    r"(лучш|best|рейтинг|rank|quality|цена\s*[-/]?\s*качеств|price\s*[-/]?\s*quality|"
    r"рекоменд|подбери|top\b)",
    re.I,
)
_HAS_AREA_RE = re.compile(
    r"(район|neighborhood|district|регион|area|квартал|микрорайон|borough|zone)",
    re.I,
)
_HAS_SEGMENT_RE = re.compile(
    r"(категор|segment|тип\w*|kinds?|types?|уровн|junior|middle|senior|desk|office)",
    re.I,
)

# Structural intent for primary-source SERP lenses (no geo/brand/topic allowlists).
# Avoid bare «рынок» — salary/housing asks also say «рынок …».
_HAS_MARKET_IR_RE = re.compile(
    r"(market\s*size|TAM|revenue|выручк|доля\s*рынка|market\s*share|"
    r"earnings|10[\s-]?K|10[\s-]?Q|filing|investor\s*relations|"
    r"press\s*release|финансов\w*\s*отч[её]т|annual\s*report|"
    r"(?:объ[её]м|volume)\s*(?:рынка|market)|regulatory\s*filing)",
    re.I,
)
_HAS_GOV_STATS_RE = re.compile(
    r"(официальн|official\s+(?:stat|data|figure|report|source)|"
    r"national\s*stats?|government\s+(?:stat|data|report)|"
    r"\b\.gov\b|stat\.gov|министерств|ministry\s+of|"
    r"бюр[оа]\s*(?:нац|стат)|statistics\s+office|national\s+bureau)",
    re.I,
)
_HAS_OSS_GH_RE = re.compile(
    r"(github|open[\s-]?source|stars?|зв[её]зд|репозитор|"
    r"multi[\s-]?agent|orchestration)",
    re.I,
)


def _primary_source_lenses(message: str, topic: str, year: int) -> list[tuple[str, str]]:
    """
    Primary-source SERP lenses — structural shapes only.
    No country TLDs, no topic brands, no probe-tuned nouns.
    Topic text already carries place/entity from the user question.
    """
    msg = message or ""
    primary: list[tuple[str, str]] = []
    if _HAS_MARKET_IR_RE.search(msg):
        primary.append(
            (
                "primary_filings",
                f"{topic} investor relations OR annual report OR earnings OR regulatory filing {year}",
            )
        )
        primary.append(
            (
                "primary_ir",
                f"{topic} press release financial results OR 10-K OR 10-Q {year}",
            )
        )
    if _HAS_GOV_STATS_RE.search(msg):
        primary.append(
            (
                "primary_gov",
                f"{topic} official statistics OR government report OR national statistics office {year}",
            )
        )
    if _HAS_OSS_GH_RE.search(msg):
        primary.append(
            (
                "primary_gh",
                f"{topic} site:github.com stars license",
            )
        )
    return primary


def _adaptive_lenses(message: str, topic: str, year: int) -> list[tuple[str, str]]:
    """
    Shape SERP lenses from question grammar/nouns — never force housing paths
    (neighborhoods/rent) onto salary/coworking/SaaS questions.

    Primary-source lenses (IR/SEC, gov, GitHub) are ordered early so they survive
    ``_max_agents`` truncation — ChatGPT browse density depends on them.
    """
    msg = message or ""
    core: list[tuple[str, str]] = [
        ("overview", f"{topic} overview statistics trends {year}"),
        ("data", f"{topic} data figures numbers {year}"),
    ]
    primary = _primary_source_lenses(msg, topic, year)

    soft: list[tuple[str, str]] = []
    if _HAS_PRICE_RE.search(msg):
        price_bits = " ".join(
            dict.fromkeys(
                m.group(0).lower()
                for m in re.finditer(
                    r"(зарплат\w*|salary|salaries|wage\w*|rent|аренд\w*|цен\w*|"
                    r"price\w*|cost\w*|ставк\w*|fee\w*|тариф\w*)",
                    msg,
                    re.I,
                )
            )
        ) or "prices costs rates"
        soft.append(("prices_data", f"{topic} {price_bits} median average {year}"))
    elif not primary:
        # Skip generic costs lens when primary filings/gov/gh already fill the budget.
        soft.append(("prices_data", f"{topic} costs rates fees {year}"))

    if _HAS_BEST_RE.search(msg):
        soft.append(("best_value", f"{topic} best options ranking comparison"))
    if _HAS_AREA_RE.search(msg):
        soft.append(("breakdown", f"{topic} neighborhoods districts areas comparison"))
    elif _HAS_SEGMENT_RE.search(msg):
        soft.append(("breakdown", f"{topic} categories levels types segments comparison"))
    elif not primary:
        soft.append(("breakdown", f"{topic} categories types segments comparison"))

    soft.append(("context_risks", f"{topic} news outlook risks regulations {year}"))

    # Bilingual dual: keep one query close to the user's phrasing (no English lens spam)
    if re.search(r"[А-Яа-яЁё]", msg):
        soft.append(("locale", f"{topic} {year}"))
    return core + primary + soft


def plan_heuristic(message: str, primary: str) -> list[SubAgentPlan]:
    """
    Domain-agnostic research lenses applied to the extracted topic.
    Adaptive to question shape (housing vs salary vs SaaS) — no fixed rent/neighborhood spam.
    """
    topic = _topic_core(message, primary)
    year = date.today().year
    lenses = _adaptive_lenses(message, topic, year)

    # Comparisons: keep two halves if "vs/и" splits cleanly — but still inject
    # primary-source lenses so GitHub/gov/IR density is not lost.
    if _COMPARE_RE.search(message or ""):
        parts = re.split(
            r"\s+(?:и|vs\.?|versus|против|and)\s+",
            message or "",
            maxsplit=1,
            flags=re.I,
        )
        if len(parts) == 2 and min(len(parts[0]), len(parts[1])) > 25:
            lenses = [
                ("aspect_a", parts[0].strip()[:160]),
                ("aspect_b", parts[1].strip()[:160]),
                ("overview", f"{topic} comparison overview {year}"),
                ("best_value", f"{topic} which better value {year}"),
            ] + _primary_source_lenses(message or "", topic, year)

    plans: list[SubAgentPlan] = []
    seen: set[str] = set()
    for role, q in lenses:
        qn = re.sub(r"\s+", " ", q).strip()
        key = f"{role}:{qn.lower()[-80:]}"
        if len(qn) < 8 or key in seen:
            continue
        seen.add(key)
        plans.append(SubAgentPlan(agent_id=f"a{len(plans)+1}", role=role, query=qn[:240]))
        if len(plans) >= _max_agents():
            break
    if len(plans) < 2:
        topic_short = topic[:80]
        for role, suffix in (
            ("overview", f"overview statistics {year}"),
            ("prices_data", f"data numbers {year}"),
            ("best_value", "best options comparison"),
            ("breakdown", "categories segments"),
            ("context_risks", f"news risks {year}"),
        ):
            qn = f"{topic_short} {suffix}".strip()
            plans.append(SubAgentPlan(agent_id=f"a{len(plans)+1}", role=role, query=qn[:240]))
            if len(plans) >= _max_agents():
                break
    if not plans:
        plans = [SubAgentPlan(agent_id="a1", role="primary", query=(primary or topic)[:240])]
    return plans[: _max_agents()]


def _ssl_ctx():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _builtin_planner_generate_fn() -> Optional[Callable[[str], str]]:
    """Cerebras first, else OpenRouter — so deep research always has a planner."""
    if not _builtin_llm_enabled():
        return None

    cerebras = (os.getenv("CEREBRAS_API_KEY") or "").strip()
    if cerebras:
        model = (os.getenv("CEREBRAS_MODEL") or "gpt-oss-120b").strip()
        base = (os.getenv("CEREBRAS_BASE_URL") or "https://api.cerebras.ai/v1").rstrip("/")

        def _cerebras(prompt: str) -> str:
            body = json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 900,
                    **({"reasoning_effort": "low"} if model.startswith("gpt-oss") else {}),
                }
            ).encode()
            req = urllib.request.Request(
                base + "/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {cerebras}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=45, context=_ssl_ctx()) as resp:
                data = json.loads(resp.read().decode())
            return (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""

        return _cerebras

    or_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if or_key:
        model = (
            os.getenv("FINKEY_WEB_ORCH_PLANNER_MODEL")
            or os.getenv("FINKEY_ROUTER_CHECKS_OPENROUTER_MODEL")
            or "qwen/qwen3.5-9b"
        ).strip()
        base = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")

        def _or(prompt: str) -> str:
            body = json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 900,
                }
            ).encode()
            req = urllib.request.Request(
                base + "/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {or_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                    "X-Title": "FinKeyWeb-orch-planner",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60, context=_ssl_ctx()) as resp:
                data = json.loads(resp.read().decode())
            return (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""

        return _or
    return None


def build_plans(
    message: str,
    primary: str,
    *,
    generate_fn: Optional[Callable[[str], str]] = None,
) -> list[SubAgentPlan]:
    fn = generate_fn
    if fn is None and _llm_plan_enabled():
        fn = _builtin_planner_generate_fn()
    if fn and _llm_plan_enabled():
        llm_plans = plan_with_llm(message, primary, fn)
        if llm_plans and len(llm_plans) >= 2:
            return llm_plans
        if llm_plans and len(llm_plans) == 1 and not should_orchestrate(message):
            return llm_plans
    return plan_heuristic(message, primary)


def run_search_agents(
    plans: list[SubAgentPlan],
    search_fn: Callable[..., list],
    *,
    max_results: int,
    language: str,
    date_restrict: str,
) -> list[SubAgentResult]:
    import time

    if not plans:
        return []
    if len(plans) == 1:
        t0 = time.perf_counter()
        try:
            rows = search_fn(
                plans[0].query,
                max_results=max_results,
                language=language,
                date_restrict=date_restrict,
            ) or []
            return [
                SubAgentResult(
                    plan=plans[0],
                    results=list(rows),
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )
            ]
        except Exception as exc:
            return [SubAgentResult(plan=plans[0], error=str(exc)[:300])]

    per = max(4, max_results // len(plans) + 3)
    workers = max(1, min(len(plans), _max_agents()))

    def _one(plan: SubAgentPlan) -> SubAgentResult:
        t0 = time.perf_counter()
        try:
            rows = search_fn(
                plan.query,
                max_results=per,
                language=language,
                date_restrict=date_restrict,
            ) or []
            return SubAgentResult(
                plan=plan,
                results=list(rows),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            return SubAgentResult(
                plan=plan,
                error=str(exc)[:300],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    out: list[SubAgentResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, p): p for p in plans}
        for fut in as_completed(futs):
            out.append(fut.result())

    by_id = {r.plan.agent_id: r for r in out}
    ordered = [by_id[p.agent_id] for p in plans if p.agent_id in by_id]
    seen_ids = {r.plan.agent_id for r in ordered}
    for r in out:
        if r.plan.agent_id not in seen_ids:
            ordered.append(r)
            seen_ids.add(r.plan.agent_id)
    return ordered


def merge_agent_results(
    agent_results: list[SubAgentResult],
    *,
    stamp_agents: bool = True,
) -> tuple[list[SearchResult], str]:
    merged: list[SearchResult] = []
    seen_urls: set[str] = set()
    lines = ["Parallel search agents:"] if stamp_agents else ["Single search agent:"]
    for ar in agent_results:
        n = len(ar.results)
        err = f" err={ar.error}" if ar.error else ""
        lines.append(
            f"  • [{ar.plan.agent_id}/{ar.plan.role}] q={ar.plan.query[:90]!r} "
            f"hits={n} {ar.elapsed_ms:.0f}ms{err}"
        )
        tag = f"[agent:{ar.plan.role}] " if stamp_agents else ""
        for r in ar.results:
            url = (r.url or "").strip()
            key = url.split("#")[0].rstrip("/").lower() if url else ""
            if key and key in seen_urls:
                continue
            if key:
                seen_urls.add(key)
            snippet = r.snippet or ""
            if tag and tag not in snippet[:48]:
                snippet = tag + snippet
            # Light relevance bump for first hits per agent (ordering only)
            rel = float(r.relevance or 0.5)
            if stamp_agents and n:
                # keep agent diversity in later rerank — no domain trust
                pass
            merged.append(
                SearchResult(
                    title=r.title,
                    url=r.url,
                    snippet=snippet,
                    date=r.date,
                    source=r.source,
                    relevance=rel,
                    enriched_text=r.enriched_text,
                )
            )
    return merged, "\n".join(lines)


def plans_from_explicit_subqueries(
    primary_query: str,
    subqueries: list[str],
) -> list[SubAgentPlan]:
    """Build agent plans from caller-supplied subqueries (skip planner LLM)."""
    cleaned: list[str] = []
    seen: set[str] = set()
    primary = (primary_query or "").strip()
    for raw in list(subqueries or []):
        q = (raw or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(q[:500])
        if len(cleaned) >= _max_agents():
            break
    if primary:
        pkey = primary.lower()
        if pkey not in seen and len(cleaned) < _max_agents():
            cleaned.insert(0, primary[:500])
        elif pkey not in seen and cleaned:
            cleaned[0] = primary[:500]
    if not cleaned and primary:
        cleaned = [primary[:500]]
    return [
        SubAgentPlan(agent_id=f"a{i+1}", role=f"sub{i+1}", query=q)
        for i, q in enumerate(cleaned)
    ]


def orchestrate_search(
    message: str,
    primary_query: str,
    search_fn: Callable[..., list],
    *,
    decision: Optional[SearchDecision] = None,
    generate_fn: Optional[Callable[[str], str]] = None,
    max_results: int = 24,
    language: str = "",
    date_restrict: str = "",
    explicit_subqueries: Optional[list[str]] = None,
) -> tuple[list[SearchResult], str, list[SubAgentPlan]]:
    explicit = [q for q in (explicit_subqueries or []) if (q or "").strip()]
    if explicit:
        plans = plans_from_explicit_subqueries(primary_query, explicit)
        logger.info(
            "Research orch (explicit subqueries): %d agents roles=%s",
            len(plans),
            [p.role for p in plans],
        )
        agent_results = run_search_agents(
            plans, search_fn, max_results=max_results, language=language, date_restrict=date_restrict
        )
        merged, note = merge_agent_results(agent_results, stamp_agents=len(plans) > 1)
        header = (
            f"Coordinator: explicit_subqueries={len(plans)} "
            f"parallel={len(plans) > 1} deep_research={is_deep_research(message)}\n"
        )
        return merged, header + note, plans

    if not should_orchestrate(message, decision):
        plans = [SubAgentPlan(agent_id="a1", role="primary", query=primary_query)]
        agent_results = run_search_agents(
            plans, search_fn, max_results=max_results, language=language, date_restrict=date_restrict
        )
        merged, note = merge_agent_results(agent_results, stamp_agents=False)
        return merged, "Single-query path (no parallel agents).\n" + note, plans

    plans = build_plans(message, primary_query, generate_fn=generate_fn)
    if len(plans) < 2:
        plans = plan_heuristic(message, primary_query)

    logger.info("Research orch: %d agents roles=%s", len(plans), [p.role for p in plans])
    agent_results = run_search_agents(
        plans, search_fn, max_results=max_results, language=language, date_restrict=date_restrict
    )
    merged, note = merge_agent_results(agent_results, stamp_agents=len(plans) > 1)
    header = f"Coordinator: parallel={len(plans) > 1} agents={len(plans)} deep_research={is_deep_research(message)}\n"
    return merged, header + note, plans


def fanout_enabled() -> bool:
    return orchestrator_enabled()


def build_fanout_queries(message: str, primary: str) -> list[str]:
    return [p.query for p in plan_heuristic(message, primary)]


def parallel_search(
    search_fn: Callable[..., list],
    queries: list[str],
    *,
    max_results: int,
    language: str,
    date_restrict: str,
    max_workers: int = 3,
) -> list:
    plans = [SubAgentPlan(agent_id=f"a{i+1}", role=f"q{i+1}", query=q) for i, q in enumerate(queries)]
    agent_results = run_search_agents(
        plans, search_fn, max_results=max_results, language=language, date_restrict=date_restrict
    )
    merged, _ = merge_agent_results(agent_results)
    return merged


__all__ = [
    "SubAgentPlan",
    "SubAgentResult",
    "should_orchestrate",
    "is_deep_research",
    "build_plans",
    "plan_heuristic",
    "plans_from_explicit_subqueries",
    "orchestrate_search",
    "orchestrator_enabled",
    "fanout_enabled",
    "build_fanout_queries",
    "parallel_search",
]
