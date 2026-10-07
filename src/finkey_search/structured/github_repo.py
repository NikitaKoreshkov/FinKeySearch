# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
GitHub REST API → SearchResult with typed stars/forks/license.

Prefer this over blog roundups that glue AutoGen + MAF star counts together.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

from finkey_search.schema import SearchResult

logger = logging.getLogger(__name__)

_GH_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)",
    re.I,
)
_REPO_COLON_RE = re.compile(
    r"\brepo:(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)\b",
    re.I,
)
# Common bare mentions in research queries
_BARE_PAIR_RE = re.compile(
    r"\b(?P<owner>langchain-ai|crewaiinc|microsoft|openai|agentscope-ai|"
    r"huggingface|anthropics)/(?P<repo>[A-Za-z0-9_.-]+)\b",
    re.I,
)

# Product-name → canonical repo (so "LangGraph stars" hits the API, not blog roundups).
def _name_rx(name: str) -> re.Pattern[str]:
    """Match product name as a token — not as a prefix of ``langchain-ai`` etc."""
    return re.compile(rf"(?<![A-Za-z0-9_-]){name}(?![A-Za-z0-9_-])", re.I)


_NAME_ALIASES: tuple[tuple[re.Pattern[str], tuple[str, str]], ...] = (
    (_name_rx("langgraph"), ("langchain-ai", "langgraph")),
    # Must not match inside owner ``langchain-ai`` when asking about LangGraph.
    (_name_rx("langchain"), ("langchain-ai", "langchain")),
    (_name_rx("crewai"), ("crewAIInc", "crewAI")),
    (_name_rx("autogen"), ("microsoft", "autogen")),
    (
        re.compile(r"\b(?:microsoft\s+)?agent\s+framework\b|\bmaf\b", re.I),
        ("microsoft", "agent-framework"),
    ),
    (_name_rx("agentscope"), ("agentscope-ai", "agentscope")),
    (
        re.compile(r"\bopenai\s+agents?\s+sdk\b|\bopenai-agents\b", re.I),
        ("openai", "openai-agents-python"),
    ),
    (_name_rx("smolagents"), ("huggingface", "smolagents")),
    (re.compile(r"\bopenhands\b|\bopendevin\b", re.I), ("All-Hands-AI", "OpenHands")),
    (re.compile(r"\bletta\b|\bmemgpt\b", re.I), ("letta-ai", "letta")),
    (re.compile(r"\bbrowser[\s-]?use\b", re.I), ("browser-use", "browser-use")),
    (_name_rx("pydantic-ai"), ("pydantic", "pydantic-ai")),
    (re.compile(r"\bmastra\b", re.I), ("mastra-ai", "mastra")),
    # Common frameworks (global product→repo when named — not probe allowlists).
    (_name_rx("vue"), ("vuejs", "vue")),
    (_name_rx("django"), ("django", "django")),
    (_name_rx("fastapi"), ("fastapi", "fastapi")),
    (_name_rx("nextjs"), ("vercel", "next.js")),
    (re.compile(r"\bnext\.js\b", re.I), ("vercel", "next.js")),
)


def github_structured_enabled() -> bool:
    raw = (os.getenv("FINKEY_WEB_STRUCTURED_GITHUB", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _token() -> str:
    return (
        (os.getenv("FINKEY_GITHUB_TOKEN") or "").strip()
        or (os.getenv("GITHUB_TOKEN") or "").strip()
    )


def parse_github_repos(
    *texts: str,
    max_repos: int = 8,
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    blob = "\n".join(t for t in texts if t)

    def _add(owner: str, repo: str) -> bool:
        owner = (owner or "").strip()
        repo = (repo or "").strip().rstrip(".git")
        if not owner or not repo:
            return False
        if repo.lower() in ("issues", "pulls", "actions", "wiki", "tree", "blob", "releases"):
            return False
        key = f"{owner.lower()}/{repo.lower()}"
        if key in seen:
            return False
        seen.add(key)
        found.append((owner, repo))
        return len(found) >= max_repos

    for rx in (_GH_URL_RE, _REPO_COLON_RE, _BARE_PAIR_RE):
        for m in rx.finditer(blob):
            if _add(m.group("owner"), m.group("repo")):
                return found

    # Name aliases only when the text looks like OSS / agent-stack research.
    if re.search(
        r"\b(github|open[\s-]?source|stars?|framework|sdk|multi[\s-]?agent|"
        r"orchestration|репозитор|зв[её]зд)\b",
        blob,
        re.I,
    ):
        for rx, (owner, repo) in _NAME_ALIASES:
            if rx.search(blob) and _add(owner, repo):
                return found
    return found


def _ssl_ctx():
    """Same certifi path as research_orchestrator — bare urlopen fails on macOS Python."""
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_github_repo_facts(
    owner: str,
    repo: str,
    *,
    timeout_s: float = 6.0,
) -> Optional[SearchResult]:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "FinKeySearch/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=_ssl_ctx()) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.info("GitHub structured miss %s/%s: %s", owner, repo, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.info("GitHub structured error %s/%s: %s", owner, repo, exc)
        return None

    if not isinstance(data, dict) or data.get("message") == "Not Found":
        return None

    stars = int(data.get("stargazers_count") or 0)
    forks = int(data.get("forks_count") or 0)
    open_issues = int(data.get("open_issues_count") or 0)
    license_obj = data.get("license") or {}
    license_spdx = ""
    if isinstance(license_obj, dict):
        license_spdx = str(license_obj.get("spdx_id") or license_obj.get("name") or "")
    html_url = str(data.get("html_url") or f"https://github.com/{owner}/{repo}")
    pushed = str(data.get("pushed_at") or "")
    desc = str(data.get("description") or "")
    full_name = str(data.get("full_name") or f"{owner}/{repo}")
    archived = bool(data.get("archived"))

    enriched = (
        f"[STRUCTURED:github_api] repo={full_name}\n"
        f"stars={stars}\n"
        f"forks={forks}\n"
        f"open_issues={open_issues}\n"
        f"license={license_spdx or 'unknown'}\n"
        f"pushed_at={pushed}\n"
        f"archived={archived}\n"
        f"html_url={html_url}\n"
        f"description={desc[:400]}\n"
        f"NOTE: Use these star/fork counts — do not invent or sum predecessor repos."
    )
    snippet = (
        f"GitHub API: {full_name} — {stars:,} stars, {forks:,} forks, "
        f"license {license_spdx or 'n/a'}, pushed {pushed[:10] or 'n/a'}."
    )
    return SearchResult(
        title=f"{full_name} (GitHub API)",
        url=html_url,
        snippet=snippet,
        date=pushed[:10] if pushed else None,
        source="github_api",
        relevance=1.15,
        enriched_text=enriched,
    )


def merge_github_structured_results(
    results: list[SearchResult],
    *,
    query: str = "",
    message: str = "",
    max_repos: int = 8,
) -> list[SearchResult]:
    """Parse repos from query/message/result URLs/titles and prepend API facts."""
    if not github_structured_enabled():
        return results

    # Titles/snippets often carry github.com/owner/repo even when URL is a blog.
    serp_blob = "\n".join(
        f"{(r.url or '')}\n{(r.title or '')}\n{(r.snippet or '')}"
        for r in (results or [])[:40]
    )
    repos = parse_github_repos(query, message, serp_blob, max_repos=max_repos)
    if not repos:
        return results

    structured: list[SearchResult] = []
    for owner, repo in repos:
        row = fetch_github_repo_facts(owner, repo)
        if row is not None:
            structured.append(row)

    if not structured:
        return results

    # Dedupe by normalized github html url — keep API row first.
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in structured + list(results or []):
        key = (r.url or "").split("#")[0].rstrip("/").lower()
        try:
            p = urlparse(key)
            if "github.com" in (p.netloc or ""):
                parts = [x for x in (p.path or "").split("/") if x]
                if len(parts) >= 2:
                    key = f"github.com/{parts[0].lower()}/{parts[1].lower()}"
        except Exception:
            pass
        if key and key in seen:
            # Prefer API-enriched row already in out
            continue
        if key:
            seen.add(key)
        out.append(r)
    logger.info("GitHub structured: injected %d repo fact row(s)", len(structured))
    return out


def append_structured_github_to_evidence(
    evidence: str,
    *extra_texts: str,
    max_repos: int = 8,
) -> str:
    """
    Ensure STRUCTURED github rows exist for any repos named in evidence/answer.
    Used before claim-verify repair so multi-repo star tables can be grounded
    without topic allowlists — only repos the model/SERP already named.
    """
    if not github_structured_enabled():
        return evidence or ""
    blob = "\n".join(t for t in (evidence, *extra_texts) if t)
    existing = set()
    for m in re.finditer(
        r"\[STRUCTURED:github_api\]\s*repo=([^\s\n]+)",
        evidence or "",
        re.I,
    ):
        existing.add(m.group(1).strip().lower())
    repos = parse_github_repos(blob, max_repos=max_repos)
    additions: list[str] = []
    for owner, repo in repos:
        key = f"{owner}/{repo}".lower()
        if key in existing:
            continue
        row = fetch_github_repo_facts(owner, repo)
        if row is None:
            continue
        additions.append((row.enriched_text or row.snippet or "").strip())
        existing.add(key)
    if not additions:
        return evidence or ""
    block = "\n\n".join(additions)
    base = (evidence or "").rstrip()
    return (base + "\n\n" + block).strip() + "\n"
