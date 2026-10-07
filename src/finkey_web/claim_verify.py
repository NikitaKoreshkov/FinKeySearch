# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Claim-level fact check: draft answer numbers vs web_search evidence.

Used by ``web_search`` mode=verify and by the native tool-loop repair path.
Domain-agnostic — no topic allowlists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from .citation_verify import answer_references_allowed_web_sources

_NUM_CLAIM_RE = re.compile(
    r"(?P<cur>[$€£¥₽₸])\s?(?P<n>\d[\d\s.,]{0,16})"
    r"(?:\s?(?P<scale>billion|million|млрд|млн|тыс|[BbMm]|k|K))?"
    r"|(?P<n2>\d[\d\s.,]{1,16})\s?(?P<suf>%|°[CF]?|ГВт|GW|МВт|MW|млрд|млн|тыс|"
    r"billion|million|stars?|зв[её]зд\w*)",
    re.I,
)

_STARS_RE = re.compile(
    r"(?P<n>\d[\d\s.,]{0,12})\s*(?:[~≈~]?\s*)?(?:k|тыс)?\s*"
    r"(?:github\s+)?(?:stars?|зв[её]зд\w*)",
    re.I,
)

_STRUCTURED_GITHUB_BLOCK_RE = re.compile(
    r"\[STRUCTURED:github_api\]\s*repo=(?P<repo>[^\s\n]+)"
    r"(?P<body>.*?)(?=\[STRUCTURED:github_api\]|\Z)",
    re.I | re.S,
)
_STRUCTURED_STARS_LINE_RE = re.compile(r"(?m)^stars=(?P<n>\d+)\s*$")


def evidence_has_structured_github(evidence: str) -> bool:
    return "[STRUCTURED:github_api]" in (evidence or "")


def extract_structured_github_stars(evidence: str) -> dict[str, int]:
    """
    Parse authoritative star counts from STRUCTURED github_api rows.
    These override blog roundups when verifying star claims.
    """
    out: dict[str, int] = {}
    for m in _STRUCTURED_GITHUB_BLOCK_RE.finditer(evidence or ""):
        repo = (m.group("repo") or "").strip().lower()
        body = m.group("body") or ""
        sm = _STRUCTURED_STARS_LINE_RE.search(body)
        if not repo or not sm:
            continue
        try:
            out[repo] = int(sm.group("n"))
        except ValueError:
            continue
    # Also accept inline ``stars=N`` after a STRUCTURED header on one line.
    if not out:
        for m in re.finditer(
            r"\[STRUCTURED:github_api\][^\n]*repo=([^\s\n]+)[^\n]*stars=(\d+)",
            evidence or "",
            re.I,
        ):
            out[m.group(1).strip().lower()] = int(m.group(2))
    return out


def _star_claims_match_structured(
    claim_value: float,
    structured: dict[str, int],
    *,
    rel: float = 0.02,
) -> tuple[bool, str]:
    """True when claim is within 2% of any STRUCTURED star count."""
    if not structured:
        return False, ""
    for repo, stars in structured.items():
        if _approx_equal(claim_value, float(stars), rel=rel, abs_tol=max(5.0, stars * 0.005)):
            return True, f"[STRUCTURED:github_api] {repo} stars={stars}"
    return False, ""


@dataclass
class ClaimHit:
    raw: str
    value: Optional[float]
    kind: str  # number | stars | percent | power
    verdict: str = "unverified"  # supported | unsupported | conflict | unverified
    evidence_snip: str = ""


@dataclass
class ClaimVerifyReport:
    claims: list[ClaimHit] = field(default_factory=list)
    citation_ok: Optional[bool] = None
    unsupported: list[ClaimHit] = field(default_factory=list)
    conflicts: list[ClaimHit] = field(default_factory=list)
    gap_queries: list[str] = field(default_factory=list)
    ok: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "citation_ok": self.citation_ok,
            "claims_checked": [
                {
                    "raw": c.raw,
                    "value": c.value,
                    "kind": c.kind,
                    "verdict": c.verdict,
                    "evidence_snip": c.evidence_snip[:160],
                }
                for c in self.claims
            ],
            "unsupported": [c.raw for c in self.unsupported],
            "conflicts": [c.raw for c in self.conflicts],
            "gap_queries": self.gap_queries,
            "note": self.note,
        }


def _parse_number(raw: str) -> Optional[float]:
    s = (raw or "").strip().replace(" ", "").replace("\u00a0", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and s.count(",") == 1 and len(s.split(",")[-1]) <= 2:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _scale_mult(scale: str) -> float:
    s = (scale or "").strip().lower()
    if s in ("b", "млрд", "billion"):
        return 1e9
    if s in ("m", "млн", "million"):
        return 1e6
    if s in ("k", "тыс"):
        return 1e3
    return 1.0


def extract_claims(text: str, *, max_claims: int = 24) -> list[ClaimHit]:
    body = re.sub(r"[*`_]+", " ", text or "")
    hits: list[ClaimHit] = []
    seen: set[str] = set()

    for m in _STARS_RE.finditer(body):
        raw = m.group(0).strip()
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        n = _parse_number(m.group("n") or "")
        # ``38,2k`` / ``38.2k`` / ``31k`` — k may be glued to digits (no \b before k).
        if n is not None and re.search(r"(?:(?<=[\d.,])k\b|\bk\b|тыс)", raw, re.I):
            n *= 1000.0
        hits.append(ClaimHit(raw=raw, value=n, kind="stars"))
        if len(hits) >= max_claims:
            return hits

    for m in _NUM_CLAIM_RE.finditer(body):
        raw = m.group(0).strip()
        key = raw.lower()
        if key in seen or len(raw) < 2:
            continue
        seen.add(key)
        if m.group("n"):
            n = _parse_number(m.group("n") or "")
            if n is not None:
                n *= _scale_mult(m.group("scale") or "")
            kind = "number"
            suf = (m.group("scale") or "").lower()
        else:
            n = _parse_number(m.group("n2") or "")
            suf = (m.group("suf") or "").lower()
            if n is not None and suf in ("k", "тыс"):
                n *= 1000.0
            elif n is not None and suf in ("млн", "million"):
                n *= 1e6
            elif n is not None and suf in ("млрд", "billion"):
                n *= 1e9
            if "star" in suf or "зв" in suf:
                kind = "stars"
            elif "%" in suf:
                kind = "percent"
            elif "гвт" in suf or suf == "gw" or "мвт" in suf or suf == "mw":
                kind = "power"
            else:
                kind = "number"
        hits.append(ClaimHit(raw=raw, value=n, kind=kind))
        if len(hits) >= max_claims:
            break
    return hits


def _approx_equal(a: float, b: float, *, rel: float = 0.08, abs_tol: float = 0.5) -> bool:
    if a == 0 and b == 0:
        return True
    diff = abs(a - b)
    if diff <= abs_tol:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return (diff / scale) <= rel


def _evidence_numbers(evidence: str) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for m in _NUM_CLAIM_RE.finditer(evidence or ""):
        raw = m.group(0)
        if m.group("n"):
            n = _parse_number(m.group("n") or "")
            if n is not None:
                n *= _scale_mult(m.group("scale") or "")
        else:
            n = _parse_number(m.group("n2") or "")
            suf = (m.group("suf") or "").lower()
            if n is not None and suf in ("k", "тыс"):
                n *= 1000.0
            elif n is not None and suf in ("млн", "million"):
                n *= 1e6
            elif n is not None and suf in ("млрд", "billion"):
                n *= 1e9
        if n is not None:
            out.append((n, raw))
    return out


def _evidence_primary_numbers(evidence: str) -> list[tuple[float, str]]:
    """
    Numbers that sit near a primary-source URL in the evidence blob.
    Used so blog teasers cannot 'support' a figure when .gov/IR disagrees.
    """
    try:
        from .url_quality import is_primary_source_url
    except Exception:
        return []
    ev = evidence or ""
    if not ev:
        return []
    chunks: list[str] = []
    for m in re.finditer(r"https?://[^\s\)\]\"\'>]+", ev):
        url = m.group(0).rstrip(".,;:\"'")
        if not is_primary_source_url(url):
            continue
        # Keep numbers local to this URL — do not bleed past the next URL
        # (blogs often sit on the next line with conflicting figures).
        before = ev[max(0, m.start() - 400) : m.start()]
        after = ev[m.end() : min(len(ev), m.end() + 220)]
        nxt = re.search(r"https?://", after)
        if nxt:
            after = after[: nxt.start()]
        chunks.append(before + url + after)
    # STRUCTURED github rows only — do not pull blog text after the block.
    for m in re.finditer(
        r"\[STRUCTURED:github_api\][^\n]*(?:\n(?:repo|stars|forks|license|full_name|watchers|open_issues|updated|html_url)=[^\n]*)*",
        ev,
        flags=re.IGNORECASE,
    ):
        chunks.append(m.group(0))
    if not chunks:
        return []
    out: list[tuple[float, str]] = []
    seen: set[float] = set()
    for ch in chunks:
        for n, raw in _evidence_numbers(ch):
            key = round(n, 6)
            if key in seen:
                continue
            seen.add(key)
            out.append((n, raw))
    return out


def verify_claims_against_evidence(
    draft_answer: str,
    evidence: str,
    *,
    citation_urls: Optional[list[str]] = None,
    require_citations: bool = False,
    focus_claims: Optional[list[str]] = None,
) -> ClaimVerifyReport:
    """
    Check numeric/stars claims in ``draft_answer`` against ``evidence``
    (typically last web_search ``synthesized`` + snippets).
    """
    report = ClaimVerifyReport()
    ev = evidence or ""
    claims = extract_claims(draft_answer)
    if focus_claims:
        focus_l = {c.strip().lower() for c in focus_claims if (c or "").strip()}
        if focus_l:
            claims = [c for c in claims if c.raw.lower() in focus_l] or claims

    structured_stars = extract_structured_github_stars(ev)
    ev_nums = _evidence_numbers(ev)
    # When STRUCTURED github rows exist, blog star counts must not "support" a wrong claim.
    if structured_stars:
        structured_vals = {float(v) for v in structured_stars.values()}
        ev_nums = [
            (n, raw)
            for n, raw in ev_nums
            if n in structured_vals
            or "[structured:github_api]" in raw.lower()
            or "stars=" in raw.lower()
            or not re.search(r"stars?|зв[её]зд", raw, re.I)
        ]
        for repo, stars in structured_stars.items():
            ev_nums.append((float(stars), f"[STRUCTURED:github_api] {repo} stars={stars}"))
    ev_lower = ev.lower()

    for claim in claims:
        # Star claims: STRUCTURED github_api is authoritative when present.
        if claim.kind == "stars" and structured_stars and claim.value is not None:
            ok_s, snip = _star_claims_match_structured(claim.value, structured_stars)
            if ok_s:
                claim.verdict = "supported"
                claim.evidence_snip = snip
            else:
                claim.verdict = "unsupported"
                best = ", ".join(f"{r}={n}" for r, n in list(structured_stars.items())[:4])
                claim.evidence_snip = f"STRUCTURED stars only: {best}"
                report.unsupported.append(claim)
            report.claims.append(claim)
            continue

        if claim.value is None:
            # Literal / near-literal support for non-numeric leftovers
            token = re.sub(r"\s+", "", claim.raw.lower())
            if token and token in re.sub(r"\s+", "", ev_lower):
                claim.verdict = "supported"
                claim.evidence_snip = claim.raw
            else:
                claim.verdict = "unverified"
            report.claims.append(claim)
            continue

        primary_nums = _evidence_primary_numbers(ev)
        # ChatGPT-like: when primary sources state a figure, blog-only support is not enough
        # if the claim disagrees with primary.
        if primary_nums and claim.kind in ("number", "power", "percent", "stars"):
            p_match = [raw for n, raw in primary_nums if _approx_equal(claim.value, n)]
            p_near = [
                raw
                for n, raw in primary_nums
                if _approx_equal(
                    claim.value, n, rel=0.25, abs_tol=max(1.0, claim.value * 0.05)
                )
            ]
            if p_match:
                claim.verdict = "supported"
                claim.evidence_snip = "primary: " + p_match[0]
                report.claims.append(claim)
                continue
            blog_matches = [raw for n, raw in ev_nums if _approx_equal(claim.value, n)]
            token = re.sub(r"\s+", "", claim.raw.lower())
            literal_in_ev = bool(token and token in re.sub(r"\s+", "", ev_lower))
            if p_near or blog_matches or literal_in_ev:
                # Official figure exists and claim is wrong / only backed by blogs.
                claim.verdict = "conflict"
                claim.evidence_snip = (
                    "primary: "
                    + " | ".join(raw for _, raw in primary_nums[:3])
                    + ("; blog said: " + (blog_matches[0] if blog_matches else claim.raw))
                )
                report.conflicts.append(claim)
                report.claims.append(claim)
                continue
            claim.verdict = "unsupported"
            claim.evidence_snip = "primary present, claim absent"
            report.unsupported.append(claim)
            report.claims.append(claim)
            continue

        # Literal / near-literal support (no primary conflict path)
        token = re.sub(r"\s+", "", claim.raw.lower())
        if token and token in re.sub(r"\s+", "", ev_lower):
            claim.verdict = "supported"
            claim.evidence_snip = claim.raw
            report.claims.append(claim)
            continue

        matches = [raw for n, raw in ev_nums if _approx_equal(claim.value, n)]
        near = [raw for n, raw in ev_nums if _approx_equal(claim.value, n, rel=0.25, abs_tol=max(1.0, claim.value * 0.05))]
        if matches:
            claim.verdict = "supported"
            claim.evidence_snip = matches[0]
        elif near and len(near) >= 2:
            # Several nearby but not close enough → conflict cluster
            claim.verdict = "conflict"
            claim.evidence_snip = " | ".join(near[:3])
            report.conflicts.append(claim)
        elif near:
            claim.verdict = "unsupported"
            claim.evidence_snip = near[0]
            report.unsupported.append(claim)
        else:
            claim.verdict = "unsupported"
            report.unsupported.append(claim)
        report.claims.append(claim)

    # If evidence has STRUCTURED stars but the answer never cites a matching figure,
    # treat as a gap (ChatGPT would use the API number).
    if structured_stars:
        star_claims = [c for c in report.claims if c.kind == "stars"]
        covered = any(c.verdict == "supported" for c in star_claims)
        if not covered:
            compact_ans = re.sub(r"[\s,\u00a0]", "", draft_answer or "")
            for _repo, stars in structured_stars.items():
                if str(int(stars)) in compact_ans:
                    covered = True
                    break
                for c in report.claims:
                    if c.value is not None and _approx_equal(
                        c.value, float(stars), rel=0.02, abs_tol=5.0
                    ):
                        covered = True
                        break
                if covered:
                    break
        if not covered:
            for repo, stars in list(structured_stars.items())[:3]:
                miss = ClaimHit(
                    raw=f"{stars} stars",
                    value=float(stars),
                    kind="stars",
                    verdict="unsupported",
                    evidence_snip=f"missing STRUCTURED {repo} stars={stars}",
                )
                report.claims.append(miss)
                report.unsupported.append(miss)
                report.gap_queries.append(
                    f"use STRUCTURED github_api stars for {repo}: {stars}"
                )
                break
        elif any(c.verdict == "unsupported" for c in star_claims):
            for repo, stars in list(structured_stars.items())[:3]:
                q = f"use STRUCTURED github_api stars for {repo}: {stars}"
                if q not in report.gap_queries:
                    report.gap_queries.append(q)

    if require_citations or citation_urls:
        report.citation_ok = answer_references_allowed_web_sources(
            draft_answer, list(citation_urls or [])
        )

    bad = report.unsupported + report.conflicts
    report.ok = not bad and (report.citation_ok is not False if require_citations else True)

    for c in bad[:6]:
        q = c.raw.strip()
        if q and q not in report.gap_queries:
            report.gap_queries.append(f"verify figure {q}")

    if not report.claims:
        report.note = "No numeric claims extracted from draft."
        report.ok = True if report.citation_ok is not False else False
    elif report.ok:
        report.note = f"Checked {len(report.claims)} claim(s); all supported."
    else:
        report.note = (
            f"Checked {len(report.claims)} claim(s); "
            f"unsupported={len(report.unsupported)} conflict={len(report.conflicts)}."
        )
    return report


def looks_like_research_answer(text: str) -> bool:
    """Heuristic: long answer with many figures → worth a verify pass."""
    s = (text or "").strip()
    if len(s) < 700:
        return False
    claims = extract_claims(s, max_claims=12)
    return len(claims) >= 4


def host_from_url(url: str) -> str:
    try:
        return urlparse(url or "").netloc.lower().replace("www.", "")
    except Exception:
        return ""
