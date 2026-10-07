# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Numeric critic — reflection on figures before the answer ships.

Domain-agnostic: no topic allowlists. Extracts numbers from evidence, checks
place-lock and outlier clusters, emits a critic block for the synthesizer /
answer model.
"""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from .evidence_gate import extract_content_nouns, extract_place_tokens
from .schema import SearchResult

logger = logging.getLogger(__name__)

_NUM_CAPTURE_RE = re.compile(
    r"(?P<cur>[$€£¥₽₸])\s?(?P<n>\d[\d\s.,]{0,14})|"
    r"(?P<n2>\d[\d\s.,]{1,14})\s?(?P<suf>€|EUR|USD|\$|₽|₸|zł|£|%|°[CF]?|k\b|тыс|млн)",
    re.I,
)


@dataclass
class FigureHit:
    value: float
    raw: str
    domain: str
    context: str
    place_ok: bool
    noun_ok: bool


@dataclass
class CriticReport:
    figures: list[FigureHit] = field(default_factory=list)
    outliers: list[FigureHit] = field(default_factory=list)
    off_place: list[FigureHit] = field(default_factory=list)
    note: str = ""
    needs_more_search: bool = False


def critic_enabled() -> bool:
    raw = (os.getenv("FINKEY_WEB_NUMERIC_CRITIC", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _parse_number(raw: str) -> Optional[float]:
    s = (raw or "").strip().replace(" ", "").replace("\u00a0", "")
    if not s:
        return None
    # European 1.234,56 vs 1,234.56
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


def _domain(url: str) -> str:
    try:
        return urlparse(url or "").netloc.lower().replace("www.", "") or "?"
    except Exception:
        return "?"


def extract_figures(
    message: str,
    results: list[SearchResult],
    *,
    max_per_row: int = 12,
    max_total: int = 80,
) -> list[FigureHit]:
    places = [p.lower() for p in extract_place_tokens(message)[:4]]
    nouns = extract_content_nouns(message)[:8]
    hits: list[FigureHit] = []
    for r in results or []:
        blob = (r.enriched_text or r.snippet or "").strip()
        if not blob:
            continue
        title = (r.title or "").lower()
        primary = f"{title}\n{(r.url or '').lower()}\n{(r.snippet or '').lower()}"
        text = blob[:20_000]
        dom = _domain(r.url or "")
        count = 0
        for m in _NUM_CAPTURE_RE.finditer(text):
            raw_n = m.group("n") or m.group("n2") or ""
            val = _parse_number(raw_n)
            if val is None or val <= 0:
                continue
            # Skip tiny ints that are years/dates noise unless currency/temp
            cur = (m.group("cur") or m.group("suf") or "").lower()
            if not cur and 1900 <= val <= 2100:
                continue
            start = max(0, m.start() - 160)
            end = min(len(text), m.end() + 160)
            ctx = re.sub(r"\s+", " ", text[start:end]).strip()
            ctx_l = ctx.lower()
            para_start = text.rfind("\n", 0, m.start())
            para_end = text.find("\n", m.end())
            if para_start < 0:
                para_start = max(0, m.start() - 240)
            if para_end < 0:
                para_end = min(len(text), m.end() + 240)
            para_l = text[para_start:para_end].lower()
            if not places:
                place_ok = True
            else:
                local_place = any(p in ctx_l or p in para_l for p in places)
                title_place = any(p in title for p in places)
                # Title on-place is enough to keep the row's figures in the sheet
                # (neighborhood bullets often omit repeating the city name).
                place_ok = local_place or title_place
            noun_ok = (not nouns) or any(n in ctx_l or n in primary for n in nouns)
            raw = (m.group(0) or "").strip()
            hits.append(
                FigureHit(
                    value=val,
                    raw=raw[:48],
                    domain=dom,
                    context=ctx[:160],
                    place_ok=place_ok,
                    noun_ok=noun_ok,
                )
            )
            count += 1
            if count >= max_per_row or len(hits) >= max_total:
                break
        if len(hits) >= max_total:
            break
    return hits


def _outlier_mask(values: list[float]) -> list[bool]:
    """Mark values >10× median of positive cluster (log-robust)."""
    if len(values) < 4:
        return [False] * len(values)
    pos = sorted(v for v in values if v > 0)
    mid = pos[len(pos) // 2]
    if mid <= 0:
        return [False] * len(values)
    return [v > 0 and (v / mid >= 10.0 or mid / v >= 10.0) for v in values]


def run_numeric_critic(message: str, results: list[SearchResult]) -> CriticReport:
    if not critic_enabled():
        return CriticReport()
    figs = extract_figures(message, results)
    report = CriticReport(figures=figs)
    if not figs:
        report.needs_more_search = True
        report.note = (
            "NUMERIC CRITIC: no usable figures found in excerpts. "
            "Do not invent numbers; say what is missing or answer qualitatively."
        )
        return report

    vals = [f.value for f in figs]
    out_mask = _outlier_mask(vals)
    for f, is_out in zip(figs, out_mask):
        if is_out:
            report.outliers.append(f)
        if not f.place_ok:
            report.off_place.append(f)

    places = extract_place_tokens(message)[:4]
    place_locked = [f for f in figs if f.place_ok]
    lines = [
        "NUMERIC CRITIC (pre-answer):",
        f"  figures={len(figs)} place_locked={len(place_locked)} "
        f"outliers={len(report.outliers)} off_place={len(report.off_place)}",
    ]
    if places:
        lines.append(f"  place_tokens={places}")
    # Sample trusted cluster
    trusted = [f for f in figs if f.place_ok and f not in report.outliers][:8]
    if trusted:
        lines.append("  trusted cluster samples:")
        for f in trusted:
            lines.append(f"    • {f.raw} ({f.domain}) — {f.context[:90]}")
    if report.outliers:
        lines.append("  OUTLIERS (treat as suspect, label separately):")
        for f in report.outliers[:5]:
            lines.append(f"    • {f.raw} ({f.domain}) — {f.context[:90]}")
    if report.off_place and places:
        lines.append(
            "  OFF-PLACE figures (do NOT use for the asked place unless excerpt "
            "explicitly compares):"
        )
        for f in report.off_place[:5]:
            lines.append(f"    • {f.raw} ({f.domain}) — {f.context[:90]}")
    lines.append(
        "  RULES: prefer place_locked non-outlier figures; cite domain per number; "
        "if trusted cluster empty — refuse invented precision."
    )
    report.note = "\n".join(lines)
    # Need more search if almost no place-locked numbers when place was asked
    if places and len(place_locked) < 2:
        report.needs_more_search = True
    logger.info(
        "Numeric critic: figs=%d place_ok=%d outliers=%d need_more=%s",
        len(figs),
        len(place_locked),
        len(report.outliers),
        report.needs_more_search,
    )
    return report


__all__ = [
    "CriticReport",
    "FigureHit",
    "critic_enabled",
    "extract_figures",
    "run_numeric_critic",
]
