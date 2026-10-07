# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Answer-quality gates for native web research (collapse, filler, junk structure).

Complements claim_verify (numbers) — catches failures that pass numeric checks
but are unusable as research deliverables.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class AnswerQualityReport:
    ok: bool = True
    reasons: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "detail": self.detail,
        }


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ0-9]{3,}")


def _repetition_ratio(text: str) -> float:
    """Fraction of tokens that are the single most-repeated token (min length 4)."""
    tokens = [t.lower() for t in _WORD_RE.findall(text or "") if len(t) >= 4]
    if len(tokens) < 40:
        return 0.0
    from collections import Counter

    counts = Counter(tokens)
    top_n, top_c = counts.most_common(1)[0]
    # Ignore common glue words
    if top_n in {
        "that",
        "this",
        "with",
        "from",
        "который",
        "которая",
        "которые",
        "данные",
        "market",
        "github",
    }:
        # use second most common if available
        if len(counts) < 2:
            return 0.0
        top_n, top_c = counts.most_common(2)[1]
    return top_c / max(len(tokens), 1)


def _is_markdown_chrome(fragment: str) -> bool:
    """True for table rules / hr / separator noise — not a real phrase loop."""
    f = (fragment or "").strip()
    if not f:
        return True
    # e.g. ------------ or |---|---|
    if re.fullmatch(r"[\s\-|:=]+", f):
        return True
    # Must contain at least one letter/digit to count as a linguistic loop.
    if not re.search(r"[A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ0-9]", f):
        return True
    return False


def _has_phrase_loop(text: str) -> bool:
    """Detect copy-paste loops like 'однозначные однозначные…' or 'the the the'."""
    s = (text or "").strip()
    if len(s) < 80:
        return False
    # Same 6–40 char token repeated 8+ times in a row (with spaces)
    for m in re.finditer(
        r"([^\s]{4,40})(?:\s+\1){7,}",
        s,
        flags=re.IGNORECASE,
    ):
        if not _is_markdown_chrome(m.group(1)):
            return True
    # Same short sentence fragment repeated (ignore markdown table rules).
    for m in re.finditer(
        r"(.{12,80})\s*\1\s*\1\s*\1",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        if not _is_markdown_chrome(m.group(1)):
            return True
    return False


def _ellipsis_collapse(text: str) -> bool:
    """Trailing garbage of only dots / one word + dots."""
    s = (text or "").rstrip()
    if len(s) < 200:
        return False
    tail = s[-400:]
    if re.search(r"(?:\.{3,}|……){2,}", tail):
        return True
    if re.search(r"(\b\w{4,}\b[\s.,]*){0,3}\.{6,}\s*$", tail):
        return True
    return False


def assess_answer_quality(text: str, *, research: bool = False) -> AnswerQualityReport:
    """
    Return ok=False when the answer is structurally unusable.
    """
    report = AnswerQualityReport()
    s = (text or "").strip()
    if not s:
        report.ok = False
        report.reasons.append("empty")
        report.detail = "Empty answer."
        return report

    if _has_phrase_loop(s):
        report.ok = False
        report.reasons.append("phrase_loop")

    ratio = _repetition_ratio(s)
    if ratio >= 0.22:
        report.ok = False
        report.reasons.append(f"token_repetition={ratio:.2f}")

    if _ellipsis_collapse(s):
        report.ok = False
        report.reasons.append("ellipsis_collapse")

    # Research answers must keep readable structure — not a single broken paragraph of filler.
    if research and len(s) >= 1200:
        # Count unique 5-grams; very low diversity → collapse
        words = _WORD_RE.findall(s.lower())
        if len(words) >= 80:
            grams = [" ".join(words[i : i + 5]) for i in range(0, len(words) - 4, 3)]
            if grams:
                uniq = len(set(grams)) / len(grams)
                if uniq < 0.35:
                    report.ok = False
                    report.reasons.append(f"low_ngram_diversity={uniq:.2f}")

    if not report.ok:
        report.detail = "Answer quality fail: " + ", ".join(report.reasons)
    return report


def looks_like_collapsed_answer(text: str) -> bool:
    return not assess_answer_quality(text, research=True).ok
