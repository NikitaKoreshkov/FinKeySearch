# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Structured (non-SERP) fact backends for web search."""

from .github_repo import (
    fetch_github_repo_facts,
    github_structured_enabled,
    merge_github_structured_results,
    parse_github_repos,
)

__all__ = [
    "fetch_github_repo_facts",
    "github_structured_enabled",
    "merge_github_structured_results",
    "parse_github_repos",
]
