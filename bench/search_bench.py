# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Deterministic micro-benchmarks for the resilience machinery: hedged fallback chain,
provider-failure availability, deadline-bounded page enrichment, breaker/limiter contracts.

No API keys, no network. Synthetic backends sleep for controlled amounts of time with
seeded jitter, so runs are reproducible in distribution. Latencies are production values
scaled by SCALE (a 2000ms hedge becomes 200ms here); the benchmark measures ratios.

    PYTHONPATH=src python3 bench/search_bench.py

Writes bench/results/resilience.json and figures/*.png.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finkey_search.backends.composite import FallbackSearchChain, GuardedBackend  # noqa: E402
from finkey_search.page_enrichment import enrich_results_with_structured_pages  # noqa: E402
from finkey_search.resilience import CircuitBreaker, SlidingWindowLimiter  # noqa: E402
from finkey_search.schema import SearchDepth, SearchResult  # noqa: E402
from finkey_search.url_policy import URLPolicy  # noqa: E402

SCALE = 0.1
TRIALS = 24
REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "figures"
RESULTS = REPO / "bench" / "results"

EVENTS: list[tuple[str, float, float]] = []
_EVENTS_LOCK = threading.Lock()


@dataclass
class Spec:
    name: str
    latency_s: float
    fail_rate: float = 0.0
    empty: bool = False
    seed: int = 0


class SynthBackend:
    """A search backend with fixed latency, optional error rate and empty-answer modes."""

    def __init__(self, spec: Spec) -> None:
        self.name = spec.name
        self._s = spec

    def search(self, query: str, *, max_results: int, language: str, date_restrict: str):
        rng = random.Random(f"{self._s.seed}:{query}")
        jitter = 1.0 + rng.uniform(-0.10, 0.15)
        started = time.perf_counter()
        time.sleep(max(0.001, self._s.latency_s * jitter))
        finished = time.perf_counter()
        with _EVENTS_LOCK:
            EVENTS.append((self.name, started, finished))
        if self._s.fail_rate and rng.random() < self._s.fail_rate:
            raise ConnectionError(f"{self.name}: simulated provider failure")
        if self._s.empty:
            return []
        return [
            SearchResult(
                title=f"{self.name} result {i}",
                url=f"https://{self.name}.test/doc{i}",
                snippet=f"synthetic payload from {self.name}",
                source=self.name,
                relevance=0.8,
            )
            for i in range(max_results)
        ]


def build_chain(specs: list[Spec]) -> FallbackSearchChain:
    return FallbackSearchChain(
        [GuardedBackend(inner=SynthBackend(s), allow=lambda: True) for s in specs]
    )


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def measure(fn, trials: int) -> list[float]:
    out: list[float] = []
    for t in range(trials):
        EVENTS.clear()
        t0 = time.perf_counter()
        fn(f"bench-{t}")
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


# ------------------------------------------------------------------ hedge vs serial

SCENARIOS: dict[str, list[Spec]] = {
    "healthy": [
        Spec("primary", 0.035, seed=1),
        Spec("backup-a", 0.050, seed=2),
        Spec("backup-b", 0.065, seed=3),
    ],
    "primary retry storm": [
        Spec("primary", 0.260, seed=4),
        Spec("backup-a", 0.050, seed=2),
        Spec("backup-b", 0.060, seed=3),
    ],
    "primary empty, first backup dies": [
        Spec("primary", 0.120, empty=True, seed=5),
        Spec("backup-a", 0.400, fail_rate=1.0, seed=6),
        Spec("backup-b", 0.060, seed=7),
    ],
    "primary hard failure": [
        Spec("primary", 0.500, fail_rate=1.0, seed=8),
        Spec("backup-a", 0.070, seed=9),
        Spec("backup-b", 0.090, seed=10),
    ],
}


def bench_hedge() -> dict:
    rows: dict[str, dict] = {}
    for label, specs in SCENARIOS.items():
        modes: dict[str, dict[str, float]] = {}
        for mode, env_val in (("serial", "0"), ("hedged", str(int(2000 * SCALE)))):
            os.environ["FINKEY_SEARCH_HEDGE_MS"] = env_val
            chain = build_chain(specs)
            lat = measure(
                lambda q, c=chain: c.search(
                    q, max_results=5, language="en", date_restrict=""
                ),
                TRIALS,
            )
            modes[mode] = {
                "p50": round(_pct(lat, 0.50), 1),
                "p95": round(_pct(lat, 0.95), 1),
            }
        gain = modes["serial"]["p95"] / max(modes["hedged"]["p95"], 0.001)
        modes["p95_ratio"] = {"x": round(gain, 2), "percent": round((gain - 1) * 100, 1)}
        rows[label] = modes
    os.environ.pop("FINKEY_SEARCH_HEDGE_MS", None)
    return rows


# -------------------------------------------------------------------- availability

def bench_availability() -> dict:
    error_rates = [0.10, 0.20, 0.30, 0.40]
    chain_sizes = [1, 2, 3, 4]
    trials = 80
    os.environ["FINKEY_SEARCH_HEDGE_MS"] = "0"
    measured: dict[str, float] = {}
    for p in error_rates:
        for k in chain_sizes:
            chain = build_chain([Spec(f"b{i}", 0.008, fail_rate=p, seed=100 + i) for i in range(k)])
            hits = sum(
                1 for t in range(trials)
                if chain.search(f"q{t}", max_results=3, language="en", date_restrict="")
            )
            measured[f"p={int(p * 100)}|k={k}"] = round(100.0 * hits / trials, 1)
    os.environ.pop("FINKEY_SEARCH_HEDGE_MS", None)
    analytical = {
        f"p={int(p * 100)}|k={k}": round(100.0 * (1 - p ** k), 1)
        for p in error_rates for k in chain_sizes
    }
    return {
        "measured": measured,
        "analytical": analytical,
        "trials": trials * len(chain_sizes) * len(error_rates),
    }


# --------------------------------------------------------- enrichment deadline bound

_PAGE_LATENCY: dict[str, float] = {}
_STRAGGLER_S = 0.600


def _make_pages(count: int, rng: random.Random) -> list[SearchResult]:
    rows: list[SearchResult] = []
    _PAGE_LATENCY.clear()
    for i in range(count):
        url = f"https://host{i % 7}.test/p{i}"
        lat = rng.choice([0.030, 0.055, 0.090]) + rng.uniform(0, 0.02)
        if i % 5 == 4:
            lat += _STRAGGLER_S  # anti-bot host or cold CDN
        _PAGE_LATENCY[url] = lat
        rows.append(
            SearchResult(
                title=f"page {i}", url=url, snippet="s",
                source=f"host{i % 7}", relevance=0.9 - i * 0.01,
            )
        )
    return rows


class _OkPage:
    ok = True
    error = ""
    screenshot = b""

    def __init__(self, text: str) -> None:
        self.main_content = text


def bench_enrichment() -> dict:
    import finkey_search.page_enrichment as pe

    def fake_fetch(url: str, cfg):
        """A fetcher cannot outlive its own timeout, so honour the budget we are given."""
        lat = _PAGE_LATENCY.get(url, 0.05)
        budget = cfg.timeout_ms / 1000.0
        time.sleep(min(lat, budget))
        if lat > budget:
            page = _OkPage("")
            page.ok = False
            page.error = "timeout"
            return page
        return _OkPage("x" * 600)

    policy = URLPolicy(block_private_and_loopback=False)
    rng = random.Random(7)
    original = pe.fetch_structured_sync
    pe.fetch_structured_sync = fake_fetch
    out: dict[str, dict[str, float]] = {}
    try:
        for label, (conc, burst, deadline) in {
            "one page at a time": ("1", "1", None),
            "parallel wave, no deadline": ("6", "12", None),
            "parallel wave, 0.35s deadline": ("6", "12", 0.35),
            "one page at a time, hostile straggler": ("1", "1", None),
            "deadline cap, hostile straggler": ("6", "12", 0.35),
        }.items():
            global _STRAGGLER_S
            _STRAGGLER_S = (
                3.0 if label.endswith("hostile straggler") else 0.600
            )
            os.environ["FINKEY_PAGE_ENRICH_CONCURRENCY"] = conc
            os.environ["FINKEY_PAGE_ENRICH_MAX_BURST"] = burst
            rows = _make_pages(12, rng)
            t0 = time.perf_counter()
            enrich_results_with_structured_pages(
                rows, max_pages=12, url_policy=policy,
                depth=SearchDepth.DEEP, deadline_s=deadline, http_fallback=False,
            )
            out[label] = {
                "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "pages_kept": sum(1 for r in rows if r.enriched_text),
            }
    finally:
        pe.fetch_structured_sync = original
        for key in ("FINKEY_PAGE_ENRICH_CONCURRENCY", "FINKEY_PAGE_ENRICH_MAX_BURST"):
            os.environ.pop(key, None)
    return out


# ------------------------------------------------------------------------ contracts

def bench_contracts() -> dict:
    checks: dict[str, str] = {}

    brk = CircuitBreaker(failure_threshold=3, recovery_seconds=0.20)
    for _ in range(3):
        brk.record_failure()
    checks["breaker opens after 3 failures"] = "PASS" if not brk.allow_request() else "FAIL"
    time.sleep(0.22)
    checks["breaker half-opens after the window"] = "PASS" if brk.allow_request() else "FAIL"
    brk.record_success()
    checks["breaker closes on a success"] = "PASS" if brk.allow_request() else "FAIL"

    lim = SlidingWindowLimiter(max_calls=5, window_seconds=0.25)
    admitted = sum(1 for _ in range(9) if lim.acquire())
    checks["limiter admits exactly the quota"] = (
        "PASS" if admitted == 5 else f"FAIL({admitted}/5)"
    )
    time.sleep(0.27)
    checks["limiter window slides"] = "PASS" if lim.acquire() else "FAIL"

    os.environ["FINKEY_SEARCH_HEDGE_MS"] = "0"
    try:
        chain = build_chain(SCENARIOS["primary empty, first backup dies"])
        EVENTS.clear()
        chain.search("contract-serial", max_results=3, language="en", date_restrict="")
        ev = {name: (s, f) for name, s, f in EVENTS}
        strictly_serial = ev["backup-a"][0] >= ev["primary"][1] and ev["backup-b"][0] >= ev["backup-a"][1]
        checks["hedge=0 never overlaps backends"] = "PASS" if strictly_serial else "FAIL"
    finally:
        os.environ["FINKEY_SEARCH_HEDGE_MS"] = "0"

    os.environ["FINKEY_SEARCH_HEDGE_MS"] = str(int(2000 * SCALE))
    try:
        EVENTS.clear()
        chain = build_chain(SCENARIOS["primary retry storm"])
        rows = chain.search("contract-hedge", max_results=3, language="en", date_restrict="")
        ev = {name: (s, f) for name, s, f in EVENTS}
        overlapped = "backup-a" in ev and ev["backup-a"][0] < ev["primary"][1]
        kept_priority = bool(rows) and rows[0].source == "primary"
        checks["hedge overlaps backups with a slow primary"] = "PASS" if overlapped else "FAIL"
        checks["hedge keeps the preferred provider's answer"] = "PASS" if kept_priority else "FAIL"
    finally:
        os.environ.pop("FINKEY_SEARCH_HEDGE_MS", None)
    return checks


# ------------------------------------------------------------------------- plots

def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_hedge(h: dict) -> None:
    import matplotlib.pyplot as plt

    labels = list(h)
    x = range(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=150)
    for off, (mode, stat, color) in enumerate([
        ("serial", "p50", "#8d99ae"), ("hedged", "p50", "#2ec4b6"),
        ("serial", "p95", "#3d4451"), ("hedged", "p95", "#07d399"),
    ]):
        vals = [h[l][mode][stat] for l in labels]
        ax.bar([i + (off - 1.5) * w for i in x], vals, w,
               label=f"{mode} {stat.upper()}", color=color)
    for i, l in enumerate(labels):
        ratio = h[l]["p95_ratio"]["x"]
        gain = f"{ratio:.2f}x" if ratio >= 1.05 else "parity"
        ax.text(i + 1.5 * w, h[l]["hedged"]["p95"] + 6, gain,
                ha="center", fontsize=8, color="#07d399")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    _style(ax, "Fallback chain tail latency — hedged vs strictly sequential (ms)", "ms")
    ax.legend(fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(FIGS / "hedge_latency.png")
    plt.close(fig)


def plot_availability(a: dict) -> None:
    import matplotlib.pyplot as plt

    rates = [10, 20, 30, 40]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    colors = {1: "#8d99ae", 2: "#f4a261", 3: "#2ec4b6", 4: "#07d399"}
    for k in (1, 2, 3, 4):
        ys = [a["measured"][f"p={p}|k={k}"] for p in rates]
        ax.plot(rates, ys, marker="o", color=colors[k], linewidth=2,
                label=f"{k} provider" + ("s" if k > 1 else ""))
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r}%" for r in rates])
    _style(ax, "Answer availability vs per-provider error rate", "% of queries answered")
    ax.set_xlabel("per-provider failure probability", fontsize=9)
    ax.set_ylim(0, 104)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "availability.png")
    plt.close(fig)


def plot_enrichment(e: dict) -> None:
    import matplotlib.pyplot as plt

    labels = list(e)
    walls = [e[l]["wall_ms"] for l in labels]
    fig, ax = plt.subplots(figsize=(8.6, 4.0), dpi=150)
    bars = ax.barh(range(len(labels)), walls,
                   color=["#8d99ae", "#2ec4b6", "#07d399"], height=0.5)
    for i, label in enumerate(labels):
        kept = e[label]["pages_kept"]
        ax.text(bars[i].get_width() + 10, i, f"{kept}/12 pages kept",
                va="center", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    _style(ax, "Page enrichment wall clock — 12 pages, one 600ms straggler", "ms")
    ax.set_xlim(0, max(walls) * 1.32)
    fig.tight_layout()
    fig.savefig(FIGS / "enrichment.png")
    plt.close(fig)


def write_results_md(h: dict, a: dict, e: dict, c: dict) -> None:
    """README/bench prose is generated from the same dict the plots use, so numbers cannot drift."""
    lines = [
        "# Measured results",
        "",
        "Regenerate: `PYTHONPATH=src python3 bench/search_bench.py`",
        "No network, no API keys. Latencies are production values scaled by 0.1 "
        "(a 2000ms hedge is 200ms here); ratios are the finding.",
        "",
        "## Hedged fallback chain vs strictly sequential (P95, ms)",
        "",
        "| scenario | serial | hedged | ratio |",
        "|---|---|---|---|",
    ]
    for label, modes in h.items():
        lines.append(
            f"| {label} | {modes['serial']['p95']:.0f} | "
            f"{modes['hedged']['p95']:.0f} | {modes['p95_ratio']['x']:.2f}x |"
        )
    lines += [
        "",
        "Parity where the preferred provider is merely slow is intentional: the chain never",
        "trades the preferred provider's answer for a faster one from a backup.",
        "",
        "## Availability vs per-provider error rate",
        "",
        "| failure rate | 1 provider | 2 | 3 | 4 |",
        "|---|---|---|---|---|",
    ]
    for p_rate in (10, 20, 30, 40):
        cells = " | ".join(
            f"{a['measured'][f'p={p_rate}|k={k}']:.0f}% "
            f"(theory {a['analytical'][f'p={p_rate}|k={k}']:.0f}%)"
            for k in (1, 2, 3, 4)
        )
        lines.append(f"| {p_rate}% | {cells} |")
    lines += [
        "",
        "80 queries per cell, "
        f"{a['trials']} in total, strictly sequential chain.",
        "",
        "## Deadline-bounded page enrichment (12 pages)",
        "",
        "| configuration | wall clock | pages kept |",
        "|---|---|---|",
    ]
    for label, v in e.items():
        lines.append(f"| {label} | {v['wall_ms']:.0f}ms | {v['pages_kept']}/12 |")
    lines += [
        "",
        "## Correctness contracts",
        "",
    ]
    for k, v in c.items():
        lines.append(f"- {'PASS' if v == 'PASS' else 'FAIL'} — {k}")
    lines += [
        "",
        "## Honest limits",
        "",
        "- Synthetic latencies: these numbers characterise our own machinery, not any provider.",
        "- A live SERP comparison would compare billing plans and rate limits, not quality.",
        f"- Healthy chain: {h['healthy']['p95_ratio']['x']:.2f}x P95 (thread launch is not free), and",
        f"  {h['primary empty, first backup dies']['p95_ratio']['x']:.2f}x the moment a layer degrades.",
        "",
    ]
    (REPO / "bench" / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("hedge vs serial ...", flush=True)
    hedge = bench_hedge()
    print("availability ...", flush=True)
    avail = bench_availability()
    print("enrichment ...", flush=True)
    enrich = bench_enrichment()
    print("contracts ...", flush=True)
    contracts = bench_contracts()

    payload = {
        "meta": {
            "latency_scale": SCALE,
            "hedge_ms_in_bench": int(2000 * SCALE),
            "trials_per_cell": TRIALS,
            "python": sys.version.split()[0],
            "network": False,
            "api_keys": False,
        },
        "hedge_vs_serial": hedge,
        "availability": avail,
        "enrichment": enrich,
        "contracts": contracts,
    }
    (RESULTS / "resilience.json").write_text(json.dumps(payload, indent=2))
    write_results_md(hedge, avail, enrich, contracts)
    plot_hedge(hedge)
    plot_availability(avail)
    plot_enrichment(enrich)

    print("\nhedged vs serial (P95, ms):")
    for label, modes in hedge.items():
        print(
            f"  {label:32s} serial {modes['serial']['p95']:7.1f}  "
            f"hedged {modes['hedged']['p95']:7.1f}  "
            f"{modes['p95_ratio']['x']:.2f}x"
        )
    print("\nenrichment:")
    for label, v in enrich.items():
        print(f"  {label:30s} {v['wall_ms']:7.1f}ms  {v['pages_kept']}/12 pages")
    print("\ncontracts:")
    for k, v in contracts.items():
        print(f"  {'OK' if v == 'PASS' else 'XX'}  {k}: {v}")
    print("\nwritten: bench/results/resilience.json, figures/*.png")


if __name__ == "__main__":
    main()
