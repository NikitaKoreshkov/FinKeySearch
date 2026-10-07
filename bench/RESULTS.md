# Measured results

Regenerate: `PYTHONPATH=src python3 bench/search_bench.py`
No network, no API keys. Latencies are production values scaled by 0.1 (a 2000ms hedge is 200ms here); ratios are the finding.

## Hedged fallback chain vs strictly sequential (P95, ms)

| scenario | serial | hedged | ratio |
|---|---|---|---|
| healthy | 44 | 45 | 0.96x |
| primary retry storm | 293 | 293 | 1.00x |
| primary empty, first backup dies | 645 | 213 | 3.03x |
| primary hard failure | 654 | 576 | 1.13x |

Parity where the preferred provider is merely slow is intentional: the chain never
trades the preferred provider's answer for a faster one from a backup.

## Availability vs per-provider error rate

| failure rate | 1 provider | 2 | 3 | 4 |
|---|---|---|---|---|
| 10% | 91% (theory 90%) | 100% (theory 99%) | 100% (theory 100%) | 100% (theory 100%) |
| 20% | 80% (theory 80%) | 99% (theory 96%) | 99% (theory 99%) | 100% (theory 100%) |
| 30% | 71% (theory 70%) | 96% (theory 91%) | 98% (theory 97%) | 99% (theory 99%) |
| 40% | 59% (theory 60%) | 88% (theory 84%) | 91% (theory 94%) | 98% (theory 97%) |

80 queries per cell, 1280 in total, strictly sequential chain.

## Deadline-bounded page enrichment (12 pages)

| configuration | wall clock | pages kept |
|---|---|---|
| one page at a time | 1942ms | 12/12 |
| parallel wave, no deadline | 805ms | 12/12 |
| parallel wave, 0.35s deadline | 371ms | 10/12 |
| one page at a time, hostile straggler | 7053ms | 12/12 |
| deadline cap, hostile straggler | 396ms | 10/12 |

## Correctness contracts

- PASS — breaker opens after 3 failures
- PASS — breaker half-opens after the window
- PASS — breaker closes on a success
- PASS — limiter admits exactly the quota
- PASS — limiter window slides
- PASS — hedge=0 never overlaps backends
- PASS — hedge overlaps backups with a slow primary
- PASS — hedge keeps the preferred provider's answer

## Honest limits

- Synthetic latencies: these numbers characterise our own machinery, not any provider.
- A live SERP comparison would compare billing plans and rate limits, not quality.
- Healthy chain: 0.96x P95 (thread launch is not free), and
  3.03x the moment a layer degrades.
