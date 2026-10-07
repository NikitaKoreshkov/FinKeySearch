<div align="center">

# FinKeySearch

**The web-search layer for AI agents that answers *before* the model does.**
Hedged multi-provider fallback, an LLM-first "do I even need to search" gate,
deadline-bounded parallel page enrichment, an evidence critic that rewrites the
query when the first pass is thin, and a deep-research fan-out — with
**zero required dependencies**.

`pip install finkey-search`

![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Deps](https://img.shields.io/badge/required%20deps-zero-brightgreen)
![Providers](https://img.shields.io/badge/providers-Serper%20%C2%B7%20Brave%20%C2%B7%20Bing%20%C2%B7%20Tavily-yellow)

</div>

---

## The problem

"Give the LLM a search tool and a prompt" breaks in production three ways:

1. **Latency cliffs.** One provider rate-limits or hangs and the whole turn stalls.
2. **Snippet-only answers.** SERP snippets are headlines; the model confidently
   quotes a number that lived one click deeper, on a page you never fetched.
3. **Over-searching & under-searching.** Regex "when to search" lists fire on
   "hi" and miss "why did it crash today".

FinKeySearch is the retrieval layer we run inside the FinKey platform, extracted
standalone. It decides *whether* to search, races providers so a slow one can't
win, reads the actual pages under a hard wall-clock budget, and critiques its own
evidence before handing it to the model.

## What makes it different

- **Hedged fallback chain.** Serper starts immediately; if it hasn't answered
  within a grace window, Brave/Bing/Tavily launch **in parallel** and the best
  non-empty result wins. Serial degradation ("6 s + 3 s + …") becomes one race
  at ~max(), not sum(). `FINKEY_SEARCH_HEDGE_MS` tunes it; set `0` for strict serial.
- **Per-provider circuit breakers + sliding-window rate limits.** A provider
  throwing 429s is skipped until it recovers — you never pay for a dead backend.
- **LLM-first classifier, not regex.** `llm_primary`/`llm_only` modes make a tiny
  model decide search-need, category and depth (FAST/DEEP); only trivial chat
  ("hi", "thanks", `2+2`) is short-circuited heuristically. No brittle keyword lists.
- **Deadline-bounded parallel enrichment.** Top hits are fetched concurrently
  under a global wall-clock budget (fast ≈ 3.5 s, deep ≈ 8 s); stragglers are
  abandoned, whoever answered in time is kept. `answer_ready` early-stop wins
  mid-wave. A hung host cannot burn the whole turn.
- **Evidence gate + gap planner.** After retrieval a critic asks "what's missing?"
  and, if the evidence is thin, fans out 3–5 new targeted queries — then merges
  with provenance.
- **Deep-research orchestrator.** A coordinator splits a hard question into
  parallel sub-agents (generic research lenses, not topic allowlists), runs them,
  and dedupes results for the answer model.
- **Graceful by construction.** No keys → the chain is empty and nothing crashes.
  No `requests`/`scrapling`/`redis` → those optional paths are skipped. No
  Playwright browser backend → enrichment falls back to HTTP. Every missing
  dependency disables exactly one feature.
- **Grounding & citation helpers.** snippet sanitization, numeric critic,
  claim/citation verification and a GitHub-repo fact fetcher ship in the box.

## Quickstart

```bash
pip install "finkey-search[http]"
export SERPER_API_KEY=...        # optional: add BRAVE_API_KEY / BING / TAVILY to hedge
```

```python
from finkey_search import WebSearchEngine

engine = WebSearchEngine()
ctx = engine.run(
    message="What is the current USD/EUR rate?",
    generate_fn=my_llm_callable,   # used by the classifier + synthesis
    conversation_key="sess-1",
)
if ctx.found:
    print(ctx.synthesized)         # dated knowledge block for the system prompt
    print(ctx.citation_urls)
```

With no keys at all the call still returns cleanly (`ctx.found == False`) — the
engine never raises because a provider is missing.

## Architecture

```
 message
   │
   ├─ SearchNeedClassifier (LLM-first) ── skip? ─▶ (no web)
   │        FAST | DEEP, category
   ├─ FallbackSearchChain ── Serper ∥(hedge)∥ Brave ∥ Bing ∥ Tavily
   │        per-provider: rate-limiter + circuit-breaker
   ├─ SearchReranker (heuristic + optional LLM trim)
   ├─ page_enrichment ── parallel fetch under one wall-clock deadline
   │        Playwright (optional) → Scrapling → plain HTTPS
   ├─ evidence_gate ── critic: enough? if not → gap-planner → 2nd SERP wave
   ├─ research_orchestrator (DEEP) ── coordinator → parallel sub-agents → merge
   └─ ResultSynthesizer ─▶ dated "LIVE WEB DATA" block + citation urls
```

## Native tool schema

Ships an OpenAI-format `web_search` tool (`tools_schema.py`) with `depth`
(fast/deep), `subqueries`, `urls` (allowlist for a second deep pass) and
`mode=verify` (fact-check a draft answer against prior evidence) — drop it
straight into a tool-calling loop.

## Tests

```bash
pip install -e ".[test]"
pytest
```

A pure-Python suite (no network, no keys): classifier short-circuits, hedged vs
serial fallback with fake backends, circuit breaker + rate limiter, URL policy,
snippet sanitization, enrichment HTTP-degradation, evidence-gate planning.

## What this is / isn't

This is the **search + retrieval** layer. The headless **browsing operator**
(stealth, fingerprint, multi-step page interaction) is a separate, non-included
FinKey component; `WebSearchEngine` detects its absence and runs search-only.
It is **not** a document-RAG index — pair it with a retriever of your choice.

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Contributions assign
copyright to FinKey (keeps the community edition AGPL *and* lets us fund it).

## License

AGPL-3.0-only. Use it, fork it, serve it — but if you modify it and offer it over
a network, your changes go back to the community. Part of the FinKey agent
platform family — see also
[FinKeyMemory](https://github.com/NikitaKoreshkov/FinKeyMemory) and
[FinKeyEvals](https://github.com/NikitaKoreshkov/FinKeyEvals).
