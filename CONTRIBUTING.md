# Contributing to FinKeyWeb

Thanks for your interest!

## Contributor licence terms

By opening a pull request you agree that copyright of your contribution is
**assigned to FinKey**, and that FinKey may relicense it (including commercially)
while keeping the community edition under AGPL-3.0. This is what lets the project
stay open *and* fundable. You keep the right to use your contribution elsewhere.

If you cannot agree to assignment, open an issue instead of a PR — we'll work it out.

## Ground rules

- Python 3.10+, **zero new required dependencies**. HTTP clients, Redis and
  Scrapling stay optional and imported lazily inside the code paths that need them.
- Graceful degradation is the contract: a missing provider key, rate limit, dead
  host, or absent optional dependency must disable exactly one feature — never
  raise into the caller. A no-keys `WebSearchEngine().run(...)` must return cleanly.
- New backends implement the `SearchBackend` protocol and are wrapped in
  `GuardedBackend` (rate limiter + circuit breaker) like the existing four.
- Every new behaviour ships with a pure-Python test (no network) in `tests/`.

## Running the suite

```bash
pip install -e ".[test]"
pytest
```
