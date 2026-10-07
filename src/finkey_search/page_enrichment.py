# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Fetch main article text for top search hits — grounding beyond snippets.

Latency model (ChatGPT-like, quality-preserving):
  • SERP / max_pages budget is unchanged — we do not cut how many sources we *may* read.
  • A **global deadline** bounds wall-clock enrich time (fast ≈ 3.5s, deep ≈ 8s).
  • Pages are fetched in parallel; whoever answers in time is kept.
  • Stragglers past the deadline are abandoned (not awaited for 12s each).
  • ``answer_ready`` early-stop still wins: if evidence is enough mid-wave, stop.
  • If the deadline hits with *zero* enriched pages, a short grace window lets the
    in-flight fastest page finish — empty enrich is worse than +2s.
"""
from __future__ import annotations

import logging
import os
import re
import statistics
import contextvars
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Optional

from finkey_search.browser_config import BrowserConfig

try:
    from finkey_search._browser.stealth_browser import fetch_structured_sync
except ImportError:  # a stealth engine is optional; enrichment degrades to HTTP
    fetch_structured_sync = None
from finkey_search.url_policy import URLPolicy
from finkey_search.progress import emit_browser_screenshot, _live_browser_enabled
from finkey_search.http_text_fetch import fetch_page_text_http
from finkey_search.metrics import InternetMetrics
from finkey_search.schema import SearchDepth, SearchResult
from finkey_search.scrapling_fetch import fetch_page_text_scrapling

logger = logging.getLogger(__name__)

_NAV_ALLOWED = frozenset({"commit", "domcontentloaded", "load", "networkidle"})


def _page_enrich_browser_timeout_ms() -> int:
    """Playwright navigation timeout per attempt (goto + extract)."""
    try:
        v = int(os.getenv("FINKEY_PAGE_ENRICH_BROWSER_TIMEOUT_MS", "8000"))
    except ValueError:
        v = 8_000
    return max(3_000, min(v, 60_000))


def _page_enrich_nav_wait_until() -> str:
    """
    ``load`` часто зависает на тяжёлых SPA; для enrich по умолчанию раньше отдаём DOM.
    Override: FINKEY_PAGE_ENRICH_NAV_WAIT_UNTIL=load|domcontentloaded|commit|networkidle
    """
    raw = (os.getenv("FINKEY_PAGE_ENRICH_NAV_WAIT_UNTIL", "domcontentloaded") or "").strip().lower()
    return raw if raw in _NAV_ALLOWED else "domcontentloaded"


def _page_enrich_http_timeout_s(browser_timeout_ms: int) -> float:
    """HTTPS fallback: its own budget, or a small margin over the browser one."""
    raw = (os.getenv("FINKEY_PAGE_ENRICH_HTTP_TIMEOUT_S") or "").strip()
    if raw:
        try:
            return max(2.0, min(float(raw), 40.0))
        except ValueError:
            pass
    bt = browser_timeout_ms / 1000.0
    # Cap fallback so a single Scrapling/HTTP retry cannot dominate the deadline.
    return max(2.0, min(bt + 1.5, 12.0))


def _http_fallback_enabled() -> bool:
    return os.getenv("FINKEY_PAGE_ENRICH_HTTP_FALLBACK", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _page_enrich_concurrency() -> int:
    try:
        v = int(os.getenv("FINKEY_PAGE_ENRICH_CONCURRENCY", "7"))
    except ValueError:
        v = 7
    return max(1, min(v, 8))


def _adaptive_enrich_hard_cap() -> int:
    """Upper bound for ``adaptive_enrich_budget`` on top of baseline ``FINKEY_WEB_PAGE_ENRICH_MAX``."""
    try:
        v = int(os.getenv("FINKEY_WEB_PAGE_ENRICH_ADAPTIVE_MAX", "14"))
    except ValueError:
        v = 14
    return max(6, min(v, 32))


def _enrich_wave_size(*, fast: bool) -> int:
    """
    Pages per enrich wave before answer_ready check.

    FAST defaults to 4 so the first parallel burst usually covers a simple fact
    question; DEEP stays at 2 so conflict detection can open a second wave.
    """
    env_key = "FINKEY_WEB_ENRICH_WAVE_SIZE_FAST" if fast else "FINKEY_WEB_ENRICH_WAVE_SIZE"
    default = "4" if fast else "2"
    # Backward-compatible: plain FINKEY_WEB_ENRICH_WAVE_SIZE still overrides both
    # when the mode-specific key is unset and the shared key is set.
    raw = (os.getenv(env_key) or os.getenv("FINKEY_WEB_ENRICH_WAVE_SIZE") or default).strip()
    try:
        v = int(raw)
    except ValueError:
        v = 4 if fast else 2
    return max(1, min(v, 8))


def _wave_early_stop_enabled() -> bool:
    raw = (os.getenv("FINKEY_WEB_ENRICH_EARLY_STOP", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _enrich_deadline_s(*, fast: bool, browser_mode: bool) -> float:
    """
    Wall-clock budget for the whole enrich pass.

    Quality stays in max_pages + answer_ready; this only caps how long we wait
    for slow hosts. Disable with FINKEY_WEB_ENRICH_DEADLINE_S=0.
    """
    # Explicit global override (tests / emergency).
    raw_all = (os.getenv("FINKEY_WEB_ENRICH_DEADLINE_S") or "").strip()
    if raw_all:
        try:
            v = float(raw_all)
            return 0.0 if v <= 0 else max(0.5, min(v, 60.0))
        except ValueError:
            pass

    if browser_mode:
        key, default = "FINKEY_WEB_ENRICH_DEADLINE_BROWSER_S", "15"
    elif fast:
        key, default = "FINKEY_WEB_ENRICH_DEADLINE_FAST_S", "3.5"
    else:
        key, default = "FINKEY_WEB_ENRICH_DEADLINE_DEEP_S", "8"
    try:
        v = float((os.getenv(key) or default).strip())
    except ValueError:
        v = float(default)
    if v <= 0:
        return 0.0
    return max(0.8, min(v, 60.0))


def _enrich_zero_grace_s() -> float:
    """Extra wait when the deadline hits with zero enriched pages."""
    try:
        v = float((os.getenv("FINKEY_WEB_ENRICH_ZERO_GRACE_S") or "2").strip())
    except ValueError:
        v = 2.0
    return max(0.0, min(v, 8.0))


def adaptive_enrich_budget(results: list[SearchResult], base: int) -> int:
    """
    Raise fetch budget when snippets are thin or numeric hints disagree across hits.
    """
    if base <= 0 or not results:
        return max(0, base)

    snippets = [(r.snippet or "").strip() for r in results[:5]]
    lens = [len(s) for s in snippets if s]
    budget = base
    if lens:
        med = float(statistics.median(lens))
        if med < 95:
            budget += 2

    floats: list[float] = []
    for r in results[:3]:
        for m in re.findall(r"\d[\d\.,]*", r.snippet or ""):
            try:
                floats.append(float(m.replace(",", ".")))
            except ValueError:
                pass
        if len(floats) > 24:
            break

    if len(floats) >= 4:
        mean = statistics.mean(floats)
        if mean > 1e-9 and (max(floats) - min(floats)) / mean > 0.035:
            budget += 1

    return min(max(budget, 0), _adaptive_enrich_hard_cap())


def _norm_url_key(url: str) -> str:
    u = (url or "").strip().split("#")[0].rstrip("/").lower()
    if u.startswith("https://www."):
        u = "https://" + u[len("https://www.") :]
    elif u.startswith("http://www."):
        u = "http://" + u[len("http://www.") :]
    return u


def enrich_results_with_structured_pages(
    results: list[SearchResult],
    *,
    max_pages: int,
    url_policy: URLPolicy,
    max_chars_per_page: int = 65536,
    browser_timeout_ms: Optional[int] = None,
    metrics: Optional[InternetMetrics] = None,
    http_fallback: Optional[bool] = None,
    fetch_mode: str = "browser",
    message: str = "",
    depth: Optional[SearchDepth] = None,
    early_stop: Optional[bool] = None,
    deadline_s: Optional[float] = None,
    enrich_url_allowlist: Optional[list[str]] = None,
) -> None:
    """
    Mutates ``results`` in place: sets ``enriched_text`` when structured fetch succeeds.
    Respects ``URLPolicy`` (allowlist / blocked hosts / blocked path fragments).

    ``http_fallback``: when a Playwright structured fetch misses, optionally retry via
    Scrapling/HTTP. ``None`` uses the env default (``FINKEY_PAGE_ENRICH_HTTP_FALLBACK``).
    The fast search path passes ``False`` to skip the slow retry tail (the biggest latency
    source on dead/anti-bot hosts) and just move on.

    ``fetch_mode``:
      • ``"browser"`` (default) — Playwright structured fetch first, HTTP tail as fallback.
      • ``"http"`` — no Playwright at all: Scrapling (TLS-spoof) → plain HTTPS. Seconds
        instead of tens of seconds; used by the FAST search path.

    Wave early-stop (ChatGPT-like): when ``message`` is set and early-stop is on, fetch
    in waves and stop when ``answer_ready`` says evidence is enough. ``max_pages`` remains
    the ceiling — we do not permanently cut the budget.

    ``deadline_s``: optional wall-clock enrich budget. ``None`` → env defaults by depth.
    ``0`` disables the deadline (legacy wait-for-every-page behaviour).
    """
    if max_pages <= 0 or not results:
        return

    http_only = (fetch_mode or "browser").strip().lower() == "http" or fetch_structured_sync is None
    fb_enabled = _http_fallback_enabled() if http_fallback is None else bool(http_fallback)
    bt_ms = browser_timeout_ms if browser_timeout_ms is not None else _page_enrich_browser_timeout_ms()
    nav_wait = _page_enrich_nav_wait_until()
    base_http_timeout = (bt_ms / 1000.0) if http_only else _page_enrich_http_timeout_s(bt_ms)

    depth_eff = depth if isinstance(depth, SearchDepth) else SearchDepth.DEEP
    fast = depth_eff == SearchDepth.FAST
    if deadline_s is None:
        deadline_s = _enrich_deadline_s(fast=fast, browser_mode=not http_only)
    deadline_mono = (
        time.monotonic() + float(deadline_s) if deadline_s and deadline_s > 0 else None
    )

    live_browser = _live_browser_enabled()

    _force_headless = os.getenv("FINKEY_PAGE_ENRICH_HEADLESS", "").strip().lower()
    if _force_headless in ("1", "true", "yes"):
        _use_headless = True
    elif _force_headless in ("0", "false", "no"):
        _use_headless = False
    else:
        _use_headless = True

    cfg = None
    if not http_only:
        cfg = BrowserConfig(
            headless=_use_headless,
            timeout_ms=bt_ms,
            navigation_wait_until=nav_wait,
            wait_after_load_ms=200 if fast else 400,
            scroll_after_load=False,
            take_screenshot=live_browser,
            max_retries=0,
        )

    allow: Optional[set[str]] = None
    if enrich_url_allowlist:
        allow = {_norm_url_key(u) for u in enrich_url_allowlist if (u or "").strip()}
        allow.discard("")

    eligible: list[tuple[int, SearchResult, str]] = []
    for idx, r in enumerate(results):
        if (r.enriched_text or "").strip():
            continue
        url = (r.url or "").strip()
        if not url.startswith("https://"):
            continue
        if allow is not None and _norm_url_key(url) not in allow:
            continue
        try:
            from .url_quality import is_blocked_for_operator_browse

            if is_blocked_for_operator_browse(url):
                logger.debug("Page enrich skip blocked host: %s", url[:96])
                continue
        except Exception:
            pass
        ok, reason = url_policy.validate(url)
        if not ok:
            logger.debug("Page enrich skip URL policy: %s (%s)", url[:96], reason)
            continue
        eligible.append((idx, r, url))

    if not eligible:
        return

    use_waves = bool((message or "").strip()) and (
        _wave_early_stop_enabled() if early_stop is None else bool(early_stop)
    )
    wave_size = _enrich_wave_size(fast=fast) if use_waves else max(max_pages, len(eligible))

    def _remaining_s() -> Optional[float]:
        if deadline_mono is None:
            return None
        return deadline_mono - time.monotonic()

    def _page_timeout_s() -> float:
        """Per-attempt timeout capped by whatever is left on the global deadline."""
        rem = _remaining_s()
        base = base_http_timeout
        if rem is None:
            return base
        if rem <= 0.05:
            return 0.05
        # Leave a tiny slice so the worker returns before the wave waiter gives up.
        return max(0.2, min(base, rem - 0.05))

    def _run_one_http(
        item: tuple[int, SearchResult, str],
        *,
        metrics_cb: Optional[InternetMetrics] = None,
        timeout_s: float,
    ) -> tuple[int, str, bool, bool]:
        """FAST path: Scrapling → HTTPS, no Playwright. Returns like _run_one."""
        idx, _r, url = item
        if metrics_cb:
            metrics_cb.record_page_enrich_attempt()
        cap = max_chars_per_page if max_chars_per_page > 0 else 12_000
        try:
            stext, _serr = fetch_page_text_scrapling(url, timeout_s=timeout_s, max_chars=cap)
            if stext.strip():
                return idx, stext.strip(), False, True
            htext, _herr = fetch_page_text_http(url, timeout_s=timeout_s, max_chars=cap)
            if htext.strip():
                return idx, htext.strip(), False, True
        except Exception as exc:
            logger.info("Page enrich (http mode) exception url=%s: %s", url[:96], exc)
        return idx, "", True, False

    def _run_one(
        item: tuple[int, SearchResult, str],
        *,
        metrics_cb: Optional[InternetMetrics] = None,
        timeout_s: Optional[float] = None,
    ) -> tuple[int, str, bool, bool]:
        """Returns (idx, text_or_empty, had_structured_miss, used_http_fallback)."""
        t_out = timeout_s if timeout_s is not None else _page_timeout_s()
        if http_only:
            return _run_one_http(item, metrics_cb=metrics_cb, timeout_s=t_out)
        idx, _r, url = item
        if metrics_cb:
            metrics_cb.record_page_enrich_attempt()
        had_miss = False
        used_http_fb = False
        # Shrink Playwright budget to the remaining deadline when possible.
        local_cfg = cfg
        if deadline_mono is not None:
            rem_ms = int(max(0.2, min(bt_ms / 1000.0, t_out)) * 1000)
            if rem_ms < cfg.timeout_ms:
                local_cfg = BrowserConfig(
                    headless=cfg.headless,
                    timeout_ms=rem_ms,
                    navigation_wait_until=cfg.navigation_wait_until,
                    wait_after_load_ms=cfg.wait_after_load_ms,
                    scroll_after_load=cfg.scroll_after_load,
                    take_screenshot=cfg.take_screenshot,
                    max_retries=0,
                )
        try:
            sp = fetch_structured_sync(url, local_cfg)
            structured_ok = bool(sp.ok and (sp.main_content or "").strip())
            if structured_ok:
                text = sp.main_content.strip()
                if max_chars_per_page > 0:
                    text = text[:max_chars_per_page]
                if sp.screenshot:
                    try:
                        import base64 as _b64
                        _img = "data:image/jpeg;base64," + _b64.b64encode(sp.screenshot).decode()
                        emit_browser_screenshot(url=url, phase="fetch", image_b64=_img)
                    except Exception:
                        pass
                return idx, text, False, False
            had_miss = True
            err_h = getattr(sp, "error", None) or ""
            status = getattr(sp, "status", None)
            logger.info(
                "Page enrich structured miss: url=%s status=%s err=%s",
                url[:120],
                status,
                (err_h or "empty_body")[:200],
            )
            if fb_enabled:
                cap = max_chars_per_page if max_chars_per_page > 0 else 12_000
                fb_timeout = _page_timeout_s()

                stext, serr = fetch_page_text_scrapling(
                    url,
                    timeout_s=fb_timeout,
                    max_chars=cap,
                )
                if stext.strip():
                    logger.info(
                        "Page enrich Scrapling fallback ok: url=%s chars=%d",
                        url[:120],
                        len(stext),
                    )
                    return idx, stext.strip(), had_miss, True

                logger.info(
                    "Page enrich Scrapling fallback failed: url=%s err=%s",
                    url[:120],
                    serr,
                )

                htext, herr = fetch_page_text_http(
                    url,
                    timeout_s=fb_timeout,
                    max_chars=cap,
                )
                if htext.strip():
                    logger.info(
                        "Page enrich HTTP fallback ok: url=%s chars=%d",
                        url[:120],
                        len(htext),
                    )
                    return idx, htext.strip(), had_miss, True
                logger.info(
                    "Page enrich HTTP fallback failed: url=%s err=%s",
                    url[:120],
                    herr,
                )
        except Exception as exc:
            logger.info("Page enrich exception url=%s: %s", url[:96], exc)
        return idx, "", had_miss, used_http_fb

    def _apply_outcome(
        idx: int,
        text: str,
        miss: bool,
        http_fb: bool,
        *,
        url: str,
    ) -> bool:
        """Apply one successful fetch onto results[idx]. Returns True if enriched."""
        if not text.strip():
            return False
        r = results[idx]
        r.enriched_text = text
        if metrics:
            if miss:
                metrics.record_page_enrich_structured_miss()
            if http_fb:
                metrics.record_page_enrich_http_fallback_ok()
            metrics.record_page_enrich_success()
        logger.debug("Page enrich ok url=%s chars=%d", url[:96], len(text))
        return True

    def _evidence_ready() -> bool:
        if not use_waves:
            return False
        try:
            from finkey_search.evidence_gate import answer_ready

            ready = answer_ready(message, results, depth=depth_eff)
            if ready.ready:
                logger.info(
                    "Page enrich early-stop: reason=%s enriched=%d hosts=%d depth=%s",
                    ready.reason,
                    ready.enriched_rows,
                    ready.unique_hosts,
                    ready.effective_depth,
                )
                return True
        except Exception as exc:
            logger.debug("answer_ready check failed: %s", exc)
        return False

    def _run_wave(wave: list[tuple[int, SearchResult, str]]) -> int:
        """
        Fetch a wave under the global deadline.

        Evidence early-stop runs *after* the wave (not mid-flight): a single
        fast page with a number must not abort the wave before a conflicting
        sibling can land — that is how we keep FX/price quality.
        """
        if not wave:
            return 0
        rem = _remaining_s()
        if rem is not None and rem <= 0.02:
            logger.info(
                "Page enrich wave skipped: deadline exhausted (%d urls left untouched)",
                len(wave),
            )
            return 0

        page_t = _page_timeout_s()
        got = 0
        conc = _page_enrich_concurrency()

        # Batch fast path (crawl4ai, opt-in via FINKEY_INTERNET_BATCH_BACKEND):
        # one async pass over the whole wave beats per-URL Playwright; anything
        # that misses falls through to the normal wave untouched.
        if not http_only and len(wave) > 1:
            from finkey_search.batch_fetch import batch_backend, fetch_pages_batch

            if batch_backend() == "crawl4ai":
                urls = [item[2] for item in wave]
                bout = fetch_pages_batch(
                    urls,
                    timeout_s=min(45.0, max(10.0, page_t * float(len(wave)))),
                    max_concurrency=conc,
                )
                if bout.get("ok"):
                    by_url: dict[str, str] = {}
                    for res in bout.get("results") or []:
                        md = str(res.get("markdown") or "").strip()
                        if res.get("success") and md and res.get("url"):
                            by_url.setdefault(str(res["url"]), md)
                    remaining = []
                    for item in wave:
                        md = by_url.get(item[2])
                        if md and _apply_outcome(item[0], md, False, True, url=item[2]):
                            got += 1
                        else:
                            remaining.append(item)
                    wave = remaining
                    if not wave:
                        return got

        if conc <= 1 or len(wave) <= 1:
            for item in wave:
                if rem is not None and time.monotonic() >= (deadline_mono or 0):
                    break
                idx, text, miss, http_fb = _run_one(
                    item, metrics_cb=metrics, timeout_s=_page_timeout_s()
                )
                if _apply_outcome(idx, text, miss, http_fb, url=item[2]):
                    got += 1
            return got

        workers = min(conc, len(wave))
        logger.info(
            "Page enrich wave: parallel workers=%d jobs=%d page_timeout=%.2fs deadline_left=%s",
            workers,
            len(wave),
            page_t,
            f"{rem:.2f}s" if rem is not None else "off",
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    contextvars.copy_context().run,
                    _run_one,
                    item,
                    metrics_cb=None,
                    timeout_s=page_t,
                ): item
                for item in wave
            }
            pending = set(futs)
            while pending:
                rem_now = _remaining_s()
                if rem_now is not None and rem_now <= 0:
                    logger.info(
                        "Page enrich deadline: abandoning %d in-flight page(s)",
                        len(pending),
                    )
                    for fut in pending:
                        fut.cancel()
                    break
                wait_s = 0.25 if rem_now is None else max(0.05, min(0.25, rem_now))
                done, pending = wait(pending, timeout=wait_s, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for fut in done:
                    item = futs[fut]
                    try:
                        idx, text, miss, http_fb = fut.result()
                        if metrics:
                            metrics.record_page_enrich_attempt()
                        if _apply_outcome(idx, text, miss, http_fb, url=item[2]):
                            got += 1
                    except Exception as exc:
                        logger.info("Page enrich worker failed url=%s: %s", item[2][:96], exc)
                        if metrics:
                            metrics.record_page_enrich_attempt()
        return got

    already = sum(1 for r in results if (r.enriched_text or "").strip())
    enriched_total = already
    cursor = 0
    wave_i = 0
    started = time.monotonic()

    while cursor < len(eligible) and enriched_total < max_pages:
        rem = _remaining_s()
        if rem is not None and rem <= 0.02:
            break
        remaining_budget = max_pages - enriched_total
        take = min(wave_size, remaining_budget, len(eligible) - cursor)
        if take <= 0:
            break
        wave = eligible[cursor : cursor + take]
        cursor += take
        wave_i += 1
        got = _run_wave(wave)
        enriched_total = sum(1 for r in results if (r.enriched_text or "").strip())
        logger.info(
            "Page enrich wave %d: fetched=%d got=%d enriched_total=%d/%d waves=%s elapsed=%.2fs",
            wave_i,
            len(wave),
            got,
            enriched_total,
            max_pages,
            use_waves,
            time.monotonic() - started,
        )
        if not use_waves:
            break
        if enriched_total >= max_pages:
            break
        if _evidence_ready():
            break

    # Grace: deadline hit with nothing enriched — try one more URL briefly so a
    # single slow-but-alive host can still land (empty enrich is worse than +2s).
    enriched_total = sum(1 for r in results if (r.enriched_text or "").strip())
    grace = _enrich_zero_grace_s()
    if (
        enriched_total == already
        and grace > 0
        and cursor < len(eligible)
        and enriched_total < max_pages
    ):
        logger.info(
            "Page enrich zero-hit grace: trying 1 more URL for %.1fs",
            grace,
        )
        grace_deadline = time.monotonic() + grace
        if deadline_mono is None or grace_deadline > deadline_mono:
            deadline_mono = grace_deadline
        wave = eligible[cursor : cursor + 1]
        cursor += 1
        _run_wave(wave)

    elapsed = time.monotonic() - started
    final_n = sum(1 for r in results if (r.enriched_text or "").strip())
    logger.info(
        "Page enrich done: enriched=%d/%d elapsed=%.2fs deadline=%s fast=%s http_only=%s",
        final_n,
        max_pages,
        elapsed,
        f"{deadline_s}s" if deadline_s else "off",
        fast,
        http_only,
    )
