# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Optional callback for web-search progress (e.g. SSE to browser). Per-thread via ContextVar."""
from __future__ import annotations

import contextvars
import os
import threading
from typing import Any, Callable, Optional

_web_progress_cb: contextvars.ContextVar[Optional[Callable[[Any], None]]] = contextvars.ContextVar(
    "finkey_web_progress_cb",
    default=None,
)


def set_web_progress_callback(cb: Optional[Callable[[Any], None]]) -> contextvars.Token:
    return _web_progress_cb.set(cb)


def reset_web_progress_callback(token: contextvars.Token) -> None:
    _web_progress_cb.reset(token)


def emit_web_progress(message: str) -> None:
    """Human-readable pipeline line → SSE `web_step`."""
    cb = _web_progress_cb.get()
    if cb and isinstance(message, str) and message.strip():
        cb({"type": "web_step", "message": message.strip()})


def emit_pipeline_stage(stage: str, detail: str = "", *, done: bool = False) -> None:
    """
    Live pre-generation pipeline stage → SSE `stage`.

    ``stage``: thinking | web_classify | web_search | web_read | memory | kb |
               routing | generating | ...
    Emitted the moment a stage starts (or finishes with ``done=True``) so the
    frontend can show real-time status text instead of a silent blob.
    """
    cb = _web_progress_cb.get()
    if not cb:
        return
    s = (stage or "").strip()
    if not s:
        return
    payload: dict = {"type": "stage", "stage": s, "done": bool(done)}
    d = (detail or "").strip()
    if d:
        payload["detail"] = d[:280]
    cb(payload)


def emit_sse_event(payload: dict) -> None:
    """Forward an arbitrary structured event into the active chat SSE stream."""
    cb = _web_progress_cb.get()
    if cb and isinstance(payload, dict) and payload.get("type"):
        cb(dict(payload))


def emit_browse_progress(
    *,
    phase: str,
    detail: str,
    url: str = "",
) -> None:
    """
    Structured live browser activity for the frontend mini-pane (`browse_frame`).
    ``phase``: open | fetch | playwright | operator | done | ...
    """
    cb = _web_progress_cb.get()
    if not cb:
        return
    d = (detail or "").strip()
    if not d:
        return
    cb(
        {
            "type": "browse_frame",
            "phase": (phase or "").strip() or "browse",
            "url": (url or "")[:2048],
            "message": d,
        }
    )


def _live_browser_enabled() -> bool:
    """Returns True unless FINKEY_LIVE_BROWSER=0 is explicitly set."""
    raw = (os.getenv("FINKEY_LIVE_BROWSER", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


_cdp_broadcast_fn: Optional[Callable[[str, str], None]] = None
_cdp_broadcast_lock = threading.Lock()


def set_cdp_broadcast(fn: Optional[Callable[[str, str], None]]) -> None:
    """Register the server-side broadcast function for CDP frames (called on startup)."""
    global _cdp_broadcast_fn
    with _cdp_broadcast_lock:
        _cdp_broadcast_fn = fn


def emit_ask_user(conv_id: str, question: str) -> None:
    """
    Emit a structured `ask_user` SSE event so the frontend shows an input panel.
    The operator is paused; the user's reply is sent back via WebSocket or HTTP.
    """
    if not question:
        return
    cb = _web_progress_cb.get()
    if not cb:
        return
    cb({"type": "ask_user", "conv_id": conv_id, "question": question})


def emit_operator_speak(message: str) -> None:
    """
    Inject the operator's status text directly into the user's streaming response.
    The frontend treats this as a `content` delta — the user sees the text appear
    inside the answer while the browser is still working.
    Use for queue waits, CAPTCHAs, login prompts, etc.
    """
    if not message:
        return
    cb = _web_progress_cb.get()
    if not cb:
        return
    cb({"type": "content", "delta": message})


def emit_cdp_frame(jpeg_b64: str, url: str = "") -> None:
    """
    Forward a CDP screencast JPEG frame to all connected WebSocket clients.
    Safe to call from any thread or async context.
    """
    if not jpeg_b64 or not _live_browser_enabled():
        return
    with _cdp_broadcast_lock:
        fn = _cdp_broadcast_fn
    if fn:
        try:
            fn(jpeg_b64, url)
        except Exception:
            pass


def emit_browser_screenshot(
    *,
    url: str = "",
    phase: str = "",
    image_b64: str,
) -> None:
    """
    Stream a live JPEG viewport screenshot from the Playwright browser to the frontend.
    Payload: { type: "browser_screenshot", url, phase, image: "data:image/jpeg;base64,..." }
    Disable with env FINKEY_LIVE_BROWSER=0.
    """
    if not image_b64 or not _live_browser_enabled():
        return
    cb = _web_progress_cb.get()
    if not cb:
        return
    cb(
        {
            "type": "browser_screenshot",
            "url": (url or "")[:2048],
            "phase": (phase or "").strip() or "browse",
            "image": image_b64,
        }
    )
