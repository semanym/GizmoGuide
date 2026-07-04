"""Tracing for GizmoGuide with a local fallback.

Three backends, picked once at first use:

* ``off``      – ``DISABLE_TRACING`` is truthy → every helper is a no-op.
* ``langfuse`` – Langfuse keys are set *and* the Langfuse host is reachable →
  observations are reported to Langfuse (v4 SDK, OpenTelemetry based).
* ``file``     – Langfuse is unavailable → the same span/generation tree is
  written to a local JSONL file (one line per root request). This is the
  fallback so tracing still works (and stops spamming an unreachable Langfuse)
  when the Langfuse stack isn't running.

The public API is unchanged:

    from app.tracing import trace_request, trace_span, trace_generation

    with trace_request(session_id, user_id) as obs:
        with trace_span("step_name", input_data=...) as (span, end_span):
            ...
            end_span(output=...)

        with trace_generation("model_call", model="deepseek-chat", input_data=[...]) as (gen, end_gen):
            ...
            end_gen(output="...", usage_details={...})
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend selection (lazy, once)
# ---------------------------------------------------------------------------

_TRACE_OFF = "off"
_TRACE_LANGFUSE = "langfuse"
_TRACE_FILE = "file"

_mode: Optional[str] = None
_client: Any = None
_trace_path: Optional[Path] = None
_write_lock = threading.Lock()
_initialised = False

# Current span for the file backend; enables parent/child nesting across the
# `with` blocks (Langfuse does this itself via OTel context).
_current_span: contextvars.ContextVar[Optional["_LocalSpan"]] = contextvars.ContextVar(
    "gizmoguide_current_span", default=None
)


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _host_reachable(url: str, timeout: float = 0.3) -> bool:
    """Quick TCP probe so we don't hand spans to an unreachable Langfuse."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _init() -> None:
    global _mode, _client, _trace_path, _initialised
    if _initialised:
        return
    _initialised = True

    if _truthy(os.getenv("DISABLE_TRACING")):
        _mode = _TRACE_OFF
        logger.info("Tracing disabled (DISABLE_TRACING set)")
        return

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if public_key and secret_key:
        if _host_reachable(host):
            try:
                from langfuse import Langfuse

                _client = Langfuse()
                _mode = _TRACE_LANGFUSE
                logger.info("Tracing backend: Langfuse (%s)", host)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse init failed, using local file tracing: %s", exc)
        else:
            logger.warning(
                "Langfuse host %s unreachable, using local file tracing instead", host
            )

    _mode = _TRACE_FILE
    _trace_path = Path(os.getenv("TRACE_FILE_PATH", "log/trace.jsonl"))
    try:
        _trace_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Cannot create trace dir %s, tracing off: %s", _trace_path.parent, exc)
        _mode = _TRACE_OFF
        return
    logger.info("Tracing backend: local file (%s)", _trace_path)


def _get_client() -> Any:
    """Return the Langfuse client if that backend is active, else None."""
    _init()
    return _client if _mode == _TRACE_LANGFUSE else None


# ---------------------------------------------------------------------------
# Local file backend
# ---------------------------------------------------------------------------


class _LocalSpan:
    """A minimal observation node for the file backend."""

    def __init__(
        self,
        name: str,
        as_type: str,
        input_data: Any = None,
        metadata: Optional[dict] = None,
        model: Optional[str] = None,
    ):
        self.name = name
        self.as_type = as_type
        self.input = input_data
        self.metadata = dict(metadata) if metadata else None
        self.model = model
        self.output: Any = None
        self.level = "DEFAULT"
        self.status_message: Optional[str] = None
        self.usage: Optional[dict] = None
        self.start = time.time()
        self.end: Optional[float] = None
        self.children: list["_LocalSpan"] = []

    def update(
        self,
        output: Any = None,
        metadata: Optional[dict] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
        usage_details: Optional[dict] = None,
        **_ignored: Any,
    ) -> None:
        """Mirror the subset of Langfuse's ``update`` the callers rely on."""
        if output is not None:
            self.output = output
        if metadata:
            self.metadata = {**(self.metadata or {}), **metadata}
        if level is not None:
            self.level = level
        if status_message is not None:
            self.status_message = status_message
        if usage_details:
            self.usage = usage_details

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.as_type,
            "duration_ms": round((self.end or time.time()) - self.start, 4) * 1000
            if self.end
            else None,
            "level": self.level,
        }
        if self.model:
            data["model"] = self.model
        if self.input is not None:
            data["input"] = self.input
        if self.output is not None:
            data["output"] = self.output
        if self.metadata:
            data["metadata"] = self.metadata
        if self.usage:
            data["usage"] = self.usage
        if self.status_message:
            data["status_message"] = self.status_message
        if self.children:
            data["children"] = [c.to_dict() for c in self.children]
        return data


def _write_trace(root: _LocalSpan) -> None:
    if _trace_path is None:
        return
    line = json.dumps(
        {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "trace": root.to_dict()},
        ensure_ascii=False,
        default=str,
    )
    try:
        with _write_lock:
            with open(_trace_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:  # noqa: BLE001
        logger.debug("Failed to write trace to %s", _trace_path, exc_info=True)


@contextmanager
def _local_observation(
    name: str,
    as_type: str,
    input_data: Any = None,
    metadata: Optional[dict] = None,
    model: Optional[str] = None,
):
    """Open a local span, wire it under the current parent, write on root exit."""
    span = _LocalSpan(name, as_type, input_data, metadata, model)
    parent = _current_span.get()
    if parent is not None:
        parent.children.append(span)
    token = _current_span.set(span)
    try:
        yield span
    except Exception as exc:  # noqa: BLE001
        span.level = "ERROR"
        span.status_message = str(exc)
        raise
    finally:
        span.end = time.time()
        _current_span.reset(token)
        if parent is None:
            _write_trace(span)


# ---------------------------------------------------------------------------
# Top-level trace lifecycle
# ---------------------------------------------------------------------------


@contextmanager
def trace_request(
    session_id: str,
    user_id: Optional[str] = None,
    input_data: Optional[Any] = None,
    metadata: Optional[dict] = None,
):
    """Create a root observation for a single API request.

    Yields the observation object (or *None* when tracing is off).
    """
    _init()
    if _mode == _TRACE_OFF:
        yield None
        return

    meta = {"session_id": session_id, "user_id": user_id or session_id}
    if metadata:
        meta.update(metadata)

    if _mode == _TRACE_FILE:
        with _local_observation("gizmoguide_request", "span", input_data, meta) as span:
            yield span
        return

    client = _client
    with client.start_as_current_observation(
        name="gizmoguide_request",
        as_type="span",
        input=input_data,
        metadata=meta,
    ) as obs:
        yield obs

    try:
        client.flush()
    except Exception:  # noqa: BLE001
        logger.debug("Langfuse flush failed", exc_info=True)


# ---------------------------------------------------------------------------
# Span helpers (arbitrary processing steps)
# ---------------------------------------------------------------------------


@contextmanager
def trace_span(
    name: str,
    input_data: Any = None,
    metadata: Optional[dict] = None,
):
    """Open a child span under the active observation.

    Yields ``(span, end_span)`` where ``end_span(output=...)`` records output.
    """
    _init()
    if _mode == _TRACE_OFF:
        yield None, lambda **kw: None
        return

    if _mode == _TRACE_FILE:
        with _local_observation(name, "span", input_data, metadata) as span:

            def end_span(
                output: Any = None,
                metadata_extra: Optional[dict] = None,
                level: str = "DEFAULT",
            ):
                span.update(output=output, metadata=metadata_extra, level=level)

            yield span, end_span
        return

    with _client.start_as_current_observation(
        name=name,
        as_type="span",
        input=input_data,
        metadata=metadata,
    ) as span:
        ended = False

        def end_span(
            output: Any = None,
            metadata_extra: Optional[dict] = None,
            level: str = "DEFAULT",
        ):
            nonlocal ended
            if ended:
                return
            ended = True
            kwargs: dict[str, Any] = {"level": level}
            if output is not None:
                kwargs["output"] = output
            if metadata_extra:
                kwargs["metadata"] = metadata_extra
            try:
                span.update(**kwargs)
            except Exception:  # noqa: BLE001
                logger.debug("Langfuse span update failed for %s", name, exc_info=True)

        try:
            yield span, end_span
        except Exception as exc:
            if not ended:
                try:
                    span.update(level="ERROR", status_message=str(exc))
                except Exception:  # noqa: BLE001
                    pass
                ended = True
            raise


# ---------------------------------------------------------------------------
# Generation helpers (LLM calls)
# ---------------------------------------------------------------------------


@contextmanager
def trace_generation(
    name: str,
    model: Optional[str] = None,
    input_data: Any = None,
    metadata: Optional[dict] = None,
):
    """Open a generation observation for an LLM call.

    Yields ``(generation, end_generation)`` where
    ``end_generation(output=..., usage_details={...})`` records the result.
    """
    _init()
    if _mode == _TRACE_OFF:
        yield None, lambda **kw: None
        return

    if _mode == _TRACE_FILE:
        with _local_observation(name, "generation", input_data, metadata, model=model) as gen:

            def end_generation(
                output: Any = None,
                usage: Optional[dict] = None,
                usage_details: Optional[dict] = None,
                metadata_extra: Optional[dict] = None,
                level: str = "DEFAULT",
            ):
                gen.update(
                    output=output,
                    metadata=metadata_extra,
                    level=level,
                    usage_details=usage_details or usage,
                )

            yield gen, end_generation
        return

    kwargs: dict[str, Any] = {
        "name": name,
        "as_type": "generation",
        "input": input_data,
        "metadata": metadata,
    }
    if model:
        kwargs["model"] = model

    with _client.start_as_current_observation(**kwargs) as gen:
        ended = False

        def end_generation(
            output: Any = None,
            usage: Optional[dict] = None,
            usage_details: Optional[dict] = None,
            metadata_extra: Optional[dict] = None,
            level: str = "DEFAULT",
        ):
            nonlocal ended
            if ended:
                return
            ended = True
            upd: dict[str, Any] = {"level": level}
            if output is not None:
                upd["output"] = output
            # Support both v2-style "usage" and v4-style "usage_details"
            ud = usage_details or usage
            if ud:
                upd["usage_details"] = ud
            if metadata_extra:
                upd["metadata"] = metadata_extra
            try:
                gen.update(**upd)
            except Exception:  # noqa: BLE001
                logger.debug("Langfuse generation update failed for %s", name, exc_info=True)

        try:
            yield gen, end_generation
        except Exception as exc:
            if not ended:
                try:
                    gen.update(level="ERROR", status_message=str(exc))
                except Exception:  # noqa: BLE001
                    pass
                ended = True
            raise
