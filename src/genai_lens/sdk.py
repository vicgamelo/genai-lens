"""Zero-dependency tracing SDK.

    from genai_lens import Lens

    lens = Lens(endpoint="https://lens.vgmelo.tech", api_key="...", project="support-bot")

    @lens.trace(kind="agent")
    def answer(question: str) -> str:
        with lens.span("anthropic.messages", kind="llm") as span:
            resp = client.messages.create(...)
            span.set_llm(model=resp.model, provider="anthropic",
                         input_tokens=resp.usage.input_tokens,
                         output_tokens=resp.usage.output_tokens)
            span.set_io(input=question, output=resp.content[0].text)
        return resp.content[0].text

Spans nest through ``contextvars`` (works with threads and asyncio), are buffered
in memory and shipped in batches by a daemon thread. The SDK never raises into
the host application: if the collector is down, spans are dropped and counted.
"""

from __future__ import annotations

import atexit
import contextvars
import functools
import inspect
import json
import logging
import os
import queue
import secrets
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal, ParamSpec, TypeVar

from .redact import redact, truncate

logger = logging.getLogger("genai_lens")

SpanKind = Literal["llm", "tool", "agent", "retrieval", "chain", "embedding", "other"]
P = ParamSpec("P")
R = TypeVar("R")

_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar("genai_lens_span", default=None)


class Span:
    __slots__ = (
        "_lens",
        "attributes",
        "end_ns",
        "error",
        "input",
        "input_tokens",
        "kind",
        "model",
        "name",
        "output",
        "output_tokens",
        "parent_id",
        "provider",
        "span_id",
        "start_ns",
        "status",
        "trace_id",
    )

    def __init__(self, lens: Lens, name: str, kind: SpanKind, parent: Span | None) -> None:
        self._lens = lens
        self.name = name
        self.kind = kind
        self.trace_id: str = parent.trace_id if parent else secrets.token_hex(16)
        self.span_id: str = secrets.token_hex(8)
        self.parent_id: str | None = parent.span_id if parent else None
        self.start_ns = time.time_ns()
        self.end_ns: int | None = None
        self.status: Literal["ok", "error"] = "ok"
        self.error: str | None = None
        self.model: str | None = None
        self.provider: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.input: str | None = None
        self.output: str | None = None
        self.attributes: dict[str, Any] = {}

    def set_llm(
        self,
        *,
        model: str,
        provider: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> Span:
        self.model, self.provider = model, provider
        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        return self

    def set_io(self, *, input: Any = None, output: Any = None) -> Span:
        if input is not None:
            self.input = self._lens._prepare_text(input)
        if output is not None:
            self.output = self._lens._prepare_text(output)
        return self

    def set_attribute(self, key: str, value: str | int | float | bool) -> Span:
        self.attributes[key] = value
        return self

    def record_error(self, exc: BaseException) -> None:
        self.status = "error"
        self.error = f"{type(exc).__name__}: {exc}"[:500]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start_time_ns": self.start_ns,
            "end_time_ns": self.end_ns,
            "status": self.status,
            "error": self.error,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input": self.input,
            "output": self.output,
            "attributes": self.attributes,
        }


class Lens:
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        project: str = "default",
        *,
        enabled: bool = True,
        capture_io: bool = True,
        redact_pii: bool = True,
        max_text_chars: int = 4000,
        batch_size: int = 50,
        flush_interval: float = 2.0,
        max_queue: int = 10_000,
        transport: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self.endpoint: str = (endpoint or os.getenv("GENAI_LENS_ENDPOINT") or "http://localhost:8080").rstrip("/")
        self.api_key: str = api_key or os.getenv("GENAI_LENS_API_KEY") or ""
        self.project = project
        self.enabled = enabled
        self.capture_io = capture_io
        self.redact_pii = redact_pii
        self.max_text_chars = max_text_chars
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.dropped = 0
        self._send = transport or self._http_send
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._flush_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        if enabled:
            self._worker = threading.Thread(target=self._run, name="genai-lens", daemon=True)
            self._worker.start()
            atexit.register(self.shutdown)

    # public API -------------------------------------------------------------
    @contextmanager
    def span(self, name: str, kind: SpanKind = "other") -> Iterator[Span]:
        span = Span(self, name, kind, _current.get())
        token = _current.set(span)
        try:
            yield span
        except BaseException as exc:
            span.record_error(exc)
            raise
        finally:
            _current.reset(token)
            span.end_ns = time.time_ns()
            self._enqueue(span)

    def trace(self, name: str | None = None, kind: SpanKind = "chain") -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator for sync and async functions."""

        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            span_name = name or fn.__qualname__

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                    with self.span(span_name, kind):
                        return await fn(*args, **kwargs)

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with self.span(span_name, kind):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def current_span() -> Span | None:
        return _current.get()

    def flush(self) -> None:
        with self._flush_lock:
            while True:
                batch = self._drain(self.batch_size)
                if not batch:
                    return
                try:
                    self._send(batch)
                except Exception as exc:  # never break the host app
                    self.dropped += len(batch)
                    logger.warning("genai-lens: dropped %d spans (%s)", len(batch), exc)

    def shutdown(self) -> None:
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5)
        self.flush()

    # internals --------------------------------------------------------------
    def _prepare_text(self, value: Any) -> str | None:
        if not self.capture_io:
            return None
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if self.redact_pii:
            text = redact(text)
        return truncate(text, self.max_text_chars)

    def _enqueue(self, span: Span) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(span.to_dict())
        except queue.Full:
            self.dropped += 1

    def _drain(self, limit: int) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        while len(batch) < limit:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop.wait(self.flush_interval):
            self.flush()

    def _http_send(self, batch: list[dict[str, Any]]) -> None:
        body = json.dumps({"project": self.project, "spans": batch}).encode()
        request = urllib.request.Request(
            f"{self.endpoint}/v1/spans",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-Lens-Key": self.api_key},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
