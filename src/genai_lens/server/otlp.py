"""Map OTLP/HTTP JSON traces to genai-lens spans.

Any app instrumented with OpenTelemetry using the GenAI semantic conventions
(``gen_ai.*`` attributes: OpenLLMetry, OpenLIT, the official OTel
instrumentations for OpenAI/Anthropic/Bedrock...) can export straight to
``POST /v1/traces`` with no genai-lens SDK installed.

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

import base64
from typing import Any

from .schemas import SpanIn

# OTLP status codes: 0 unset, 1 ok, 2 error
_STATUS_ERROR = 2

_OPERATION_TO_KIND = {
    "chat": "llm",
    "text_completion": "llm",
    "generate_content": "llm",
    "embeddings": "embedding",
    "execute_tool": "tool",
    "invoke_agent": "agent",
    "create_agent": "agent",
}


def _any_value(value: dict[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])  # int64 is serialized as a string in OTLP JSON
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_any_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return _attributes(value["kvlistValue"].get("values", []))
    return None


def _attributes(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {item["key"]: _any_value(item.get("value", {})) for item in items}


def _hex_id(value: str | None) -> str | None:
    """OTLP JSON spec says hex, but some exporters send base64. Accept both."""
    if not value:
        return None
    try:
        int(value, 16)
        return value.lower()
    except ValueError:
        return base64.b64decode(value).hex()


def otlp_to_spans(payload: dict[str, Any]) -> list[tuple[str, SpanIn]]:
    """Return ``(project, span)`` pairs. Project comes from ``service.name``."""
    out: list[tuple[str, SpanIn]] = []
    for resource_spans in payload.get("resourceSpans", []):
        resource = _attributes(resource_spans.get("resource", {}).get("attributes", []))
        project = str(resource.get("service.name") or "default")[:100]
        for scope_spans in resource_spans.get("scopeSpans", []):
            for raw in scope_spans.get("spans", []):
                attrs = _attributes(raw.get("attributes", []))
                operation = attrs.get("gen_ai.operation.name")
                status = raw.get("status", {})
                is_error = status.get("code") in (_STATUS_ERROR, "STATUS_CODE_ERROR")
                input_tokens = attrs.pop("gen_ai.usage.input_tokens", attrs.pop("gen_ai.usage.prompt_tokens", None))
                output_tokens = attrs.pop(
                    "gen_ai.usage.output_tokens", attrs.pop("gen_ai.usage.completion_tokens", None)
                )
                span = SpanIn(
                    trace_id=_hex_id(raw["traceId"]) or "",
                    span_id=_hex_id(raw["spanId"]) or "",
                    parent_id=_hex_id(raw.get("parentSpanId")),
                    name=str(raw.get("name", "span"))[:300],
                    kind=_OPERATION_TO_KIND.get(str(operation), "other"),  # type: ignore[arg-type]
                    start_time_ns=int(raw["startTimeUnixNano"]),
                    end_time_ns=int(raw["endTimeUnixNano"]) if raw.get("endTimeUnixNano") else None,
                    status="error" if is_error else "ok",
                    error=status.get("message") if is_error else None,
                    model=attrs.pop("gen_ai.response.model", None) or attrs.pop("gen_ai.request.model", None),
                    provider=attrs.pop("gen_ai.provider.name", None) or attrs.pop("gen_ai.system", None),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attributes={k: v for k, v in attrs.items() if isinstance(v, str | int | float | bool)},
                )
                out.append((project, span))
    return out
