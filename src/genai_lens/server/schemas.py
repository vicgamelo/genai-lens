from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Kind = Literal["llm", "tool", "agent", "retrieval", "chain", "embedding", "other"]


class SpanIn(BaseModel):
    trace_id: str = Field(min_length=1, max_length=32)
    span_id: str = Field(min_length=1, max_length=32)
    parent_id: str | None = Field(default=None, max_length=32)
    name: str = Field(max_length=300)
    kind: Kind = "other"
    start_time_ns: int = Field(gt=0)
    end_time_ns: int | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    model: str | None = None
    provider: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    input: str | None = None
    output: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("end_time_ns")
    @classmethod
    def _end_after_start(cls, v: int | None, info: Any) -> int | None:
        start = info.data.get("start_time_ns")
        if v is not None and start is not None and v < start:
            raise ValueError("end_time_ns must be >= start_time_ns")
        return v


class SpanBatch(BaseModel):
    project: str = Field(default="default", min_length=1, max_length=100, pattern=r"^[\w.\-]+$")
    spans: list[SpanIn] = Field(max_length=1000)


class IngestResult(BaseModel):
    accepted: int
