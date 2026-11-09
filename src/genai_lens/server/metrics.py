"""Aggregations for the dashboard.

Percentiles are computed in Python so the same code runs on SQLite and Postgres;
the queried window is bounded (max 30 days) to keep this cheap.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import SpanRow

WINDOWS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30)}


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation percentile (same as numpy's default)."""
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * q
    low = int(pos)
    high = min(low + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (pos - low)


def _since(window: str, now: datetime | None = None) -> datetime:
    return (now or datetime.now(UTC)) - WINDOWS[window]


def _rows(session: Session, project: str | None, since: datetime) -> list[SpanRow]:
    stmt = select(SpanRow).where(SpanRow.start_time >= since)
    if project:
        stmt = stmt.where(SpanRow.project == project)
    return list(session.scalars(stmt))


def summary(session: Session, project: str | None, window: str, now: datetime | None = None) -> dict[str, Any]:
    rows = _rows(session, project, _since(window, now))
    llm = [r for r in rows if r.kind == "llm"]
    roots = [r for r in rows if r.parent_id is None]
    latencies = [r.duration_ms for r in llm]
    errors = sum(1 for r in roots if r.status == "error")

    by_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "_lat": []}
    )
    for r in llm:
        m = by_model[r.model or "unknown"]
        m["calls"] += 1
        m["errors"] += r.status == "error"
        m["input_tokens"] += r.input_tokens or 0
        m["output_tokens"] += r.output_tokens or 0
        m["cost_usd"] += r.cost_usd or 0.0
        m["_lat"].append(r.duration_ms)
    models = []
    for name, m in sorted(by_model.items(), key=lambda kv: -kv[1]["cost_usd"]):
        lat = m.pop("_lat")
        models.append({"model": name, **m, "cost_usd": round(m["cost_usd"], 6), "p95_ms": percentile(lat, 0.95)})

    return {
        "window": window,
        "traces": len(roots),
        "llm_calls": len(llm),
        "error_rate": round(errors / len(roots), 4) if roots else 0.0,
        "p50_ms": percentile(latencies, 0.5),
        "p95_ms": percentile(latencies, 0.95),
        "input_tokens": sum(r.input_tokens or 0 for r in llm),
        "output_tokens": sum(r.output_tokens or 0 for r in llm),
        "cost_usd": round(sum(r.cost_usd or 0.0 for r in llm), 6),
        "models": models,
        "projects": sorted({r.project for r in rows}),
    }


def timeseries(session: Session, project: str | None, window: str, now: datetime | None = None) -> list[dict[str, Any]]:
    """LLM calls bucketed by hour (or by day for 7d/30d)."""
    since = _since(window, now)
    by_day = window in {"7d", "30d"}
    buckets: dict[datetime, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "errors": 0, "cost_usd": 0.0, "tokens": 0, "_lat": []}
    )
    for r in _rows(session, project, since):
        if r.kind != "llm":
            continue
        ts = r.start_time if r.start_time.tzinfo else r.start_time.replace(tzinfo=UTC)
        key = ts.replace(minute=0, second=0, microsecond=0)
        if by_day:
            key = key.replace(hour=0)
        b = buckets[key]
        b["calls"] += 1
        b["errors"] += r.status == "error"
        b["cost_usd"] += r.cost_usd or 0.0
        b["tokens"] += (r.input_tokens or 0) + (r.output_tokens or 0)
        b["_lat"].append(r.duration_ms)
    out = []
    for key in sorted(buckets):
        b = buckets[key]
        lat = b.pop("_lat")
        out.append({"t": key.isoformat(), **b, "cost_usd": round(b["cost_usd"], 6), "p95_ms": percentile(lat, 0.95)})
    return out


def recent_traces(session: Session, project: str | None, limit: int) -> list[dict[str, Any]]:
    stmt = select(SpanRow).where(SpanRow.parent_id.is_(None)).order_by(SpanRow.start_time.desc()).limit(limit)
    if project:
        stmt = stmt.where(SpanRow.project == project)
    roots = list(session.scalars(stmt))
    if not roots:
        return []
    children = list(session.scalars(select(SpanRow).where(SpanRow.trace_id.in_([r.trace_id for r in roots]))))
    agg: dict[str, dict[str, Any]] = defaultdict(lambda: {"spans": 0, "tokens": 0, "cost_usd": 0.0, "errors": 0})
    for c in children:
        a = agg[c.trace_id]
        a["spans"] += 1
        a["tokens"] += (c.input_tokens or 0) + (c.output_tokens or 0)
        a["cost_usd"] += c.cost_usd or 0.0
        a["errors"] += c.status == "error"
    return [
        {
            "trace_id": r.trace_id,
            "name": r.name,
            "project": r.project,
            "start_time": r.start_time.isoformat(),
            "duration_ms": r.duration_ms,
            "status": "error" if agg[r.trace_id]["errors"] else r.status,
            **{k: (round(v, 6) if k == "cost_usd" else v) for k, v in agg[r.trace_id].items() if k != "errors"},
        }
        for r in roots
    ]


def trace_detail(session: Session, trace_id: str) -> list[dict[str, Any]]:
    rows = list(session.scalars(select(SpanRow).where(SpanRow.trace_id == trace_id).order_by(SpanRow.start_time)))
    return [
        {
            "span_id": r.span_id,
            "parent_id": r.parent_id,
            "name": r.name,
            "kind": r.kind,
            "status": r.status,
            "error": r.error,
            "start_time": r.start_time.isoformat(),
            "duration_ms": r.duration_ms,
            "model": r.model,
            "provider": r.provider,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": r.cost_usd,
            "input": r.input,
            "output": r.output,
            "attributes": r.attributes,
        }
        for r in rows
    ]
