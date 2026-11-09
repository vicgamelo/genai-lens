"""genai-lens collector + dashboard (FastAPI)."""

import hmac
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib.resources import files
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import delete
from sqlalchemy.orm import Session

from .. import __version__
from ..pricing import PriceTable
from .db import SpanRow, make_engine, make_sessionmaker
from .metrics import WINDOWS, recent_traces, summary, timeseries, trace_detail
from .otlp import otlp_to_spans
from .schemas import IngestResult, SpanBatch, SpanIn


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GENAI_LENS_")

    database_url: str = "sqlite:///./genai-lens.db"
    # Comma separated. Empty means ingestion is open (fine for local dev only).
    api_keys: str = ""
    retention_days: int = 30


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    engine = make_engine(cfg.database_url)
    SessionLocal = make_sessionmaker(engine)
    prices = PriceTable.from_env()
    keys = [k.strip() for k in cfg.api_keys.split(",") if k.strip()]
    dashboard_html = files("genai_lens.server").joinpath("static/index.html").read_text()

    app = FastAPI(title="genai-lens", version=__version__, docs_url="/docs", redoc_url=None)
    app.state.engine = engine

    def db() -> Iterator[Session]:
        with SessionLocal() as session:
            yield session

    def require_key(
        x_lens_key: Annotated[str | None, Header()] = None, authorization: str | None = Header(None)
    ) -> None:
        if not keys:
            return
        supplied = x_lens_key or (authorization or "").removeprefix("Bearer ").strip()
        if not any(hmac.compare_digest(supplied, k) for k in keys):
            raise HTTPException(status_code=401, detail="invalid or missing ingest key")

    DB = Annotated[Session, Depends(db)]
    Window = Annotated[str, Query(pattern="^(" + "|".join(WINDOWS) + ")$")]

    def store(session: Session, items: list[tuple[str, SpanIn]]) -> int:
        now_ns = time.time_ns()
        for project, s in items:
            end = s.end_time_ns or s.start_time_ns
            session.merge(
                SpanRow(
                    span_id=s.span_id,
                    trace_id=s.trace_id,
                    parent_id=s.parent_id,
                    project=project,
                    name=s.name,
                    kind=s.kind,
                    status=s.status,
                    error=s.error,
                    start_time=datetime.fromtimestamp(s.start_time_ns / 1e9, tz=UTC),
                    duration_ms=(end - s.start_time_ns) / 1e6,
                    model=s.model,
                    provider=s.provider,
                    input_tokens=s.input_tokens,
                    output_tokens=s.output_tokens,
                    cost_usd=prices.cost(s.model, s.input_tokens, s.output_tokens),
                    input=s.input,
                    output=s.output,
                    attributes=s.attributes,
                    ingested_at_ns=now_ns,
                )
            )
        session.commit()
        return len(items)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/spans", response_model=IngestResult, dependencies=[Depends(require_key)])
    def ingest(batch: SpanBatch, session: DB) -> IngestResult:
        return IngestResult(accepted=store(session, [(batch.project, s) for s in batch.spans]))

    @app.post("/v1/traces", dependencies=[Depends(require_key)])
    async def ingest_otlp(request: Request, session: DB) -> dict[str, Any]:
        if "json" not in request.headers.get("content-type", ""):
            raise HTTPException(415, "only OTLP/HTTP JSON is supported (set OTEL_EXPORTER_OTLP_PROTOCOL=http/json)")
        try:
            items = otlp_to_spans(await request.json())
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(400, f"invalid OTLP payload: {exc}") from exc
        store(session, items)
        return {"partialSuccess": {}}

    @app.get("/api/summary")
    def api_summary(session: DB, window: Window = "24h", project: str | None = None) -> dict[str, Any]:
        return summary(session, project, window)

    @app.get("/api/timeseries")
    def api_timeseries(session: DB, window: Window = "24h", project: str | None = None) -> list[dict[str, Any]]:
        return timeseries(session, project, window)

    @app.get("/api/traces")
    def api_traces(
        session: DB, project: str | None = None, limit: Annotated[int, Query(ge=1, le=200)] = 50
    ) -> list[dict[str, Any]]:
        return recent_traces(session, project, limit)

    @app.get("/api/traces/{trace_id}")
    def api_trace(trace_id: str, session: DB) -> list[dict[str, Any]]:
        spans = trace_detail(session, trace_id)
        if not spans:
            raise HTTPException(404, "trace not found")
        return spans

    @app.post("/api/retention", dependencies=[Depends(require_key)])
    def apply_retention(session: DB) -> dict[str, int]:
        cutoff = datetime.now(UTC).timestamp() - cfg.retention_days * 86400
        result = session.execute(delete(SpanRow).where(SpanRow.start_time < datetime.fromtimestamp(cutoff, UTC)))
        session.commit()
        return {"deleted": result.rowcount or 0}  # type: ignore[attr-defined]

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        return dashboard_html

    return app


def main() -> None:
    import os

    import uvicorn

    uvicorn.run(
        "genai_lens.server.app:create_app",
        factory=True,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        proxy_headers=True,
    )
