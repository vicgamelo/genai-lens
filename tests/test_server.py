import secrets
import time
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from genai_lens import Lens
from genai_lens.server.app import Settings, create_app
from genai_lens.server.metrics import percentile


@pytest.fixture
def client(db_url: str) -> TestClient:
    return TestClient(create_app(Settings(database_url=db_url, api_keys="k1,k2")))


H = {"X-Lens-Key": "k1"}


def span(trace: str, parent: str | None = None, **kw: Any) -> dict[str, Any]:
    start = kw.pop("start", time.time_ns())
    dur_ms = kw.pop("dur_ms", 100)
    return {
        "trace_id": trace,
        "span_id": kw.pop("span_id", secrets.token_hex(8)),
        "parent_id": parent,
        "name": kw.pop("name", "s"),
        "start_time_ns": start,
        "end_time_ns": start + int(dur_ms * 1e6),
        **kw,
    }


def test_ingest_requires_key(client: TestClient) -> None:
    body = {"spans": [span("t1")]}
    assert client.post("/v1/spans", json=body).status_code == 401
    assert client.post("/v1/spans", json=body, headers={"X-Lens-Key": "nope"}).status_code == 401
    assert client.post("/v1/spans", json=body, headers={"Authorization": "Bearer k2"}).status_code == 200


def test_rejects_end_before_start(client: TestClient) -> None:
    s = span("t1")
    s["end_time_ns"] = s["start_time_ns"] - 1
    assert client.post("/v1/spans", json={"spans": [s]}, headers=H).status_code == 422


def test_summary_cost_and_error_rate(client: TestClient) -> None:
    root_ok, root_err = (
        span("t1", kind="agent", span_id="a" * 16),
        span("t2", kind="agent", status="error", span_id="b" * 16),
    )
    spans = [
        root_ok,
        span(
            "t1",
            root_ok["span_id"],
            kind="llm",
            model="gpt-4o-mini",
            input_tokens=1_000_000,
            output_tokens=0,
            dur_ms=200,
        ),
        root_err,
        span(
            "t2",
            root_err["span_id"],
            kind="llm",
            model="gpt-4o-mini",
            input_tokens=0,
            output_tokens=1_000_000,
            dur_ms=400,
            status="error",
        ),
    ]
    assert client.post("/v1/spans", json={"project": "bot", "spans": spans}, headers=H).json() == {"accepted": 4}
    s = client.get("/api/summary?window=1h").json()
    assert s["traces"] == 2 and s["llm_calls"] == 2
    assert s["error_rate"] == 0.5
    assert s["cost_usd"] == pytest.approx(0.15 + 0.6)
    assert s["p50_ms"] == pytest.approx(300, rel=0.01)
    assert s["models"][0]["model"] == "gpt-4o-mini" and s["models"][0]["errors"] == 1
    assert s["projects"] == ["bot"]


def test_reingesting_same_span_is_idempotent(client: TestClient) -> None:
    s = span("t1", kind="llm", model="gpt-4o", input_tokens=10)
    client.post("/v1/spans", json={"spans": [s]}, headers=H)
    client.post("/v1/spans", json={"spans": [s]}, headers=H)
    assert client.get("/api/summary?window=1h").json()["llm_calls"] == 1


def test_traces_list_and_detail(client: TestClient) -> None:
    root = span("t9", name="agent.run", kind="agent")
    child = span("t9", root["span_id"], name="llm", kind="llm", input_tokens=7, output_tokens=3, input="oi")
    client.post("/v1/spans", json={"spans": [root, child]}, headers=H)
    traces = client.get("/api/traces").json()
    assert traces[0]["trace_id"] == "t9" and traces[0]["spans"] == 2 and traces[0]["tokens"] == 10
    detail = client.get("/api/traces/t9").json()
    assert [d["name"] for d in detail] == ["agent.run", "llm"]
    assert client.get("/api/traces/missing").status_code == 404


def test_project_filter_and_window_validation(client: TestClient) -> None:
    client.post("/v1/spans", json={"project": "a", "spans": [span("x1", kind="llm")]}, headers=H)
    client.post("/v1/spans", json={"project": "b", "spans": [span("x2", kind="llm")]}, headers=H)
    assert client.get("/api/summary?window=1h&project=a").json()["llm_calls"] == 1
    assert client.get("/api/summary?window=2y").status_code == 422


def test_old_spans_outside_window(client: TestClient) -> None:
    old = span("old", kind="llm", start=int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1e9))
    client.post("/v1/spans", json={"spans": [old]}, headers=H)
    assert client.get("/api/summary?window=30d").json()["llm_calls"] == 0
    assert client.post("/api/retention", headers=H).json() == {"deleted": 1}


def test_timeseries_buckets(client: TestClient) -> None:
    client.post("/v1/spans", json={"spans": [span("t", kind="llm"), span("u", kind="llm", status="error")]}, headers=H)
    series = client.get("/api/timeseries?window=24h").json()
    assert sum(p["calls"] for p in series) == 2
    assert sum(p["errors"] for p in series) == 1


def test_sdk_end_to_end_through_http(client: TestClient) -> None:
    def via_testclient(batch: list[dict[str, Any]]) -> None:
        r = client.post("/v1/spans", json={"project": "sdk", "spans": batch}, headers=H)
        r.raise_for_status()

    lens = Lens(transport=via_testclient, flush_interval=60)
    with lens.span("agent", kind="agent"), lens.span("call", kind="llm") as s:
        s.set_llm(model="claude-haiku-4-5", provider="anthropic", input_tokens=1000, output_tokens=200)
    lens.flush()
    summary = client.get("/api/summary?window=1h&project=sdk").json()
    assert summary["traces"] == 1
    assert summary["cost_usd"] == pytest.approx((1000 * 0.8 + 200 * 4.0) / 1e6)


def test_dashboard_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "genai" in r.text


def test_percentile_matches_numpy_linear() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10, 20, 30, 40, 50], 0.95) == pytest.approx(48)
    assert percentile([], 0.5) is None
