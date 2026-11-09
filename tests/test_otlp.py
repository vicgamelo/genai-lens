import base64

from fastapi.testclient import TestClient

from genai_lens.server.app import Settings, create_app
from genai_lens.server.otlp import otlp_to_spans

TRACE = "5b8efff798038103d269b633813fc60c"
PAYLOAD = {
    "resourceSpans": [
        {
            "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "rag-api"}}]},
            "scopeSpans": [
                {
                    "spans": [
                        {
                            "traceId": TRACE,
                            "spanId": "eee19b7ec3c1b174",
                            "parentSpanId": "eee19b7ec3c1b173",
                            "name": "chat gpt-4o-mini",
                            "startTimeUnixNano": "1762700000000000000",
                            "endTimeUnixNano": "1762700001500000000",
                            "attributes": [
                                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                                {"key": "gen_ai.provider.name", "value": {"stringValue": "openai"}},
                                {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4o-mini"}},
                                {"key": "gen_ai.response.model", "value": {"stringValue": "gpt-4o-mini-2024-07-18"}},
                                {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "120"}},
                                {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "42"}},
                                {"key": "gen_ai.request.temperature", "value": {"doubleValue": 0.2}},
                            ],
                            "status": {"code": 2, "message": "upstream timeout"},
                        }
                    ]
                }
            ],
        }
    ]
}


def test_maps_genai_semconv() -> None:
    [(project, span)] = otlp_to_spans(PAYLOAD)
    assert project == "rag-api"
    assert span.kind == "llm"
    assert span.model == "gpt-4o-mini-2024-07-18"
    assert span.provider == "openai"
    assert (span.input_tokens, span.output_tokens) == (120, 42)
    assert span.status == "error" and span.error == "upstream timeout"
    assert span.attributes["gen_ai.request.temperature"] == 0.2
    assert span.end_time_ns is not None and span.end_time_ns - span.start_time_ns == 1_500_000_000


def test_accepts_base64_ids_and_legacy_token_names() -> None:
    raw = bytes.fromhex(TRACE)
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": base64.b64encode(raw).decode(),
                                "spanId": base64.b64encode(bytes.fromhex("eee19b7ec3c1b174")).decode(),
                                "name": "embeddings",
                                "startTimeUnixNano": "1762700000000000000",
                                "attributes": [
                                    {"key": "gen_ai.operation.name", "value": {"stringValue": "embeddings"}},
                                    {"key": "gen_ai.system", "value": {"stringValue": "openai"}},
                                    {"key": "gen_ai.usage.prompt_tokens", "value": {"intValue": "9"}},
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    [(project, span)] = otlp_to_spans(payload)
    assert project == "default"
    assert span.trace_id == TRACE
    assert span.kind == "embedding" and span.provider == "openai" and span.input_tokens == 9


def test_otlp_endpoint_stores_spans(db_url: str) -> None:
    client = TestClient(create_app(Settings(database_url=db_url)))
    assert client.post("/v1/traces", json=PAYLOAD).json() == {"partialSuccess": {}}
    detail = client.get(f"/api/traces/{TRACE}").json()
    assert detail[0]["cost_usd"] is not None and detail[0]["status"] == "error"
    assert (
        client.post("/v1/traces", content=b"\x00", headers={"content-type": "application/x-protobuf"}).status_code
        == 415
    )
    assert client.post("/v1/traces", json={"resourceSpans": [{"scopeSpans": [{"spans": [{}]}]}]}).status_code == 400
