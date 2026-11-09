# genai-lens

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-GenAI%20semconv-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

Self-hosted observability for LLM applications and agents. Traces, token usage, cost,
latency percentiles and errors per model, in a single container you can run on a $5 VPS.

![dashboard](docs/dashboard.jpg)

## Why another one?

Langfuse, Phoenix and LangSmith are great, but they're heavy to self-host or are SaaS-first.
genai-lens is the minimum a team needs to answer *"what are our agents doing, how much does it
cost and where is it slow?"*:

- **One container, SQLite or Postgres.** No ClickHouse, Redis or queue to run.
- **OpenTelemetry native.** `POST /v1/traces` accepts OTLP/HTTP JSON and maps the
  [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), so
  OpenLLMetry, OpenLIT or the official OTel instrumentations work without our SDK.
- **A zero-dependency Python SDK** (stdlib only) with nested spans via `contextvars`, batching in
  a background thread, and a hard rule: it never raises into your app.
- **PII redaction on the client.** E-mails, CPF/CNPJ, phones, card numbers and API keys are
  masked before anything leaves the process.
- **Cost at ingestion time** from a price table matched by longest model prefix, overridable
  with a JSON file.

## Architecture

```mermaid
flowchart LR
    subgraph Your apps
      A[Python app + genai-lens SDK]
      B[Any app with OTel GenAI instrumentation]
    end
    A -- "POST /v1/spans (batched)" --> C
    B -- "POST /v1/traces (OTLP/HTTP JSON)" --> C
    C[genai-lens collector<br/>FastAPI] -->|cost + normalize| D[(SQLite / Postgres)]
    D --> E[/api/summary · /api/timeseries · /api/traces/]
    E --> F[Dashboard]
```

## Quickstart

```bash
docker compose up -d                 # collector + Postgres on :8080, ingest key "dev-key"
uv run python examples/demo_traffic.py --endpoint http://localhost:8080 --api-key dev-key
open http://localhost:8080
```

### Instrument your code

```python
from genai_lens import Lens

lens = Lens(endpoint="http://localhost:8080", api_key="dev-key", project="support-bot")

@lens.trace(kind="agent")
def answer(question: str) -> str:
    with lens.span("anthropic.messages", kind="llm") as span:
        resp = client.messages.create(model="claude-haiku-4-5", max_tokens=500,
                                      messages=[{"role": "user", "content": question}])
        span.set_llm(model=resp.model, provider="anthropic",
                     input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)
        span.set_io(input=question, output=resp.content[0].text)   # redacted + truncated
    return resp.content[0].text
```

Works with `async def` too; concurrent tasks keep separate traces.

### Or just point OpenTelemetry at it

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:8080/v1/traces
export OTEL_EXPORTER_OTLP_PROTOCOL=http/json
export OTEL_EXPORTER_OTLP_HEADERS="x-lens-key=dev-key"
export OTEL_SERVICE_NAME=rag-api          # becomes the project
```

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `GENAI_LENS_DATABASE_URL` | `sqlite:///./genai-lens.db` | Any SQLAlchemy URL (`postgresql+psycopg://...`) |
| `GENAI_LENS_API_KEYS` | *(empty: open)* | Comma separated ingest keys (`X-Lens-Key` or `Authorization: Bearer`) |
| `GENAI_LENS_RETENTION_DAYS` | `30` | Used by `POST /api/retention` (run it from a cron) |
| `GENAI_LENS_PRICING_FILE` | — | JSON `{ "model-prefix": {"input_per_mtok": 1, "output_per_mtok": 2} }` |

> The dashboard and read APIs have no built-in auth: put them behind your reverse proxy
> (basic auth in Caddy or nginx is enough).

## API

| Method | Path | |
| --- | --- | --- |
| `POST` | `/v1/spans` | Native batch ingest (used by the SDK) |
| `POST` | `/v1/traces` | OTLP/HTTP JSON ingest |
| `GET` | `/api/summary?window=1h\|24h\|7d\|30d&project=` | KPIs and per-model breakdown |
| `GET` | `/api/timeseries` | Calls, errors, cost, tokens and p95 per hour/day |
| `GET` | `/api/traces`, `/api/traces/{id}` | Recent traces and span waterfall |
| `GET` | `/docs` | OpenAPI |

## Development

```bash
uv sync --all-extras
uv run pytest                      # SQLite in memory
GENAI_LENS_TEST_DATABASE_URL=postgresql+psycopg://lens:lens@localhost:5432/lens uv run pytest
uv run ruff check . && uv run mypy
```

## Roadmap

- [ ] OTLP protobuf ingest
- [ ] Evaluations: attach scores (LLM-as-judge, user feedback) to traces
- [ ] Alerts on error rate / cost budget (webhook)
- [ ] Session view for multi-turn conversations

## License

MIT © Victor Gabriel Andrade de Melo
