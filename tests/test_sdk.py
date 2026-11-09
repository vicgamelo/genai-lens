import asyncio
import threading
from typing import Any

import pytest

from genai_lens import Lens
from genai_lens.pricing import Price, PriceTable
from genai_lens.redact import redact


class Collector:
    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def __call__(self, batch: list[dict[str, Any]]) -> None:
        with self.lock:
            self.spans.extend(batch)


def make_lens(**kw: Any) -> tuple[Lens, Collector]:
    sink = Collector()
    return Lens(transport=sink, flush_interval=60, **kw), sink


def test_nested_spans_share_trace_and_link_parent() -> None:
    lens, sink = make_lens()

    @lens.trace(kind="agent")
    def agent() -> None:
        with lens.span("llm", kind="llm") as s:
            s.set_llm(model="gpt-4o-mini", input_tokens=10, output_tokens=5)

    agent()
    lens.flush()
    child, root = sink.spans  # child finishes first
    assert root["parent_id"] is None and root["kind"] == "agent"
    assert child["parent_id"] == root["span_id"]
    assert child["trace_id"] == root["trace_id"]
    assert child["end_time_ns"] >= child["start_time_ns"]


def test_exception_marks_span_error_and_propagates() -> None:
    lens, sink = make_lens()
    with pytest.raises(ZeroDivisionError), lens.span("boom"):
        _ = 1 / 0
    lens.flush()
    assert sink.spans[0]["status"] == "error"
    assert "ZeroDivisionError" in sink.spans[0]["error"]


def test_async_tasks_keep_separate_traces() -> None:
    lens, sink = make_lens()

    @lens.trace()
    async def job(i: int) -> None:
        with lens.span(f"inner-{i}"):
            await asyncio.sleep(0.01)

    async def main() -> None:
        await asyncio.gather(job(1), job(2))

    asyncio.run(main())
    lens.flush()
    by_name = {s["name"]: s for s in sink.spans}
    roots = [s for s in sink.spans if s["parent_id"] is None]
    assert len(roots) == 2
    for i in (1, 2):
        inner = by_name[f"inner-{i}"]
        parent = next(r for r in roots if r["span_id"] == inner["parent_id"])
        assert parent["trace_id"] == inner["trace_id"]


def test_io_is_redacted_and_truncated() -> None:
    lens, sink = make_lens(max_text_chars=60)
    with lens.span("x") as s:
        s.set_io(input="meu cpf é 529.982.247-25 e email joao@empresa.com.br", output="a" * 200)
    lens.flush()
    span = sink.spans[0]
    assert "529.982" not in span["input"] and "[CPF]" in span["input"] and "[EMAIL]" in span["input"]
    assert len(span["output"]) == 60


def test_capture_io_disabled() -> None:
    lens, sink = make_lens(capture_io=False)
    with lens.span("x") as s:
        s.set_io(input="secret", output="secret")
    lens.flush()
    assert sink.spans[0]["input"] is None


def test_transport_failure_never_raises() -> None:
    def broken(_: list[dict[str, Any]]) -> None:
        raise ConnectionError("collector down")

    lens = Lens(transport=broken, flush_interval=60)
    with lens.span("x"):
        pass
    lens.flush()
    assert lens.dropped == 1


def test_disabled_lens_records_nothing() -> None:
    lens, sink = make_lens(enabled=False)
    with lens.span("x"):
        pass
    lens.flush()
    assert sink.spans == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ligue (12) 99123-4567", "ligue [PHONE]"),
        ("cnpj 11.222.333/0001-81", "cnpj [CNPJ]"),
        ("key sk-ant-abcdefghijklmnopqrstu", "key [API_KEY]"),
        ("cartão 4111 1111 1111 1111", "cartão [CARD]"),
        ("pedido 4821 de 2026-11-09", "pedido 4821 de 2026-11-09"),
    ],
)
def test_redact(text: str, expected: str) -> None:
    assert redact(text) == expected


def test_pricing_longest_prefix_and_provider_prefix() -> None:
    table = PriceTable({"gpt-4o": Price(2.5, 10), "gpt-4o-mini": Price(0.15, 0.6)})
    assert table.cost("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    assert table.cost("openai/gpt-4o", 0, 1_000_000) == pytest.approx(10)
    assert table.cost("unknown-model", 100, 100) is None
    assert PriceTable().cost("qwen2.5:7b", 5000, 5000) == 0
