"""Generate realistic synthetic traffic so the dashboard has something to show.

    uv run python examples/demo_traffic.py --endpoint http://localhost:8080 --traces 300

Simulates a support agent: a router LLM call, an optional retrieval step, an
optional tool call and a final answer, with a few rate-limit errors sprinkled in.
Spans are back-dated over the last 24h, so it uses the raw ingest API instead of
the SDK (which always records "now").
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import time
import urllib.request

MODELS = [("claude-haiku-4-5", "anthropic", 0.45), ("claude-sonnet-4-5", "anthropic", 0.3),
          ("gpt-4o-mini", "openai", 0.15), ("qwen2.5:7b", "ollama", 0.10)]
QUESTIONS = ["Qual o status do meu pedido 4821?", "Como troco a senha?", "Vocês emitem nota fiscal?",
             "Quero cancelar a assinatura", "O boleto venceu, e agora?", "Tem desconto para estudante?"]


def span(trace_id, parent, name, kind, start, dur_ms, **extra):
    return {"trace_id": trace_id, "span_id": secrets.token_hex(8), "parent_id": parent, "name": name, "kind": kind,
            "start_time_ns": start, "end_time_ns": start + int(dur_ms * 1e6), **extra}


def fake_trace(now_ns: int) -> list[dict]:
    trace_id = secrets.token_hex(16)
    t0 = now_ns - random.randint(0, 24 * 3600) * 10**9
    model, provider, _ = random.choices(MODELS, weights=[m[2] for m in MODELS])[0]
    question = random.choice(QUESTIONS)
    spans, cursor = [], t0 + int(5e6)
    root = span(trace_id, None, "support_agent.run", "agent", t0, 0, input=question)

    def llm(name, tin, tout, base_ms):
        nonlocal cursor
        dur = random.lognormvariate(0, 0.35) * base_ms * (0.4 if provider == "ollama" else 1)
        failed = random.random() < 0.03
        s = span(trace_id, root["span_id"], name, "llm", cursor, dur, model=model, provider=provider,
                 input_tokens=tin, output_tokens=0 if failed else tout,
                 status="error" if failed else "ok", error="RateLimitError: 429 Too Many Requests" if failed else None)
        cursor += int(dur * 1e6) + int(3e6)
        spans.append(s)
        return failed

    failed = llm("router.classify", random.randint(300, 600), random.randint(5, 20), 450)
    if not failed and random.random() < 0.7:
        dur = random.uniform(40, 180)
        spans.append(span(trace_id, root["span_id"], "kb.search", "retrieval", cursor, dur, attributes={"top_k": 5}))
        cursor += int(dur * 1e6)
    if not failed and random.random() < 0.4:
        dur = random.uniform(80, 600)
        spans.append(span(trace_id, root["span_id"], "tool.get_order_status", "tool", cursor, dur))
        cursor += int(dur * 1e6)
    if not failed:
        failed = llm("answer.generate", random.randint(1200, 4000), random.randint(120, 600), 1800)
    root["end_time_ns"] = cursor + int(2e6)
    root["status"] = "error" if failed else "ok"
    root["output"] = None if failed else "Resposta enviada ao cliente."
    return [root, *spans]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--traces", type=int, default=300)
    parser.add_argument("--project", default="support-bot")
    args = parser.parse_args()
    now = time.time_ns()
    batch: list[dict] = []
    for _ in range(args.traces):
        batch += fake_trace(now)
    for i in range(0, len(batch), 500):
        body = json.dumps({"project": args.project, "spans": batch[i : i + 500]}).encode()
        req = urllib.request.Request(f"{args.endpoint}/v1/spans", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "X-Lens-Key": args.api_key})
        urllib.request.urlopen(req, timeout=10).read()
    print(f"sent {len(batch)} spans in {args.traces} traces")


if __name__ == "__main__":
    main()
