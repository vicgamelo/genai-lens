"""Tracing a real Anthropic call with the SDK.

    export ANTHROPIC_API_KEY=...  GENAI_LENS_ENDPOINT=http://localhost:8080
    uv run --with anthropic python examples/anthropic_agent.py
"""

import anthropic

from genai_lens import Lens

lens = Lens(project="examples")
client = anthropic.Anthropic()


@lens.trace(kind="agent")
def ask(question: str) -> str:
    with lens.span("anthropic.messages.create", kind="llm") as span:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": question}],
        )
        text = resp.content[0].text
        span.set_llm(model=resp.model, provider="anthropic",
                     input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)
        span.set_io(input=question, output=text)
    return text


if __name__ == "__main__":
    print(ask("Explique em duas frases o que é observabilidade de LLMs."))
    lens.flush()
