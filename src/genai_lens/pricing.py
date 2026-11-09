"""Token pricing used to estimate cost at ingestion time.

Prices are USD per 1M tokens and change often: treat this table as a default and
override it with ``GENAI_LENS_PRICING_FILE`` (a JSON file with the same shape).
Models are matched by longest prefix, so ``gpt-4o-mini-2024-07-18`` matches
``gpt-4o-mini`` and not ``gpt-4o``. Local models (Ollama, vLLM) cost zero.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


DEFAULT_PRICES: dict[str, Price] = {
    "claude-opus": Price(15.0, 75.0),
    "claude-sonnet": Price(3.0, 15.0),
    "claude-haiku": Price(0.8, 4.0),
    "gpt-4o-mini": Price(0.15, 0.6),
    "gpt-4o": Price(2.5, 10.0),
    "gpt-4.1-mini": Price(0.4, 1.6),
    "gpt-4.1": Price(2.0, 8.0),
    "text-embedding-3-small": Price(0.02, 0.0),
    "llama": Price(0.0, 0.0),
    "qwen": Price(0.0, 0.0),
    "mistral": Price(0.0, 0.0),
}


class PriceTable:
    def __init__(self, prices: dict[str, Price] | None = None) -> None:
        self.prices = dict(prices or DEFAULT_PRICES)
        # longest prefix first
        self._keys = sorted(self.prices, key=len, reverse=True)

    @classmethod
    def from_env(cls) -> PriceTable:
        path = os.getenv("GENAI_LENS_PRICING_FILE")
        if not path:
            return cls()
        raw = json.loads(Path(path).read_text())
        prices = {k: Price(float(v["input_per_mtok"]), float(v["output_per_mtok"])) for k, v in raw.items()}
        return cls({**DEFAULT_PRICES, **prices})

    def lookup(self, model: str | None) -> Price | None:
        if not model:
            return None
        name = model.lower().split("/")[-1]  # "openai/gpt-4o" -> "gpt-4o"
        for key in self._keys:
            if name.startswith(key):
                return self.prices[key]
        return None

    def cost(self, model: str | None, input_tokens: int | None, output_tokens: int | None) -> float | None:
        price = self.lookup(model)
        if price is None or (input_tokens is None and output_tokens is None):
            return None
        return round(
            ((input_tokens or 0) * price.input_per_mtok + (output_tokens or 0) * price.output_per_mtok) / 1e6,
            8,
        )
