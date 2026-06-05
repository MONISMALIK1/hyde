"""The HyDE step: generate one or more hypothetical answer documents.

The paper samples N hypothetical documents (with temperature) and averages their
embeddings to reduce the influence of any single hallucinated detail. Lexically we
pool the passages instead (see ``core.expand_query``); generating N>1 needs a
sampling backend (temperature > 0) to actually differ.
"""

from __future__ import annotations

from .llm import chat
from .prompts import HYDE_PROMPT


def generate_hypotheticals(
    question: str,
    n: int = 1,
    chat_fn=chat,
    model: str | None = None,
    temperature: float = 1.0,
) -> list[str]:
    """Return up to ``n`` hypothetical answer passages for ``question``."""
    out: list[str] = []
    prompt = HYDE_PROMPT.format(question=question)
    for _ in range(max(1, n)):
        text = chat_fn(prompt, model=model, temperature=temperature).strip()
        if text:
            out.append(text)
    return out


__all__ = ["generate_hypotheticals"]
