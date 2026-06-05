"""The HyDE pipeline: hypothesize -> retrieve with the hypothesis -> answer.

Reference: Gao, Ma, Lin, Callan 2022, "Precise Zero-Shot Dense Retrieval without
Relevance Labels", https://arxiv.org/abs/2212.10496

    question
       |
    generate a hypothetical answer document (the LLM)        <- the HyDE step
       |
    expand the query with that document's text
       |
    retrieve real documents with the expanded query (BM25)
       |
    answer from the real passages, with citations -- or abstain

Set ``use_hyde=False`` to retrieve with the raw query instead — the baseline HyDE
is meant to beat. Only the generation calls touch the network (injectable
``chat_fn``); query expansion and retrieval are pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .generate import generate_hypotheticals
from .llm import chat
from .prompts import ANSWER_PROMPT
from .retriever import BM25Retriever, Retrieved

_INSUFFICIENT = "INSUFFICIENT"


@dataclass
class HyDEResult:
    query: str
    answer: str | None
    abstained: bool
    reason: str
    used_hyde: bool
    hypotheticals: list[str] = field(default_factory=list)
    expanded_query: str = ""
    retrieved: list[Retrieved] = field(default_factory=list)


def expand_query(query: str, hypotheticals: list[str]) -> str:
    """Pool the raw query with the hypothetical passages (lexical HyDE)."""
    return " ".join([query, *hypotheticals]).strip()


def hyde_search(
    query: str,
    retriever: BM25Retriever,
    n_hyde: int = 1,
    k: int = 4,
    chat_fn=chat,
    model: str | None = None,
    temperature: float = 1.0,
) -> tuple[list[Retrieved], list[str], str]:
    """Generate hypothetical docs, expand the query, and retrieve. Returns
    (hits, hypotheticals, expanded_query)."""
    hyps = generate_hypotheticals(query, n=n_hyde, chat_fn=chat_fn, model=model,
                                  temperature=temperature)
    expanded = expand_query(query, hyps)
    return retriever.search(expanded, k=k), hyps, expanded


def _looks_insufficient(text: str) -> bool:
    return text.strip().upper().startswith(_INSUFFICIENT)


def _format_context(hits: list[Retrieved]) -> str:
    return "\n".join(f"[{i}] ({h.doc.title}) {h.doc.text}" for i, h in enumerate(hits, 1))


def answer(
    query: str,
    retriever: BM25Retriever,
    use_hyde: bool = True,
    n_hyde: int = 1,
    k: int = 4,
    max_context: int = 4,
    chat_fn=chat,
    model: str | None = None,
) -> HyDEResult:
    """Answer ``query`` against ``retriever``, optionally using HyDE retrieval."""
    if use_hyde:
        hits, hyps, expanded = hyde_search(query, retriever, n_hyde=n_hyde, k=k,
                                           chat_fn=chat_fn, model=model)
    else:
        hits, hyps, expanded = retriever.search(query, k=k), [], query

    if not hits:
        suffix = " (even with HyDE)" if use_hyde else ""
        return HyDEResult(query=query, answer=None, abstained=True,
                          reason=f"no documents matched the query{suffix}",
                          used_hyde=use_hyde, hypotheticals=hyps, expanded_query=expanded)

    context = _format_context(hits[:max_context])
    raw = chat_fn(ANSWER_PROMPT.format(question=query, context=context), model=model).strip()
    if _looks_insufficient(raw) or not raw:
        return HyDEResult(query=query, answer=None, abstained=True,
                          reason="retrieved passages did not contain the answer",
                          used_hyde=use_hyde, hypotheticals=hyps, expanded_query=expanded,
                          retrieved=hits)

    how = "via HyDE" if use_hyde else "baseline (raw query)"
    return HyDEResult(query=query, answer=raw, abstained=False,
                      reason=f"answered from {len(hits[:max_context])} passage(s), {how}",
                      used_hyde=use_hyde, hypotheticals=hyps, expanded_query=expanded,
                      retrieved=hits)


def default_retriever() -> BM25Retriever:
    from .corpus import CORPUS
    return BM25Retriever(CORPUS)


__all__ = ["HyDEResult", "expand_query", "hyde_search", "answer", "default_retriever"]
