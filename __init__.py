"""HyDE — Hypothetical Document Embeddings, from scratch and dependency-free.

Improve retrieval by asking the LLM to write a hypothetical answer document, then
searching with *that* instead of the raw query — so the search query carries the
relevant document's vocabulary even when the user's wording does not.

Reference: Gao, Ma, Lin, Callan, 2022, "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (HyDE), https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

__version__ = "0.1.0"

from .corpus import CORPUS, EVAL_QUESTIONS, Document, QAExample, matches, normalize
from .core import HyDEResult, answer, default_retriever, expand_query, hyde_search
from .generate import generate_hypotheticals
from .retriever import BM25Retriever, Retrieved, tokenize

__all__ = [
    "__version__",
    "Document", "QAExample", "CORPUS", "EVAL_QUESTIONS", "matches", "normalize",
    "HyDEResult", "answer", "default_retriever", "expand_query", "hyde_search",
    "generate_hypotheticals",
    "BM25Retriever", "Retrieved", "tokenize",
]
