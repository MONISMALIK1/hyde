# hyde

[![tests](https://github.com/MONISMALIK1/hyde/actions/workflows/test.yml/badge.svg)](https://github.com/MONISMALIK1/hyde/actions/workflows/test.yml)

A from-scratch, dependency-free implementation of **HyDE** — *Hypothetical Document
Embeddings*. Instead of searching with the user's short query, HyDE first asks the
LLM to **write a hypothetical answer document**, then retrieves real documents using
*that*. The hypothetical text carries the relevant document's vocabulary even when
the user's wording does not — so it finds documents the raw query would miss.

> Gao, Ma, Lin, Callan (2022), *Precise Zero-Shot Dense Retrieval without Relevance
> Labels.* [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

## The idea in one picture

```
        query: "Why does my money buy less than it used to?"
           │
           ▼  ask the LLM to WRITE the answer (it may be wrong — that's fine)
   hypothetical: "Inflation is the sustained rise in the general price level,
                  which reduces the purchasing power of money over time…"
           │
           ▼  search with the hypothetical (its words match the real doc)
   retrieved: [Inflation] ✓   ← the raw query shares no terms with this doc
           │
           ▼  answer from the REAL retrieved passages, with citations
```

The hypothetical document is a throwaway: it's never shown to the user and may
contain false details. Its only job is to land near the right documents; retrieval
grounds everything back to the real corpus.

## Faithful technique, pragmatic adaptation

The paper embeds the hypothetical document with a **dense encoder** (Contriever) and
matches by meaning. This series is dependency-free, so here retrieval is **BM25** —
making this *lexical HyDE*: the hypothetical answer's keywords expand the query so a
lexical index can match documents the everyday-worded query misses. Same method and
same control flow; swapping in a dense retriever leaves the HyDE pipeline unchanged.
The paper also samples **N** hypotheticals and averages their embeddings; here we
pool their text (`--n-hyde`, which needs a sampling backend to actually differ).

## How HyDE differs from the other RAG repos

| | retrieve with… | reflection |
| --- | --- | --- |
| [rag](https://github.com/MONISMALIK1/rag) | raw query | none |
| **hyde** (this repo) | **a hypothetical answer document** | none — it improves the *query* |
| [self_rag](https://github.com/MONISMALIK1/self_rag) | raw query | grade & **abstain** |
| [corrective_rag](https://github.com/MONISMALIK1/corrective_rag) | raw query | grade & **correct** |

Self-RAG and CRAG fix things *after* retrieval; HyDE fixes the retrieval *query*
itself — so it composes with them.

## Install

No third-party dependencies. Python 3.11+.

```bash
git clone https://github.com/MONISMALIK1/hyde.git
cd hyde && pip install -e .      # optional; or just run from the parent dir
```

Point it at any OpenAI-compatible backend:

```bash
export OPENROUTER_API_KEY=sk-or-...                                   # OpenRouter (default)
# …or a local model:
export HYDE_BASE_URL=http://localhost:11434/v1/chat/completions       # Ollama
export HYDE_MODEL=qwen2.5:7b
```

## Use

```bash
# HyDE retrieval (default)
python -m hyde "How do I stop my program from crashing on bad input?"

# show the hypothetical document and what it retrieved
python -m hyde "Why does my money buy less than it used to?" --show-trace

# retrieve with the raw query instead — the baseline HyDE beats
python -m hyde "..." --no-hyde

# compare retrieval@k of the raw query vs HyDE on the bundled eval set
python -m hyde --bench
```

## Design

Query expansion and retrieval are pure stdlib and unit-tested offline; only the
hypothetical-document and answer generations touch the network.

| Module | Responsibility |
| --- | --- |
| `retriever.py` | from-scratch **BM25** over an in-memory corpus (pure, deterministic) |
| `prompts.py` | the hypothetical-document prompt + grounded-answer prompt |
| `generate.py` | the **HyDE step** — generate N hypothetical answer passages |
| `core.py` | expand the query, retrieve, answer (or abstain) → `HyDEResult` |
| `corpus.py` | a vocabulary-mismatch corpus + eval set (where HyDE beats the raw query) |
| `llm.py` | backend-agnostic OpenAI-compatible client (OpenRouter or local) |

## Test

```bash
make test        # or: python -m unittest discover -s hyde/tests -t . -v
```

24 offline tests, no API key required. The headline test shows the whole point of
HyDE: a query phrased in everyday words shares no terms with the relevant document
(so raw BM25 misses it), but retrieving with a scripted hypothetical answer that
uses the document's vocabulary surfaces it.

## Limitations

- **Hypothetical quality is the base model's.** A weak model writes a vague
  hypothetical that doesn't carry the right vocabulary, and HyDE gains little.
- **Lexical, not dense.** BM25 matches words; a paraphrase in the hypothetical that
  avoids the corpus's exact terms won't help. A dense retriever is a drop-in swap.
- **Extra generation cost.** HyDE adds one (or `--n-hyde`) generation call before
  retrieval — worth it when raw retrieval is the bottleneck, wasteful when it isn't.

## License

MIT
