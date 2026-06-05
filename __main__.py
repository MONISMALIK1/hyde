"""CLI for HyDE.

Usage:
    # Answer a question (HyDE retrieval by default)
    python -m hyde "How do I stop my program from crashing on bad input?"

    # Show the hypothetical document and what it retrieved
    python -m hyde "Why does my money buy less than it used to?" --show-trace

    # Retrieve with the raw query instead (the baseline HyDE beats)
    python -m hyde "..." --no-hyde

    # Compare retrieval@k: raw query vs HyDE, on the bundled eval set
    python -m hyde --bench
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .core import answer, default_retriever, hyde_search
from .corpus import EVAL_QUESTIONS, matches
from .llm import DEFAULT_MODEL


def _print_trace(res) -> None:
    print("--- HyDE trace ---")
    if res.used_hyde and res.hypotheticals:
        print("Hypothetical document(s):")
        for h in res.hypotheticals:
            head = h.replace("\n", " ")
            print(f"  ~ {head[:100]}{'...' if len(head) > 100 else ''}")
    else:
        print("Hypothetical document(s): (none — raw-query baseline)")
    print("Retrieved:")
    for h in res.retrieved:
        print(f"  [{h.rank}] {h.doc.id:<14} bm25={h.score:<7} {h.doc.title}")
    print("------------------")


def _answer_one(args, retriever) -> int:
    res = answer(args.query, retriever, use_hyde=not args.no_hyde, n_hyde=args.n_hyde,
                 k=args.k, model=args.model)
    if args.show_trace:
        _print_trace(res)
    print("=" * 60)
    if res.abstained:
        print(f"I don't know — {res.reason}")
    else:
        print(res.answer)
        print(f"({res.reason})")
    return 0


def _bench(args, retriever) -> int:
    base_hit = hyde_hit = total = 0
    for i, ex in enumerate(EVAL_QUESTIONS, 1):
        base = retriever.search(ex.question, k=args.k)
        hyde, _, _ = hyde_search(ex.question, retriever, n_hyde=args.n_hyde, k=args.k,
                                 model=args.model)
        b = ex.gold_id in {h.doc.id for h in base}
        hh = ex.gold_id in {h.doc.id for h in hyde}
        base_hit += int(b)
        hyde_hit += int(hh)
        total += 1
        print(f"[{i}] gold={ex.gold_id:<14} raw={'HIT ' if b else 'miss'}  "
              f"hyde={'HIT ' if hh else 'miss'}   {ex.question}", flush=True)

    print("\n" + "=" * 64)
    print(f"HyDE retrieval@{args.k} on bundled eval set — model={args.model or DEFAULT_MODEL}")
    print("=" * 64)
    print(f"  raw query : {base_hit}/{total} gold docs retrieved")
    print(f"  HyDE      : {hyde_hit}/{total} gold docs retrieved")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="hyde",
        description="HyDE (Gao et al., 2022): generate a hypothetical answer document, "
                    "then retrieve real documents using it.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("query", nargs="?", help="The question to answer.")
    p.add_argument("--no-hyde", action="store_true",
                   help="Retrieve with the raw query (baseline) instead of HyDE.")
    p.add_argument("--n-hyde", type=int, default=1,
                   help="Hypothetical documents to generate and pool (default: 1).")
    p.add_argument("--k", type=int, default=4, help="Passages to retrieve (default: 4).")
    p.add_argument("--model", default=None, help=f"Model slug (default: {DEFAULT_MODEL}).")
    p.add_argument("--show-trace", action="store_true",
                   help="Print the hypothetical document and the retrieved passages.")
    p.add_argument("--bench", action="store_true",
                   help="Compare retrieval@k of the raw query vs HyDE on the eval set.")
    args = p.parse_args()

    retriever = default_retriever()

    if args.bench:
        return _bench(args, retriever)
    if not args.query:
        p.error("provide a question to answer, or use --bench")

    print(f"\nQuestion: {args.query}", file=sys.stderr)
    print(f"Model: {args.model or DEFAULT_MODEL}\n", file=sys.stderr, flush=True)
    return _answer_one(args, retriever)


if __name__ == "__main__":
    raise SystemExit(main())
