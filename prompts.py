"""Prompts for HyDE: generate a hypothetical document, then answer from real ones.

The hypothetical-document prompt is the heart of the method (Gao et al., 2022): ask
the model to *write the answer* as if quoting a reference, so the text carries the
vocabulary of relevant documents — even details that may be wrong, since retrieval
grounds it back to the real corpus.
"""

# Generate a hypothetical answer passage to use as the retrieval query.
HYDE_PROMPT = """Write a short, factual passage (2-3 sentences) that directly answers the \
question, as if it were an excerpt from an encyclopedia or technical reference. Use precise, \
domain-specific terminology. It is fine to be confident and specific even if unsure — this \
passage is used only to improve search, not shown to the user.

Question: {question}

Passage:"""

# Answer the user's question from the *real* retrieved passages, with citations.
ANSWER_PROMPT = """Answer the question using ONLY the passages below. Keep it to one sentence.
If the passages do not contain the answer, reply with exactly: INSUFFICIENT

Question: {question}

Passages:
{context}

End your answer by citing the passage number(s) in square brackets, e.g. [1].
Answer:"""

__all__ = ["HYDE_PROMPT", "ANSWER_PROMPT"]
