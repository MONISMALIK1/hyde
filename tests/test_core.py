"""The HyDE pipeline end to end, with real BM25 + a scripted generator.

The headline test: a query phrased in everyday words shares no terms with the
relevant document, so raw BM25 misses it — but retrieving with a (scripted)
hypothetical answer that uses the document's vocabulary surfaces it. That is the
whole point of HyDE, shown offline.
"""

import unittest

from hyde.core import answer, expand_query, hyde_search
from hyde.corpus import Document
from hyde.retriever import BM25Retriever


def make_fake(hypothetical, answer_text="You handle it with try/except. [1]"):
    def fake(prompt, model=None, **kwargs):
        if "Write a short" in prompt:            # HYDE_PROMPT
            return hypothetical
        if "using ONLY the passages" in prompt:  # ANSWER_PROMPT
            return answer_text
        return "?"
    return fake


class HydeRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document("exc", "Exception handling",
                     "In Python you handle runtime errors using try and except blocks "
                     "to catch an exception."),
            Document("gw", "Great Wall",
                     "The Great Wall of China is thousands of kilometres long."),
        ]
        self.retr = BM25Retriever(self.docs)
        self.query = "How do I stop my program from crashing on bad input?"
        self.hyp = ("To stop a program from crashing, wrap risky code in a try except "
                    "block to catch the exception and handle runtime errors in Python.")

    def test_baseline_misses_vocabulary_mismatch(self):
        base = self.retr.search(self.query, k=2)
        self.assertNotIn("exc", {h.doc.id for h in base})

    def test_hyde_surfaces_the_doc(self):
        hits, hyps, expanded = hyde_search(self.query, self.retr, chat_fn=make_fake(self.hyp))
        self.assertEqual(hits[0].doc.id, "exc")
        self.assertEqual(hyps, [self.hyp])
        self.assertIn("except", expanded)

    def test_answer_is_grounded_via_hyde(self):
        res = answer(self.query, self.retr, chat_fn=make_fake(self.hyp))
        self.assertFalse(res.abstained)
        self.assertTrue(res.used_hyde)
        self.assertIn("try/except", res.answer)
        self.assertEqual(res.retrieved[0].doc.id, "exc")

    def test_abstains_when_hyde_also_misses(self):
        res = answer(self.query, self.retr, chat_fn=make_fake("Bananas are a yellow fruit."))
        self.assertTrue(res.abstained)
        self.assertIn("even with HyDE", res.reason)


class BaselineTests(unittest.TestCase):
    def test_no_hyde_uses_raw_query_and_no_generation(self):
        docs = [Document("exc", "Exceptions", "Use try and except to catch an exception.")]
        retr = BM25Retriever(docs)

        def fake(prompt, model=None, **kwargs):
            if "Write a short" in prompt:
                raise AssertionError("HyDE generation should not run when use_hyde=False")
            return "Use try/except. [1]"

        res = answer("how to catch an exception", retr, use_hyde=False, chat_fn=fake)
        self.assertFalse(res.used_hyde)
        self.assertEqual(res.hypotheticals, [])
        self.assertFalse(res.abstained)


class ExpandTests(unittest.TestCase):
    def test_expand_query_pools_text(self):
        self.assertEqual(expand_query("q", ["a a", "b"]), "q a a b")

    def test_expand_query_no_hypotheticals(self):
        self.assertEqual(expand_query("just the query", []), "just the query")


if __name__ == "__main__":
    unittest.main()
