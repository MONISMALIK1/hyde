"""BM25 ranking basics."""

import unittest

from hyde.corpus import Document
from hyde.retriever import BM25Retriever, tokenize


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize("Try/except, in Python 3!"),
                         ["try", "except", "in", "python", "3"])


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.retr = BM25Retriever([
            Document("a", "Cats", "Cats are small domesticated felines."),
            Document("b", "Dogs", "Dogs are loyal domesticated canines."),
        ])

    def test_ranks_relevant_first(self):
        hits = self.retr.search("domesticated felines", k=2)
        self.assertEqual(hits[0].doc.id, "a")

    def test_offtopic_returns_empty(self):
        self.assertEqual(self.retr.search("quantum chromodynamics", k=2), [])

    def test_respects_k(self):
        self.assertLessEqual(len(self.retr.search("domesticated", k=1)), 1)


if __name__ == "__main__":
    unittest.main()
