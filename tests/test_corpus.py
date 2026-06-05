"""The bundled corpus + eval set."""

import unittest

from hyde.corpus import CORPUS, EVAL_QUESTIONS, matches, normalize


class CorpusTests(unittest.TestCase):
    def test_unique_ids(self):
        ids = [d.id for d in CORPUS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_eval_gold_id_exists(self):
        ids = {d.id for d in CORPUS}
        for ex in EVAL_QUESTIONS:
            self.assertIn(ex.gold_id, ids, ex.question)


class MatchTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("  Try   EXCEPT "), "try except")

    def test_matches_substring(self):
        ex = EVAL_QUESTIONS[0]  # answer "exception"
        self.assertTrue(matches("Use a try/except to catch an exception.", ex))
        self.assertFalse(matches("Reboot the computer.", ex))

    def test_no_match_on_none(self):
        self.assertFalse(matches(None, EVAL_QUESTIONS[0]))


if __name__ == "__main__":
    unittest.main()
