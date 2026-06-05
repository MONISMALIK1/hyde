"""Hypothetical-document generation."""

import unittest

from hyde.generate import generate_hypotheticals


class GenerateTests(unittest.TestCase):
    def test_generates_n_documents(self):
        seen = []

        def fake(prompt, model=None, temperature=1.0):
            seen.append(temperature)
            return f"doc {len(seen)}"

        out = generate_hypotheticals("q", n=3, chat_fn=fake)
        self.assertEqual(out, ["doc 1", "doc 2", "doc 3"])
        self.assertTrue(all(t == 1.0 for t in seen))  # sampling temperature passed through

    def test_skips_empty_generations(self):
        def fake(prompt, model=None, **kwargs):
            return "   "
        self.assertEqual(generate_hypotheticals("q", n=2, chat_fn=fake), [])

    def test_at_least_one_attempt(self):
        calls = []

        def fake(prompt, model=None, **kwargs):
            calls.append(1)
            return "x"
        generate_hypotheticals("q", n=0, chat_fn=fake)  # n<1 -> still one attempt
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
