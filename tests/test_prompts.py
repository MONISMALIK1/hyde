"""Prompt template sanity checks."""

import unittest

from hyde.prompts import ANSWER_PROMPT, HYDE_PROMPT


class PromptTests(unittest.TestCase):
    def test_hyde_prompt_has_question_placeholder(self):
        self.assertIn("{question}", HYDE_PROMPT)
        self.assertIn("Write a short", HYDE_PROMPT)

    def test_answer_prompt_has_placeholders(self):
        self.assertIn("{question}", ANSWER_PROMPT)
        self.assertIn("{context}", ANSWER_PROMPT)
        self.assertIn("INSUFFICIENT", ANSWER_PROMPT)


if __name__ == "__main__":
    unittest.main()
