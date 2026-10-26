"""Regression tests for routing/grading.py and its collector integration.

Every grader error demonstrated in the Phase 1 audit has a test here, next
to legitimate answers that must stay accepted.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from tests.util import install_network_guard

install_network_guard()

from config import settings  # noqa: E402
from routing.grading import (  # noqa: E402
    AMBIGUOUS,
    CONTRADICTORY,
    EMPTY,
    GRADED,
    NEEDS_REVIEW,
    NO_FINAL_ANSWER,
    REFUSAL,
    THINKING_REMOVED,
    TRUNCATED,
    UNCLOSED_THINKING,
    grade_answer,
    grading_copy,
)
from scripts.collect_outcomes import grade  # noqa: E402


def choice(ref, answer, finish=None):
    return grade_answer("choice", ref, answer, finish)


def numeric(ref, answer, finish=None):
    return grade_answer("numeric", ref, answer, finish)


class TestChoiceRegressions(unittest.TestCase):
    """Old grader: whole-token search over the first 80 lowercased chars."""

    def test_article_a_is_not_option_a(self):
        text = "The answer is a bit unclear, but I think C."
        result = choice("A", text)
        self.assertEqual((result.status, result.score), (GRADED, 0.0))
        self.assertEqual(result.extracted, "C")
        self.assertEqual(choice("C", text).score, 1.0)

    def test_other_option_mentioned_needs_review(self):
        # Old grader: correct for B. No regex interprets the negation here.
        for ref in ("A", "B"):
            result = choice(ref, "A. Paris (not B)")
            self.assertEqual(result.status, NEEDS_REVIEW)
            self.assertIn(AMBIGUOUS, result.flags)

    def test_negated_truth_value_uses_explicit_answer(self):
        text = "It is not True; the answer is False."
        self.assertEqual(choice("True", text).score, 0.0)
        result = choice("False", text)
        self.assertEqual((result.status, result.score), (GRADED, 1.0))
        self.assertEqual(result.method, "final_marker")

    def test_legitimate_answers_still_accepted(self):
        cases = [
            ("B", "B"), ("B", "(B)"), ("B", "B."), ("B", "B) hydrogen"),
            ("B", "**B**"), ("B", "Answer: B"), ("C", "The correct answer is (C)."),
            ("D", "\\boxed{D}"), ("E", "E - Dan finished second"),
            ("True", "True"), ("False", "False."), ("Yes", "Yes"),
            ("No", "No, because the outcome would have happened anyway."),
            ("C", "I think C."), ("A", "(A) hydrogen"),
            ("Yes", "The answer is yes because the charge completed."),
        ]
        for ref, text in cases:
            with self.subTest(text=text):
                result = choice(ref, text)
                self.assertEqual((result.status, result.score), (GRADED, 1.0))

    def test_wrong_answers_graded_wrong_not_reviewed(self):
        for ref, text in [("B", "C"), ("B", "Answer: D"), ("True", "False"),
                          ("Yes", "No.")]:
            with self.subTest(text=text):
                result = choice(ref, text)
                self.assertEqual((result.status, result.score), (GRADED, 0.0))

    def test_conflicting_explicit_answers_are_contradictory(self):
        for text in ("Answer: A. On reflection, final answer: B",
                     "B. Checking again, the answer is C."):
            with self.subTest(text=text):
                result = choice("B", text)
                self.assertEqual(result.status, NEEDS_REVIEW)
                self.assertIn(CONTRADICTORY, result.flags)

    def test_sentence_starting_with_article_has_no_answer(self):
        result = choice("A", "A bakery cannot sell more than it bakes.")
        self.assertEqual(result.status, NEEDS_REVIEW)
        self.assertIn(NO_FINAL_ANSWER, result.flags)

    def test_prose_with_both_truth_words_needs_review(self):
        result = choice("Yes", "There is no clear cause, so yes.")
        self.assertEqual(result.status, NEEDS_REVIEW)
        self.assertIn(AMBIGUOUS, result.flags)


class TestNumericRegressions(unittest.TestCase):
    """Old grader: the last number anywhere in the answer."""

    def test_explicit_answer_beats_trailing_number(self):
        result = numeric("18", "The answer is 18 dollars, computed over 2 days.")
        self.assertEqual((result.status, result.score), (GRADED, 1.0))
        self.assertEqual(result.method, "final_marker")

    def test_fraction_parsed(self):
        self.assertEqual(numeric("0.5", "1/2").score, 1.0)
        self.assertEqual(numeric("0.5", "The answer is 1/2.").score, 1.0)

    def test_currency_and_separators(self):
        self.assertEqual(numeric("1000", "$1,000.00").score, 1.0)
        self.assertEqual(numeric("72000", "The answer is 72,000.").score, 1.0)
        self.assertEqual(numeric("72,000", "Final: 72000").score, 1.0)

    def test_trailing_final_number(self):
        text = "Step 1: ... Step 5: so x = 7"
        self.assertEqual(numeric("5", text).score, 0.0)
        result = numeric("7", text)
        self.assertEqual((result.score, result.method), (1.0, "trailing_number"))

    def test_marker_formats(self):
        for text in ("#### 42", "\\boxed{42}", "Final answer: 42",
                     "The answer is approximately 42.", "42", "42 apples."):
            with self.subTest(text=text):
                self.assertEqual(numeric("42", text).score, 1.0)
        self.assertEqual(numeric("-5", "The answer is -5.").score, 1.0)

    def test_expression_after_marker_is_not_the_answer(self):
        result = numeric("8", "Answer: 5 + 3 = 8")
        self.assertEqual((result.score, result.extracted), (1.0, "8"))

    def test_conflicting_explicit_answers_are_contradictory(self):
        result = numeric("42", "The answer is 40. Final answer: 42")
        self.assertEqual(result.status, NEEDS_REVIEW)
        self.assertIn(CONTRADICTORY, result.flags)

    def test_no_final_number_needs_review(self):
        text = "We know 3 apples and 4 pears are involved but need more detail."
        result = numeric("7", text)
        self.assertEqual(result.status, NEEDS_REVIEW)
        self.assertIn(NO_FINAL_ANSWER, result.flags)


class TestResponseFlags(unittest.TestCase):
    def test_empty_output_is_a_graded_failure(self):
        for text in ("", "   ", None):
            with self.subTest(text=text):
                result = numeric("4", text)
                self.assertEqual((result.status, result.score), (GRADED, 0.0))
                self.assertIn(EMPTY, result.flags)

    def test_refusal_is_flagged(self):
        for text in ("I'm sorry, but I can't help with that.",
                     "I can’t answer that question."):
            with self.subTest(text=text):
                result = choice("B", text)
                self.assertEqual((result.status, result.score), (GRADED, 0.0))
                self.assertIn(REFUSAL, result.flags)

    def test_truncated_without_answer_is_failure_with_flag(self):
        result = numeric("9", "Let me compute 4 + 5 and then multiply by", "length")
        self.assertEqual((result.status, result.score), (GRADED, 0.0))
        self.assertIn(TRUNCATED, result.flags)

    def test_truncated_with_explicit_answer_keeps_flag(self):
        result = numeric("9", "Answer: 9. To explain, first we", "length")
        self.assertEqual((result.status, result.score), (GRADED, 1.0))
        self.assertIn(TRUNCATED, result.flags)

    def test_thinking_removed_from_grading_copy_only(self):
        raw = "<think>maybe the answer is A</think>\nB"
        result = choice("B", raw)
        self.assertEqual((result.status, result.score), (GRADED, 1.0))
        self.assertIn(THINKING_REMOVED, result.flags)
        self.assertEqual(raw, "<think>maybe the answer is A</think>\nB")
        self.assertEqual(numeric("42", "<think>answer is 40</think>The answer is 42").score, 1.0)

    def test_unclosed_thinking_yields_no_answer(self):
        copy, flags = grading_copy("<think>still working on 42")
        self.assertEqual(copy, "")
        self.assertIn(UNCLOSED_THINKING, flags)
        self.assertIn(EMPTY, numeric("42", "<think>still working on 42").flags)

    def test_result_serialises(self):
        self.assertEqual(
            set(choice("B", "B").to_dict()),
            {"status", "score", "flags", "extracted", "method"},
        )

    def test_legacy_grade_interface(self):
        self.assertEqual(grade("choice", "B", "(B)"), 1.0)
        self.assertEqual(grade("choice", "B", "A. Paris (not B)"), 0.0)
        self.assertEqual(grade("contains", "paris", "It is Paris."), 1.0)
        self.assertEqual(grade("exact", "paris", " Paris "), 1.0)
        with self.assertRaises(ValueError):
            grade("fuzzy", "x", "x")


class TestCollectorRecordsGrades(unittest.TestCase):
    _FIELDS = ("mock_mode", "tier_mode")

    def setUp(self):
        self.saved = {f: getattr(settings, f) for f in self._FIELDS}

    def tearDown(self):
        for field, value in self.saved.items():
            setattr(settings, field, value)

    def _args(self, tmp, grades=None):
        return SimpleNamespace(
            tasks=os.path.join(tmp, "tasks.json"),
            out=os.path.join(tmp, "out.json"),
            grades=grades,
            quality_threshold=0.6,
            local_cost_per_second=2e-5,
            cheap_in_per_mtok=settings.cheap_input_per_mtok,
            cheap_out_per_mtok=settings.cheap_output_per_mtok,
            strong_in_per_mtok=settings.strong_input_per_mtok,
            strong_out_per_mtok=settings.strong_output_per_mtok,
        )

    def test_review_rows_keep_raw_answers_and_are_not_finalized(self):
        from scripts.collect_outcomes import collect, finalize, _load_existing

        settings.mock_mode = True
        settings.tier_mode = "remote_pair"
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp)
            with open(args.tasks, "w") as fh:
                json.dump([
                    {"task_id": "graded", "prompt": "Name a colour.",
                     "reference": "answer to", "grader": "contains"},
                    # Mock answers contain no number -> review, not "wrong".
                    {"task_id": "review", "prompt": "Name a colour.",
                     "reference": "4", "grader": "numeric"},
                ], fh)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(collect(args), 0)
            rows = _load_existing(args.out + ".work.jsonl")
            with open(args.out) as fh:
                dataset = json.load(fh)

            review = rows["review"]
            self.assertEqual(review["remote_grade"]["status"], NEEDS_REVIEW)
            self.assertEqual(review["remote_quality"], -1.0)
            self.assertTrue(review["remote_answer"].startswith("[mock-strong]"))
            self.assertEqual(rows["graded"]["remote_grade"]["status"], GRADED)
            self.assertEqual([r["task_id"] for r in dataset["records"]], ["graded"])

            # A one-sided verdict is enough when only that side was review.
            rows["review"]["local_quality"] = 0.0
            grades = os.path.join(tmp, "grades.json")
            with open(grades, "w") as fh:
                json.dump({"review": {"strong": 1.0}}, fh)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(finalize(self._args(tmp, grades), rows), 0)
            with open(args.out) as fh:
                ids = {r["task_id"]: r for r in json.load(fh)["records"]}
            self.assertEqual(ids["review"]["remote_quality"], 1.0)


if __name__ == "__main__":
    unittest.main()
