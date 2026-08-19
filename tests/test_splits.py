"""Group-aware splitting: leakage prevention, duplicates, determinism."""
from __future__ import annotations

import unittest

from tests.util import make_test_dataset

from routing.dataset import OutcomeRecord
from routing.splits import (
    LeakageError,
    find_exact_duplicates,
    find_near_duplicates,
    make_splits,
)


def _rec(task_id, prompt, group_id, ok=True):
    return OutcomeRecord(
        task_id=task_id,
        prompt=prompt,
        group_id=group_id,
        local_quality=0.9 if ok else 0.1,
        remote_quality=0.9,
        local_ok=ok,
    )


class TestSplits(unittest.TestCase):
    def test_groups_never_cross_splits(self):
        dataset = make_test_dataset(n=200)
        splits = make_splits(dataset.records, seed=3)
        group_to_split = {}
        for rec in dataset.records:
            split = splits.assignment[rec.task_id]
            prior = group_to_split.setdefault(rec.group_id, split)
            self.assertEqual(
                prior, split, f"group {rec.group_id} crosses {prior}/{split}"
            )

    def test_all_records_assigned_and_ratios_respected(self):
        dataset = make_test_dataset(n=200)
        splits = make_splits(dataset.records, seed=3, ratios=(0.7, 0.15, 0.15))
        self.assertEqual(len(splits.assignment), 200)
        n_train = len(splits.ids("train"))
        self.assertGreater(n_train, 100)  # ~140 expected; grouping adds slack
        self.assertGreater(len(splits.ids("val")), 10)
        self.assertGreater(len(splits.ids("test")), 10)

    def test_deterministic_given_seed(self):
        dataset = make_test_dataset(n=100)
        a = make_splits(dataset.records, seed=3)
        b = make_splits(dataset.records, seed=3)
        self.assertEqual(a.assignment, b.assignment)

    def test_seed_changes_assignment(self):
        dataset = make_test_dataset(n=100)
        a = make_splits(dataset.records, seed=3)
        b = make_splits(dataset.records, seed=4)
        self.assertNotEqual(a.assignment, b.assignment)

    def test_bad_ratios_rejected(self):
        dataset = make_test_dataset(n=20)
        with self.assertRaises(ValueError):
            make_splits(dataset.records, ratios=(0.5, 0.2, 0.2))


class TestDuplicates(unittest.TestCase):
    def test_exact_duplicates_across_groups_is_error(self):
        records = [
            _rec("a", "What is the capital of France?", "g1"),
            _rec("b", "what is the capital of france", "g2"),  # same normalized
        ]
        self.assertEqual(find_exact_duplicates(records), [("a", "b")])
        with self.assertRaises(LeakageError):
            make_splits(records)

    def test_exact_duplicates_same_group_are_fine(self):
        records = [
            _rec("a", "What is the capital of France?", "g1"),
            _rec("b", "what is the capital of france", "g1"),
        ]
        self.assertEqual(find_exact_duplicates(records), [])

    def test_near_duplicates_get_merged(self):
        base = (
            "The quick brown fox jumps over the lazy dog near the quiet river "
            "bank on a warm summer evening in the north country"
        )
        records = [
            _rec("a", base + " one", "g1"),
            _rec("b", base + " two", "g2"),  # near-dup of a, different group
            _rec("c", "Completely unrelated short prompt about cheese.", "g3"),
        ] + [
            _rec(f"pad{i}", f"Unique padding prompt number {i} with topic {i}.", f"p{i}")
            for i in range(12)
        ]
        near = find_near_duplicates(records)
        self.assertTrue(any({p[0], p[1]} == {"a", "b"} for p in near))
        splits = make_splits(records, seed=1)
        self.assertEqual(
            splits.assignment["a"], splits.assignment["b"],
            "near-duplicates must land in the same split",
        )
        self.assertTrue(splits.merged_groups)


if __name__ == "__main__":
    unittest.main()
