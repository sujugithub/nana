"""Dataset format: validation, label consistency, versioning, round-trip."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest

from tests.util import make_test_dataset  # noqa: F401  (installs sys.path + guard)

from routing.dataset import (
    DatasetError,
    OutcomeDataset,
    default_group_id,
    load_dataset,
    normalize_prompt,
    save_dataset,
    validate_dataset,
)


def _valid_raw() -> dict:
    return {
        "schema_version": "1.0",
        "meta": {
            "local_model": "l",
            "remote_model": "r",
            "quality_threshold": 0.6,
        },
        "records": [
            {
                "task_id": "a",
                "prompt": "What is two plus two?",
                "local_quality": 0.9,
                "remote_quality": 0.95,
                "local_ok": True,
            },
            {
                "task_id": "b",
                "prompt": "Prove the Riemann hypothesis.",
                "local_quality": 0.1,
                "remote_quality": 0.8,
                "local_ok": False,
            },
        ],
    }


class TestValidation(unittest.TestCase):
    def test_valid_dataset_has_no_problems(self):
        self.assertEqual(validate_dataset(_valid_raw()), [])

    def test_schema_version_mismatch_rejected(self):
        raw = _valid_raw()
        raw["schema_version"] = "0.9"
        problems = validate_dataset(raw)
        self.assertTrue(any("schema_version" in p for p in problems))

    def test_missing_required_meta(self):
        raw = _valid_raw()
        del raw["meta"]["quality_threshold"]
        self.assertTrue(validate_dataset(raw))

    def test_duplicate_task_ids_rejected(self):
        raw = _valid_raw()
        raw["records"][1]["task_id"] = "a"
        self.assertTrue(any("duplicate" in p for p in validate_dataset(raw)))

    def test_quality_out_of_range_rejected(self):
        raw = _valid_raw()
        raw["records"][0]["local_quality"] = 1.5
        raw["records"][0]["local_ok"] = True
        self.assertTrue(any("[0, 1]" in p for p in validate_dataset(raw)))

    def test_label_must_match_quality_threshold(self):
        raw = _valid_raw()
        raw["records"][1]["local_ok"] = True  # contradicts quality 0.1 vs 0.6
        problems = validate_dataset(raw)
        self.assertTrue(any("contradicts" in p for p in problems))

    def test_empty_prompt_rejected(self):
        raw = _valid_raw()
        raw["records"][0]["prompt"] = "   "
        self.assertTrue(validate_dataset(raw))

    def test_negative_cost_rejected(self):
        raw = _valid_raw()
        raw["records"][0]["local_cost"] = -1.0
        self.assertTrue(any(">= 0" in p for p in validate_dataset(raw)))


class TestLoadSave(unittest.TestCase):
    def test_round_trip(self):
        dataset = make_test_dataset(n=20)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.json")
            save_dataset(dataset, path)
            loaded = load_dataset(path)
        self.assertEqual(len(loaded.records), 20)
        self.assertEqual(loaded.content_hash(), dataset.content_hash())

    def test_load_missing_file(self):
        with self.assertRaises(DatasetError):
            load_dataset("/nonexistent/nope.json")

    def test_load_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w") as fh:
                fh.write("{not json")
            with self.assertRaises(DatasetError):
                load_dataset(path)

    def test_load_invalid_content_lists_problems(self):
        raw = _valid_raw()
        raw["records"][0]["local_ok"] = False  # contradicts 0.9 >= 0.6
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w") as fh:
                json.dump(raw, fh)
            with self.assertRaises(DatasetError) as ctx:
                load_dataset(path)
        self.assertIn("contradicts", str(ctx.exception))

    def test_save_refuses_invalid(self):
        dataset = make_test_dataset(n=10)
        broken = copy.deepcopy(dataset)
        broken.records[0].local_quality = 5.0
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DatasetError):
                save_dataset(broken, os.path.join(tmp, "x.json"))

    def test_content_hash_changes_with_content(self):
        a = make_test_dataset(n=10)
        b = make_test_dataset(n=10)
        b.records[0].local_quality = round(1 - b.records[0].local_quality, 3)
        b.records[0].local_ok = b.records[0].local_quality >= 0.6
        self.assertNotEqual(a.content_hash(), b.content_hash())


class TestNormalization(unittest.TestCase):
    def test_normalize_prompt(self):
        self.assertEqual(
            normalize_prompt("  What is   2+2?! "), normalize_prompt("what is 2 2")
        )

    def test_default_group_id_stable(self):
        self.assertEqual(default_group_id("Hello!"), default_group_id("hello"))
        self.assertNotEqual(default_group_id("Hello!"), default_group_id("bye"))


if __name__ == "__main__":
    unittest.main()
