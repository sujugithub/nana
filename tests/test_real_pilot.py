"""Integrity checks for the tracked real-outcome pilot task set."""
from __future__ import annotations

import json
import os
import unittest
from collections import Counter

from scripts.collect_outcomes import grade
from routing.dataset import load_dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(REPO, "tasks", "real_pilot_qwen_pro.json")
MANIFEST = os.path.join(REPO, "tasks", "real_pilot_qwen_pro.manifest.json")
EXTENSION_TASKS = os.path.join(
    REPO, "tasks", "real_extension_fast_qwen_pro.json"
)
EXTENSION_MANIFEST = os.path.join(
    REPO, "tasks", "real_extension_fast_qwen_pro.manifest.json"
)
COMBINED_DATASET = os.path.join(REPO, "data", "real_qwen_pro_600.json")


class TestRealPilotIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(TASKS) as fh:
            cls.tasks = json.load(fh)
        with open(MANIFEST) as fh:
            cls.manifest = json.load(fh)

    def test_balanced_120_task_pilot(self):
        self.assertEqual(len(self.tasks), 120)
        self.assertEqual(
            Counter(t["category"] for t in self.tasks),
            {"math": 40, "knowledge": 40, "reasoning": 40},
        )

    def test_ids_groups_and_prompts_are_unique(self):
        for field in ("task_id", "group_id"):
            values = [t[field] for t in self.tasks]
            self.assertEqual(len(values), len(set(values)))
        prompts = [" ".join(t["prompt"].lower().split()) for t in self.tasks]
        self.assertEqual(len(prompts), len(set(prompts)))

    def test_every_reference_is_accepted_by_its_grader(self):
        for task in self.tasks:
            with self.subTest(task_id=task["task_id"]):
                self.assertEqual(
                    grade(task["grader"], str(task["reference"]),
                          str(task["reference"])),
                    1.0,
                )

    def test_manifest_binds_intended_model_pair(self):
        self.assertEqual(self.manifest["task_count"], len(self.tasks))
        self.assertEqual(
            self.manifest["intended_pair"],
            {
                "tier_mode": "local_remote",
                "cheap": "Qwen/Qwen2.5-1.5B-Instruct",
                "strong": "accounts/fireworks/models/deepseek-v4-pro",
            },
        )
        self.assertEqual(
            {source["license"] for source in self.manifest["sources"]},
            {"MIT"},
        )
        self.assertTrue(
            all(source.get("input_sha256") for source in self.manifest["sources"])
        )


class TestRealPilotExtensionIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(TASKS) as fh:
            cls.pilot = json.load(fh)
        with open(EXTENSION_TASKS) as fh:
            cls.tasks = json.load(fh)
        with open(EXTENSION_MANIFEST) as fh:
            cls.manifest = json.load(fh)

    def test_balanced_disjoint_480_task_extension(self):
        self.assertEqual(len(self.tasks), 480)
        self.assertEqual(
            Counter(t["category"] for t in self.tasks),
            {"math": 160, "knowledge": 160, "reasoning": 160},
        )
        old_prompts = {" ".join(t["prompt"].lower().split()) for t in self.pilot}
        new_prompts = {" ".join(t["prompt"].lower().split()) for t in self.tasks}
        self.assertFalse(old_prompts & new_prompts)

    def test_extension_manifest_binds_excluded_pilot(self):
        self.assertEqual(self.manifest["task_count"], len(self.tasks))
        self.assertEqual(
            self.manifest["prompt_protocol"], {"math_answer_only": True}
        )
        self.assertEqual(
            self.manifest["excluded_tasks"]["count"], len(self.pilot)
        )
        self.assertTrue(self.manifest["excluded_tasks"]["sha256"])


class TestCombinedRealDatasetIntegrity(unittest.TestCase):
    def test_combined_dataset_is_valid_pair_specific_600(self):
        dataset = load_dataset(COMBINED_DATASET)
        self.assertEqual(len(dataset.records), 600)
        self.assertEqual(dataset.content_hash(), "30c51583d6c8ab07")
        self.assertEqual(
            dataset.meta["local_model"], "Qwen/Qwen2.5-1.5B-Instruct"
        )
        self.assertEqual(
            dataset.meta["remote_model"],
            "accounts/fireworks/models/deepseek-v4-pro",
        )
        self.assertEqual(
            Counter(record.category for record in dataset.records),
            {"math": 200, "knowledge": 200, "reasoning": 200},
        )


if __name__ == "__main__":
    unittest.main()
