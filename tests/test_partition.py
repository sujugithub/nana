"""S3: task partition invariants and collection verification (offline)."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

from tests.util import install_network_guard

install_network_guard()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import check_collection as cc  # noqa: E402
import partition_tasks as pt  # noqa: E402
from config import settings  # noqa: E402
from routing.identity import canonical_file_sha256  # noqa: E402

INPUTS = [os.path.join(REPO, "tasks", name) for name in
          ("real_pilot_qwen_pro.json", "real_extension_fast_qwen_pro.json")]
PRIOR = os.path.join(REPO, "data", "real_qwen_pro_600.json")
PAIR_V2 = os.path.join(REPO, "tasks", "pair_v2")


def _load(name):
    with open(os.path.join(PAIR_V2, name), encoding="utf-8") as fh:
        return json.load(fh)


class TestCommittedPartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = pt.load_tasks(INPUTS)
        cls.exposed = pt.prior_test_ids(PRIOR)
        cls.parts = {name: _load(f"{name}.json")
                     for name in ("pilot", "development_remaining", "final_test")}
        cls.manifest = _load("manifest.json")

    def test_regenerates_identically(self):
        fresh = pt.partition(self.tasks, self.exposed)
        self.assertEqual(fresh, self.parts)

    def test_parts_are_disjoint_and_complete(self):
        ids = [str(t["task_id"]) for rows in self.parts.values() for t in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {str(t["task_id"]) for t in self.tasks})
        self.assertEqual([len(self.parts[k]) for k in
                          ("pilot", "development_remaining", "final_test")], [60, 390, 150])

    def test_final_test_isolated_from_development(self):
        groups = pt.effective_groups(self.tasks)
        final = {groups[str(t["task_id"])] for t in self.parts["final_test"]}
        dev = {groups[str(t["task_id"])] for name in ("pilot", "development_remaining")
               for t in self.parts[name]}
        self.assertEqual(final & dev, set())  # no group, incl. near-dups, straddles
        self.assertEqual(len(self.exposed), 90)
        self.assertFalse({str(t["task_id"]) for t in self.parts["final_test"]} & self.exposed)

    def test_compositions(self):
        from collections import Counter

        self.assertEqual(Counter(t["source"] for t in self.parts["pilot"]), pt.PILOT_QUOTA)
        self.assertEqual(Counter(t["source"] for t in self.parts["final_test"]),
                         pt.FINAL_TEST_QUOTA)
        self.assertEqual(Counter(t["category"] for t in self.parts["final_test"]),
                         {"math": 50, "knowledge": 50, "reasoning": 50})

    def test_manifest_fingerprints_and_licences(self):
        for name in self.parts:
            self.assertEqual(self.manifest["parts"][name]["canonical_sha256"],
                             canonical_file_sha256(os.path.join(PAIR_V2, f"{name}.json")))
        licences = {s["name"]: s["license"] for s in self.manifest["sources"]}
        self.assertEqual(licences, {"GSM8K": "MIT", "MMLU": "MIT", "BIG-Bench Hard": "MIT"})
        bbh = next(s for s in self.manifest["sources"] if s["name"] == "BIG-Bench Hard")
        self.assertEqual(bbh["upstream"]["license"], "Apache-2.0")
        self.assertIn("RESERVED", self.manifest["use"]["final_test"])

    def test_refuses_to_overwrite(self):
        with redirect_stderr(io.StringIO()):
            self.assertEqual(pt.main(["--tasks", INPUTS[0], "--out-dir", PAIR_V2]), 2)


class TestCheckCollection(unittest.TestCase):
    _FIELDS = ("mock_mode", "tier_mode", "local_backend", "local_model_name",
               "local_context_tokens", "fireworks_api_key")

    def setUp(self):
        self._saved = {f: getattr(settings, f) for f in self._FIELDS}
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        for f, v in self._saved.items():
            setattr(settings, f, v)
        self.tmp.cleanup()

    def _mock_collection(self):
        from scripts import collect_outcomes as co

        settings.mock_mode, settings.tier_mode = True, "remote_pair"
        tasks = os.path.join(self.tmp.name, "tasks.json")
        with open(tasks, "w", encoding="utf-8") as fh:
            json.dump([{"task_id": t, "prompt": f"Name a colour ({t}).",
                        "reference": "answer to", "grader": "contains"} for t in "abc"], fh)
        out = os.path.join(self.tmp.name, "out.json")
        args = SimpleNamespace(
            tasks=tasks, out=out, grades=None, quality_threshold=0.6,
            local_cost_per_second=2e-5, max_consecutive_failures=3,
            local_context_tokens=None,
            cheap_in_per_mtok=1.0, cheap_out_per_mtok=1.0,
            strong_in_per_mtok=1.0, strong_out_per_mtok=1.0)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(co.collect(args), 0)
        return out, tasks

    def test_verify_accepts_consistent_and_rejects_tampering(self):
        out, tasks = self._mock_collection()
        result = cc.verify(out, tasks)
        self.assertEqual(result["problems"], [])
        self.assertTrue(result["synthetic"])
        self.assertEqual(len(result["files"]), 4)

        with open(out, encoding="utf-8") as fh:
            dataset = json.load(fh)
        dataset["meta"]["pair_identity"]["strong"]["model"] = "other"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(dataset, fh)
        self.assertIn("dataset pair_identity differs from the manifest",
                      cc.verify(out, tasks)["problems"])

        with open(out + ".work.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"task_id": "a", "run_id": "nope", "attempt_id": "x",
                                 "tier_mode": "remote_pair"}) + "\n")
        problems = " ".join(cc.verify(out, tasks)["problems"])
        self.assertIn("runs not in the manifest", problems)
        self.assertIn("summary attempts_total", problems)

    def test_checksum_file_matches_exact_bytes(self):
        out, tasks = self._mock_collection()
        checksums = os.path.join(self.tmp.name, "sums.sha256")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cc.main(["verify", "--out", out, "--tasks", tasks,
                                      "--write-checksums", checksums]), 0)
        with open(checksums, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        self.assertEqual(len(lines), 4)
        digest, name = lines[0].split("  ")
        self.assertEqual(digest, cc.file_sha256(os.path.join(self.tmp.name, name)))

    def test_identity_report_never_guesses_or_leaks(self):
        import ollama_client

        settings.mock_mode = False
        settings.tier_mode, settings.local_backend = "local_remote", "ollama"
        settings.local_model_name, settings.local_context_tokens = "fixture:27b", 8192
        settings.fireworks_api_key = "unit-test-placeholder-value"
        replies = {"/api/show": {"details": {"quantization_level": "Q4_K_M"},
                                 "template": "T", "parameters": "p"},
                   "/api/tags": {"models": [{"name": "fixture:27b", "digest": "abc"}]},
                   "/api/version": {"version": "0.0-test"}}
        with mock.patch.object(ollama_client, "request",
                               side_effect=lambda path, *a, **k: replies[path]):
            report = cc.identity_report()
        self.assertEqual(report["pair_identity"]["cheap"]["digest"], "abc")
        self.assertEqual(report["unknown_fields"], [])
        self.assertTrue(report["settings"]["FIREWORKS_API_KEY_SET"])
        self.assertNotIn("unit-test-placeholder-value", json.dumps(report))

        with mock.patch.object(ollama_client, "request",
                               side_effect=RuntimeError("Ollama is unavailable")):
            down = cc.identity_report()
        self.assertIn("cheap.digest", down["unknown_fields"])
        self.assertIn("RuntimeError", down["probe_error"])


if __name__ == "__main__":
    unittest.main()
