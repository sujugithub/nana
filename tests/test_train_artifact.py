"""Training: reproducibility, artifact round-trip, failure modes."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from tests.util import make_test_dataset, trained_once

from routing import ARTIFACT_FORMAT_VERSION
from routing.artifact import ArtifactError, load_artifact, save_artifact
from routing.train import train


class TestTraining(unittest.TestCase):
    def test_end_to_end_train_produces_usable_artifact(self):
        dataset, (artifact, report, splits, candidates) = trained_once()
        p = artifact.predict_p_local(["What is the capital of France?"])
        self.assertEqual(p.shape, (1,))
        self.assertTrue(0.0 <= p[0] <= 1.0)
        self.assertEqual(report["selected_candidate"], artifact.model_name)
        self.assertIn("logreg_tfidf", report["candidates"])
        self.assertIn("heuristic_reference", report["candidates"])

    def test_learned_beats_heuristic_on_val_calibration(self):
        # The test dataset has a clean lexical signal; if the learned router
        # cannot at least MATCH the keyword rules' ranking and clearly beat
        # their calibration here, the pipeline is broken. (The val split is
        # small, so ranking can tie at 1.0 — Brier/ECE cannot.)
        _, (_, report, _, _) = trained_once()
        learned = report["candidates"]["logreg_tfidf"]["val_classification"]
        heur = report["candidates"]["heuristic_reference"]["val_classification"]
        self.assertGreaterEqual(learned["roc_auc"], heur["roc_auc"])
        self.assertLess(learned["brier"], heur["brier"])
        self.assertLess(learned["ece"], heur["ece"])

    def test_reproducible_given_seed(self):
        dataset = make_test_dataset()
        probe = [r.prompt for r in dataset.records[:10]]
        a, _, _, _ = train(dataset, seed=3, candidates=["logreg_tfidf"])
        b, _, _, _ = train(dataset, seed=3, candidates=["logreg_tfidf"])
        self.assertEqual(a.threshold, b.threshold)
        self.assertEqual(a.version, b.version)
        np.testing.assert_allclose(
            a.predict_p_local(probe), b.predict_p_local(probe), atol=1e-9
        )

    def test_single_class_train_split_rejected(self):
        dataset = make_test_dataset(n=120)
        for rec in dataset.records:
            rec.local_quality = 0.9
            rec.local_ok = True
        with self.assertRaises(ValueError):
            train(dataset, seed=3, candidates=["logreg_tfidf"])

    def test_too_small_dataset_rejected(self):
        dataset = make_test_dataset(n=30)
        with self.assertRaises(ValueError):
            train(dataset, seed=3, candidates=["logreg_tfidf"])

    def test_report_records_provenance(self):
        dataset, (_, report, _, _) = trained_once()
        self.assertEqual(report["dataset_hash"], dataset.content_hash())
        self.assertEqual(report["seed"], 3)
        self.assertIn("split_strategy", report)
        self.assertEqual(
            sum(report["split_strategy"]["sizes"].values()), len(dataset.records)
        )

    def test_postgen_analysis_present(self):
        _, (_, report, _, _) = trained_once()
        self.assertIn("recommended", report["postgen"])
        self.assertEqual(
            set(report["postgen"]["auc_by_stat"]), {"mean", "min", "low_frac"}
        )


class TestArtifactIO(unittest.TestCase):
    def test_round_trip_preserves_predictions(self):
        _, (artifact, _, _, _) = trained_once()
        probe = ["Prove that 17 is prime.", "What is the capital of Peru?"]
        expected = artifact.predict_p_local(probe)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.joblib")
            save_artifact(artifact, path)
            loaded = load_artifact(path)
        np.testing.assert_allclose(loaded.predict_p_local(probe), expected)
        self.assertEqual(loaded.threshold, artifact.threshold)
        self.assertEqual(loaded.version, artifact.version)

    def test_missing_artifact(self):
        with self.assertRaises(ArtifactError) as ctx:
            load_artifact("/nonexistent/router.joblib")
        self.assertIn("not found", str(ctx.exception))

    def test_corrupt_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corrupt.joblib")
            with open(path, "wb") as fh:
                fh.write(b"this is not a joblib file at all")
            with self.assertRaises(ArtifactError):
                load_artifact(path)

    def test_wrong_bundle_shape(self):
        import joblib

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "notabundle.joblib")
            joblib.dump({"something": "else"}, path)
            with self.assertRaises(ArtifactError):
                load_artifact(path)

    def test_incompatible_format_version(self):
        import joblib

        _, (artifact, _, _, _) = trained_once()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.joblib")
            save_artifact(artifact, path)
            payload = joblib.load(path)
            payload["format_version"] = "0.0"
            joblib.dump(payload, path)
            with self.assertRaises(ArtifactError) as ctx:
                load_artifact(path)
        self.assertIn("incompatible", str(ctx.exception))
        self.assertEqual(artifact.format_version, ARTIFACT_FORMAT_VERSION)

    def test_incompatible_feature_version(self):
        import joblib

        _, (artifact, _, _, _) = trained_once()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "feat.joblib")
            save_artifact(artifact, path)
            payload = joblib.load(path)
            payload["feature_version"] = "99"
            joblib.dump(payload, path)
            with self.assertRaises(ArtifactError) as ctx:
                load_artifact(path)
        self.assertIn("feature_version", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
