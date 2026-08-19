"""Feature extraction: determinism, and the pre-route/post-gen boundary."""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from tests.util import make_test_dataset  # noqa: F401

import routing.features as features_module
from routing.dataset import POST_GEN_FIELDS
from routing.features import (
    ENGINEERED_FEATURE_NAMES,
    engineered_matrix,
    engineered_vector,
)


class TestEngineeredFeatures(unittest.TestCase):
    def test_vector_matches_declared_names(self):
        vec = engineered_vector("What is 2 + 2?")
        self.assertEqual(len(vec), len(ENGINEERED_FEATURE_NAMES))

    def test_deterministic(self):
        prompt = "Prove that the sum of two odd numbers is even."
        self.assertEqual(engineered_vector(prompt), engineered_vector(prompt))

    def test_matrix_shape_and_dtype(self):
        m = engineered_matrix(["a?", "compute 3 * 4 please"])
        self.assertEqual(m.shape, (2, len(ENGINEERED_FEATURE_NAMES)))
        self.assertEqual(m.dtype, np.float64)

    def test_signals_fire(self):
        names = ENGINEERED_FEATURE_NAMES
        vec = dict(zip(names, engineered_vector("Calculate 12 * 9 step by step")))
        self.assertEqual(vec["signal_math"], 1.0)
        self.assertEqual(vec["signal_sentiment"], 0.0)

    def test_empty_prompt_is_safe(self):
        vec = engineered_vector("")
        self.assertTrue(all(np.isfinite(v) for v in vec))


class TestPreRouteBoundary(unittest.TestCase):
    """The pre-router must not be able to see post-generation information."""

    def test_extractors_take_only_prompt_strings(self):
        sig = inspect.signature(engineered_matrix)
        self.assertEqual(list(sig.parameters), ["prompts"])
        sig = inspect.signature(engineered_vector)
        self.assertEqual(list(sig.parameters), ["prompt"])

    def test_feature_source_never_references_post_gen_fields(self):
        source = inspect.getsource(features_module)
        for field in POST_GEN_FIELDS:
            self.assertNotIn(
                field, source,
                f"features.py references post-generation field {field!r}",
            )

    def test_features_ignore_outcome_changes(self):
        # Same prompt, wildly different outcomes → identical features.
        dataset = make_test_dataset(n=4)
        rec = dataset.records[0]
        before = engineered_vector(rec.prompt)
        rec.local_quality = 0.0
        rec.local_confidence = 0.01
        after = engineered_vector(rec.prompt)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
