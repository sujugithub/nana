"""Pre-route feature extraction: prompt text + safe metadata ONLY.

The contract that keeps the learned pre-router honest: every feature here is
computable BEFORE any model generates a token. The public entry points take
prompts (strings), never OutcomeRecords, so post-generation fields cannot leak
in by accident — tests/test_features.py locks this down.

Feature groups:
1. Engineered numerics (ENGINEERED_FEATURE_NAMES): surface statistics plus the
   existing heuristic router's signals re-used as features. The hand-tuned
   patterns in confidence.py encode real domain knowledge about what a small
   model fails at; letting the learned model weigh them (instead of trusting
   the hand-set 0.75s) is the cheapest possible transfer of that knowledge.
2. TF-IDF word 1-2 grams and char_wb 3-5 grams, fit on TRAINING data only
   (they live inside the sklearn Pipeline, so fitting discipline is
   structural, not procedural).

Deliberately NOT a feature: the task category label. It is rarely available
for a live prompt, and on curated datasets it can proxy for the grader —
a router that just looks up "math => remote" has learned the dataset's
folder structure, not difficulty.

FEATURE_VERSION (routing/__init__.py) must be bumped on any change here that
alters the feature space; artifact loading refuses a version mismatch.
"""
from __future__ import annotations

import math
import re
from typing import List, Sequence

import numpy as np

from confidence import _SIGNAL_PATTERNS, complexity_score, length_score

_CODE_FENCE_RE = re.compile(r"```")
_MATH_OP_RE = re.compile(r"\d\s*[-+*/^%=]\s*\d")
_QUESTION_WORD_RE = re.compile(
    r"^\s*(what|who|when|where|which|why|how|is|are|do|does|can|did)\b", re.I
)
_ALLCAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")

# One binary feature per heuristic signal pattern, in a frozen order.
_SIGNAL_NAMES: List[str] = sorted(_SIGNAL_PATTERNS)

ENGINEERED_FEATURE_NAMES: List[str] = [
    "n_chars_log",
    "n_words_log",
    "avg_word_len",
    "n_sentences",
    "n_lines",
    "n_question_marks",
    "digit_ratio",
    "upper_ratio",
    "n_math_ops",
    "has_code_fence",
    "starts_with_question_word",
    "n_allcaps_words",
    "heuristic_length_score",
    "heuristic_complexity_score",
] + [f"signal_{name}" for name in _SIGNAL_NAMES]


def engineered_vector(prompt: str) -> List[float]:
    words = prompt.split()
    n_chars = len(prompt)
    n_words = len(words)
    features = [
        math.log1p(n_chars),
        math.log1p(n_words),
        (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        float(len(re.findall(r"[.!?]+", prompt))),
        float(prompt.count("\n") + 1),
        float(prompt.count("?")),
        (sum(c.isdigit() for c in prompt) / n_chars) if n_chars else 0.0,
        (sum(c.isupper() for c in prompt) / n_chars) if n_chars else 0.0,
        float(len(_MATH_OP_RE.findall(prompt))),
        1.0 if _CODE_FENCE_RE.search(prompt) else 0.0,
        1.0 if _QUESTION_WORD_RE.search(prompt) else 0.0,
        float(len(_ALLCAPS_WORD_RE.findall(prompt))),
        length_score(prompt),
        complexity_score(prompt),
    ]
    for name in _SIGNAL_NAMES:
        _, pattern = _SIGNAL_PATTERNS[name]
        features.append(1.0 if pattern.search(prompt) else 0.0)
    return features


def engineered_matrix(prompts: Sequence[str]) -> np.ndarray:
    """Top-level function (not a lambda/closure) so sklearn Pipelines using it
    via FunctionTransformer stay picklable into the artifact."""
    return np.asarray([engineered_vector(p) for p in prompts], dtype=np.float64)


def build_text_engineered_union():
    """FeatureUnion: TF-IDF (word + char) sparse blocks + scaled engineered
    numerics. Returned unfitted; the training pipeline fits it on train only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    return FeatureUnion(
        [
            (
                "tfidf_word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "tfidf_char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "engineered",
                Pipeline(
                    [
                        ("extract", FunctionTransformer(engineered_matrix)),
                        # with_mean=False keeps the output stackable with the
                        # sparse TF-IDF blocks without densifying them.
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
            ),
        ]
    )
