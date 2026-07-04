"""Confidence estimation: "can the LOCAL model answer this accurately?"

Why this module is the heart of the submission
----------------------------------------------
Scoring = token count + output accuracy, and LOCAL tokens count as ZERO.
So the optimal policy is: answer locally whenever the small model is accurate
enough, and pay remote tokens only when accuracy is genuinely at risk.
Confidence estimation is therefore framed as *risk detection*: cheap signals
that a query exceeds what a 1–3B model handles reliably.

Design choices (deliberate, and worth defending to judges):
- Pure-Python heuristics (regex + arithmetic). They run in microseconds, cost
  zero tokens, and are trivial to debug live: "why did task 7 go remote?" →
  print the per-signal breakdown that every decision carries.
- Every scorer returns 0..1 where 1 = "local can handle it". The estimator
  combines them with weights, so on kickoff day you tune ONE dict.
- Deliberately OPTIMISTIC: borderline queries go local, because
  router.post_check() gives a second line of defense on the local OUTPUT and
  escalates to remote if it looks wrong. Optimism here maximizes free tokens;
  the post-check bounds the accuracy downside.

Kickoff-day extension point: a model-based scorer — e.g. have the local model
draft an answer and use its mean token log-prob as confidence. Local compute
is FREE under the scoring rules, so its only cost is latency. If the
heuristics misroute the real task set, add such a scorer as one more entry in
ConfidenceEstimator.scorers with its own weight; nothing else changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class ConfidenceReport:
    score: float  # 0..1 — higher = more confident LOCAL can handle it
    signals: Dict[str, float]  # per-scorer sub-scores, for live debugging


# Complexity signals: each pattern marks a query family where 1–3B models are
# known to be weak. Matching a pattern SUBTRACTS its penalty from the
# complexity score. Tune penalties / add patterns once real tasks are known.
_SIGNAL_PATTERNS = {
    # multi-step / symbolic math is where small models fail hardest
    "math": (
        0.35,
        re.compile(
            r"\b(calculate|compute|solve|prove|derivative|integral|equation"
            r"|probability|theorem)\b|\d+\s*[-+*/^%]\s*\d+",
            re.I,
        ),
    ),
    # code generation/debugging beyond one-liners
    "code": (
        0.35,
        re.compile(
            r"\b(function|implement|algorithm|python|javascript|regex|debug"
            r"|refactor|unit test)\b|```",
            re.I,
        ),
    ),
    # explicit multi-step reasoning demands
    "reasoning": (
        0.25,
        re.compile(
            r"\b(step[ -]by[ -]step|explain why|analy[sz]e|compare|evaluate"
            r"|trade[ -]?offs?|justify)\b",
            re.I,
        ),
    ),
    # several questions / ordered sub-tasks in one prompt
    "multi_part": (
        0.20,
        re.compile(r"\?.*\?|\b\d\.\s|\bfirst\b.*\bthen\b|\bfinally\b", re.I | re.S),
    ),
}

# Length ramp: very short prompts are usually simple lookups a small model
# nails; very long prompts mean lots of context to track — a weakness of
# small models. Linear in between; crude but effective and free.
_SHORT_WORDS = 15
_LONG_WORDS = 120


def length_score(prompt: str) -> float:
    n_words = len(prompt.split())
    if n_words <= _SHORT_WORDS:
        return 1.0
    if n_words >= _LONG_WORDS:
        return 0.0
    return 1.0 - (n_words - _SHORT_WORDS) / (_LONG_WORDS - _SHORT_WORDS)


def complexity_score(prompt: str) -> float:
    penalty = 0.0
    for weight, pattern in _SIGNAL_PATTERNS.values():
        if pattern.search(prompt):
            penalty += weight
    return max(0.0, 1.0 - penalty)


# Relative importance of each scorer. Complexity dominates because a short
# prompt can still be brutally hard ("prove this theorem"), while a long
# prompt of trivial content is merely tedious.
DEFAULT_WEIGHTS = {"length": 0.4, "complexity": 0.6}


class ConfidenceEstimator:
    """Combines independent scorers into one 0..1 confidence value.

    Pluggable by design: scorers is a plain dict of name → callable, weights
    a dict of name → float. Swap or extend either on kickoff day without
    touching the router.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        scorers: Optional[Dict[str, Callable[[str], float]]] = None,
    ):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.scorers = dict(
            scorers or {"length": length_score, "complexity": complexity_score}
        )

    def estimate(
        self, prompt: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ConfidenceReport:
        signals = {
            name: round(fn(prompt), 3) for name, fn in self.scorers.items()
        }
        total_weight = sum(self.weights.get(name, 0.0) for name in signals) or 1.0
        score = (
            sum(signals[name] * self.weights.get(name, 0.0) for name in signals)
            / total_weight
        )
        return ConfidenceReport(score=round(score, 3), signals=signals)
