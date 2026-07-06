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


# Complexity signals: each pattern marks a query family. POSITIVE weight =
# penalty (a family 1–3B models are weak at — pushes remote); NEGATIVE
# weight = boost (a family they handle reliably — pushes local, e.g. so the
# length ramp doesn't send a long-but-easy sentiment prompt remote). The
# score is clamped to 0..1 after summing. Tuned for the Track-1 kickoff
# categories (2026-07-07): hard-remote = math, code debug/gen, logic;
# easy-local = sentiment, NER, summarization, short factual.
#
# Why the hard categories weigh 0.75: the kickoff scoring is an accuracy
# GATE (fail → excluded), so hard categories must go remote even on SHORT
# prompts — and with length weighted 0.4, a short prompt (length ≈ 0.96)
# stays local unless its penalties exceed ~0.72. One decisive hard signal
# beats stacking-and-hoping; the cost of a false positive is a few remote
# tokens, the cost of a false negative is the whole submission.
_SIGNAL_PATTERNS = {
    # multi-step / symbolic math is where small models fail hardest;
    # includes GSM8K-style word problems (how many/much + numbers)
    "math": (
        0.75,
        re.compile(
            r"\b(calculate|compute|solve|prove|derivative|integral|equation"
            r"|probability|theorem|remainder|percent(age)?|average of|sum of"
            r"|product of)\b|\d+\s*[-+*/^%]\s*\d+"
            r"|\bhow (many|much)\b.*\d|\d.*\bhow (many|much)\b",
            re.I | re.S,
        ),
    ),
    # code generation/debugging beyond one-liners
    "code": (
        0.75,
        re.compile(
            r"\b(function|implement|algorithm|python|javascript|regex|debug"
            r"|refactor|unit test)\b|```",
            re.I,
        ),
    ),
    # debugging specifically: stacks with "code" so fix-this-bug prompts
    # (kickoff category: code debugging) clear the remote bar even when short
    "code_debug": (
        0.35,
        re.compile(
            r"\b(bug|fix (the|this|my)|error|traceback|exception|crash(es)?"
            r"|stack trace|doesn'?t (work|run)|not working|wrong output)\b",
            re.I,
        ),
    ),
    # logical/deductive reasoning puzzles (kickoff category): constraint
    # satisfaction, syllogisms, truth-tellers — reliably beyond small models
    "logic": (
        0.75,
        re.compile(
            r"\b(deduce|deduction|premises?|syllogism|riddle|puzzle|paradox"
            r"|logic(al|ally)? (puzzle|reasoning)|must be true"
            r"|valid (conclusion|argument)|who is (lying|telling the truth)"
            r"|truth[- ]?tellers?|knights? and knaves)\b|\bif all\b",
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
    # ── Easy-local boosts (negative = ADDS confidence) ──────────────────
    # These key on INSTRUCTION words, which stay reliable even when the
    # attached body text accidentally trips a penalty pattern above.
    # sentiment classification: a category small models nail
    "sentiment": (
        -0.40,
        re.compile(
            r"\bsentiment\b|\bpositive,? negative,? or neutral\b"
            r"|\bpositive or negative\b"
            r"|classify (this|the following) (review|tweet|comment|text)",
            re.I,
        ),
    ),
    # named-entity recognition / extraction
    "ner": (
        -0.40,
        re.compile(
            r"named entit|\bentit(y|ies)\b"
            r"|\b(extract|list|identify|find) (all )?(the )?"
            r"(names|people|persons?|organi[sz]ations?|locations?|dates"
            r"|entities)\b",
            re.I,
        ),
    ),
    # summarization: content length matters less than the length ramp thinks
    "summarize": (
        -0.30,
        re.compile(
            r"\bsummari[sz]e\b|\bsummary\b|\btl;?dr\b"
            r"|in (one|a single|1) sentence",
            re.I,
        ),
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
    # min() clamp: boost weights are negative, and a boosted-only prompt
    # must not push the score past 1.0.
    return max(0.0, min(1.0, 1.0 - penalty))


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
