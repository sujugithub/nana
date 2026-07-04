"""Decision layer: given a task, choose LOCAL or REMOTE — and double-check
local outputs after the fact.

The two-stage design (this is the part being judged)
----------------------------------------------------
1. PRE-ROUTE (decide): zero-cost heuristics from confidence.py score the
   query. High confidence → local (free tokens). Low confidence → straight to
   remote: a doomed local attempt would only add latency and then get
   escalated anyway, so don't bother trying.

2. POST-CHECK (post_check): when the local model DID run, inspect its output
   for the failure modes small models actually exhibit — empty output,
   repetition loops, prompt echo, hedging/refusals. If it looks bad and
   escalation is enabled, main.py retries the task remotely. Local tokens are
   free, so a failed local attempt costs nothing but latency: the cascade
   converts "risky local win" into "safe remote fallback" instead of a wrong
   answer.

Net effect: settings.confidence_threshold controls how aggressively we chase
free tokens; the post-check bounds the accuracy downside. Tune the threshold
first, the post-check rules second.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import ROUTE_LOCAL, ROUTE_REMOTE, settings
from confidence import ConfidenceEstimator
from schemas import Task

# Kickoff-day shortcut: once task categories are known, hard-route entire
# categories and skip the heuristics for them. e.g.:
#   "long_form_reasoning": ROUTE_REMOTE,
#   "classification": ROUTE_LOCAL,
FORCE_ROUTE_BY_CATEGORY: Dict[str, str] = {}

# Phrases that mean the local model punted. A hedge is not always wrong, but
# with a free retry available, escalating on hedges is the cheap safe play.
# Matched with word boundaries and only near the START of the output: genuine
# punts open with the hedge, while a correct answer may legitimately contain
# one mid-sentence ("AI can't...", a translation of "je ne sais pas", ...).
_HEDGE_PATTERNS = tuple(
    re.compile(r"\b" + re.escape(marker) + r"\b")
    for marker in (
        "i don't know",
        "i do not know",
        "i cannot",
        "i can't",
        "as an ai",
        "i'm not sure",
        "i am not sure",
    )
)
_HEDGE_SCAN_CHARS = 120


@dataclass
class RoutingDecision:
    target: str  # ROUTE_LOCAL or ROUTE_REMOTE
    confidence: float
    signals: Dict[str, float]  # per-scorer breakdown, printed for debugging
    reason: str


class Router:
    def __init__(
        self,
        estimator: Optional[ConfidenceEstimator] = None,
        threshold: Optional[float] = None,
    ):
        self.estimator = estimator or ConfidenceEstimator()
        # Snapshot the threshold at construction time so a single run is
        # internally consistent even if settings gets mutated later.
        self.threshold = (
            settings.confidence_threshold if threshold is None else threshold
        )

    def decide(self, task: Task) -> RoutingDecision:
        metadata = task.metadata or {}  # tolerate Task built with metadata=None
        forced = FORCE_ROUTE_BY_CATEGORY.get(str(metadata.get("category")))
        if forced in (ROUTE_LOCAL, ROUTE_REMOTE):
            return RoutingDecision(
                target=forced,
                confidence=1.0,
                signals={},
                reason=f"forced by category={metadata.get('category')!r}",
            )

        report = self.estimator.estimate(task.prompt, metadata)
        if report.score >= self.threshold:
            target, verdict = ROUTE_LOCAL, ">="
        else:
            target, verdict = ROUTE_REMOTE, "<"
        return RoutingDecision(
            target=target,
            confidence=report.score,
            signals=report.signals,
            reason=f"confidence {report.score} {verdict} threshold {self.threshold}",
        )

    def post_check(self, prompt: str, output: str) -> Tuple[bool, List[str]]:
        """Sanity-check a LOCAL output. Returns (ok, list_of_problems).

        These rules target cheap-to-detect, high-precision failure modes.
        Deliberately NOT a quality judge — a wrong-but-fluent answer passes.
        If the real task set has verifiable outputs (exact match, JSON
        schema, unit tests), add a task-specific validator here: it is the
        single highest-leverage accuracy upgrade available.
        """
        problems: List[str] = []
        stripped = output.strip()
        lowered = stripped.lower()

        if len(stripped) < settings.post_check_min_chars:
            problems.append("empty_or_truncated")

        head = lowered[:_HEDGE_SCAN_CHARS]
        if any(pattern.search(head) for pattern in _HEDGE_PATTERNS):
            problems.append("hedging_or_refusal")

        # Base-model-style echo: output begins by repeating the prompt.
        prompt_prefix = prompt.strip().lower()[:40]
        if len(prompt.strip()) > 20 and lowered.startswith(prompt_prefix):
            problems.append("prompt_echo")

        # Degeneration: one trigram dominating the output = repetition loop.
        words = stripped.split()
        if len(words) >= 12:
            trigrams = list(zip(words, words[1:], words[2:]))
            _, top_count = Counter(trigrams).most_common(1)[0]
            if top_count / len(trigrams) > 0.3:
                problems.append("degenerate_repetition")

        return (not problems, problems)
