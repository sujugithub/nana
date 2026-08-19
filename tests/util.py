"""Shared test fixtures: a network guard and a fast synthetic dataset.

The synthetic dataset here is intentionally SMALLER and simpler than
scripts/make_toy_dataset.py — tests need speed and a clean learnable signal,
not realism. Prompts carry an explicit lexical difficulty cue so a TF-IDF
model converges reliably on ~200 rows.
"""
from __future__ import annotations

import atexit
import os
import random
import socket
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.dataset import OutcomeDataset, OutcomeRecord  # noqa: E402

_REAL_CONNECT = socket.socket.connect


def _blocked_connect(self, *args, **kwargs):
    raise RuntimeError(
        f"network access attempted during tests (connect{args!r}) — the "
        f"test suite must be fully offline"
    )


def install_network_guard() -> None:
    socket.socket.connect = _blocked_connect
    atexit.register(lambda: setattr(socket.socket, "connect", _REAL_CONNECT))


install_network_guard()

# The last body in each list is an EXCEPTION the keyword heuristics misroute
# (easy prompt with a math keyword; hard prompt with no trigger words), so a
# learned model has measurable headroom over the heuristic on this data.
_EASY_BODIES = [
    "What is the capital of {x}?",
    "Is this review positive or negative: the {x} was lovely and worked well.",
    "Name the largest city in {x}.",
    "Calculate the sum of two and three for the {x} report.",
]
_HARD_BODIES = [
    "Prove that the sum of two odd {x} numbers is even, step by step.",
    "Write a python function to balance a {x} binary search tree.",
    "Solve the equation 3x + {x}7 = 5x - 9 and verify the solution.",
    "In what ways would the {x} strategy break under load, and what would "
    "you change about its retry behaviour in each situation?",
]
_FILLERS = [
    "france", "japan", "peru", "kenya", "norway", "brazil", "canada", "vietnam",
    "spain", "egypt", "chile", "india", "italy", "ghana", "poland", "cuba",
]
_CONTEXT_WORDS = [
    "harbor", "violet", "ledger", "quartz", "meadow", "copper", "sonnet",
    "glacier", "lantern", "prairie", "cobalt", "thicket", "ember", "willow",
    "granite", "orchard", "raven", "saffron", "tundra", "velvet", "zephyr",
    "bramble", "cinder", "dapple", "fable", "gossamer", "hollow", "isle",
]


def make_test_dataset(
    n: int = 200, seed: int = 3, quality_threshold: float = 0.6
) -> OutcomeDataset:
    rng = random.Random(seed)
    records = []
    i = 0
    while len(records) < n:
        hard = i % 2 == 1
        body = (_HARD_BODIES if hard else _EASY_BODIES)[i % 4]
        filler = _FILLERS[(i * 7) % len(_FILLERS)]
        # Distinct context words per group keep instantiations of the same
        # template below the near-duplicate threshold — otherwise every
        # template collapses into one giant group and the splits degenerate.
        context = " ".join(rng.sample(_CONTEXT_WORDS, 3))
        group = f"grp-{i}"
        # two paraphrase variants per group
        for variant in (body, "Please answer this: " + body):
            prompt = variant.format(x=filler) + f" Context tag: {context} {i}."
            local_q = round(
                max(0.0, min(1.0, (0.25 if hard else 0.85) + rng.gauss(0, 0.1))), 3
            )
            conf = max(0.0, min(1.0, 0.4 + 0.5 * local_q + rng.gauss(0, 0.05)))
            records.append(
                OutcomeRecord(
                    task_id=f"tt-{len(records):04d}",
                    prompt=prompt,
                    category="hard" if hard else "easy",
                    source="unit-test",
                    group_id=group,
                    local_quality=local_q,
                    remote_quality=round(min(1.0, 0.9 + rng.gauss(0, 0.03)), 3),
                    local_ok=local_q >= quality_threshold,
                    local_prompt_tokens=20,
                    local_completion_tokens=30,
                    remote_prompt_tokens=20,
                    remote_completion_tokens=40,
                    local_cost=0.00001,
                    remote_cost=0.001,
                    local_latency_s=1.0,
                    remote_latency_s=2.0,
                    local_confidence=round(conf, 3),
                    local_min_token_prob=round(conf * 0.5, 3),
                    local_low_token_frac=round(1 - conf, 3),
                )
            )
            if len(records) >= n:
                break
        i += 1
    return OutcomeDataset(
        meta={
            "local_model": "test-local",
            "remote_model": "test-remote",
            "quality_threshold": quality_threshold,
        },
        records=records,
    )


@lru_cache(maxsize=1)
def trained_once():
    """One shared (artifact, report, splits, candidates) per test process —
    training is the slow part, so every module reuses this."""
    from routing.train import train

    dataset = make_test_dataset()
    return dataset, train(dataset, seed=3, candidates=["logreg_tfidf"])
