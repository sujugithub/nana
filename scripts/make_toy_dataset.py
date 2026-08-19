"""Generate a deterministic TOY outcome dataset for end-to-end pipeline runs.

    python3 scripts/make_toy_dataset.py --out data/toy_dataset.json --n 900

Every number in the output is SYNTHETIC. The generator exists so the full
train -> evaluate -> deploy loop can be exercised offline, with zero paid
calls and zero downloads; results measured on it say the pipeline works, not
that the router works on real workloads. Collect real outcomes with
scripts/collect_outcomes.py before believing any number.

Shape of the synthesis (designed to resemble the real problem, not to
flatter the router):
- 8 task categories matching the project's evaluation plan, each with
  template banks; every base template yields several PARAPHRASES that share
  a group_id — exercising the group-aware split machinery.
- A latent difficulty drives local quality: category base rate + per-template
  modifier + prompt-length effect + noise. Crucially, ~15% of templates are
  EXCEPTIONS (trivial math like "what is 4 + 5", obscure-entity factual
  lookups) so category keywords alone cannot route optimally — the headroom
  a learned router is supposed to exploit.
- The remote model is strong everywhere but not perfect on the hardest items.
- Post-generation signals (mean/min/low-frac token probability) are noisy
  transforms of one latent correctness signal, including a slice of
  CONFIDENT-WRONG answers (high stated confidence, wrong) — mirroring the
  real failure mode documented in ARCHITECTURE.md.
- Costs use plausible-but-invented rates, stated in meta.notes.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.dataset import (  # noqa: E402
    OutcomeDataset,
    OutcomeRecord,
    normalize_prompt,
    save_dataset,
)

QUALITY_THRESHOLD = 0.6

# Invented, plausible cost rates (documented in meta.notes):
LOCAL_COST_PER_TOKEN = 0.02 / 1e6  # amortised local compute
REMOTE_IN_PER_TOKEN = 0.5 / 1e6
REMOTE_OUT_PER_TOKEN = 1.5 / 1e6

_TOPICS = [
    "the Amazon river", "photosynthesis", "the French Revolution", "black holes",
    "the stock market", "renewable energy", "ancient Rome", "machine learning",
    "the human heart", "plate tectonics", "the printing press", "coral reefs",
]
_NAMES = ["Maria", "Chen", "Amara", "Lukas", "Priya", "Diego", "Yuki", "Fatima"]
_CITIES = [
    "Paris", "Nairobi", "Osaka", "Lima", "Oslo", "Hanoi", "Porto", "Quebec",
]
_COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella Labs", "Stark Industries"]
_PRODUCTS = ["laptop", "blender", "headset", "coffee maker", "monitor", "keyboard"]
_OBSCURE = [
    "the Treaty of Kuchuk-Kainarji", "the Diprotodon", "the Antikythera mechanism",
    "the Carrington Event", "the Voynich manuscript", "Göbekli Tepe",
]

# (category, base_difficulty, [(template_variants, difficulty_modifier)])
# Variants within one tuple share a group (paraphrases of one base prompt).
_TEMPLATES: List[Tuple[str, float, List[Tuple[List[str], float]]]] = [
    (
        "factual",
        0.18,
        [
            (
                [
                    "What is the capital of {city_country}?",
                    "Name the capital city of {city_country}.",
                    "Which city is the capital of {city_country}?",
                ],
                0.0,
            ),
            (
                [
                    "In what year did {topic} first become widely known?",
                    "Roughly when did {topic} first become widely known?",
                ],
                0.1,
            ),
            (
                [  # EXCEPTION: obscure factual — looks easy, is hard
                    "What is {obscure} and why is it significant?",
                    "Explain what {obscure} is and why it matters.",
                    "Give a short account of {obscure} and its significance.",
                ],
                0.55,
            ),
        ],
    ),
    (
        "sentiment",
        0.10,
        [
            (
                [
                    "Classify the sentiment of this review as positive or negative: "
                    "The {product} exceeded my expectations, {name} loved it too.",
                    "Is this review positive or negative? The {product} exceeded "
                    "my expectations, {name} loved it too.",
                ],
                0.0,
            ),
            (
                [
                    "Classify the sentiment of the following review as positive, "
                    "negative, or neutral: I wanted to like the {product} but it "
                    "stopped working after a week and support never replied.",
                    "Decide whether this review is positive, negative or neutral: "
                    "I wanted to like the {product} but it stopped working after "
                    "a week and support never replied.",
                ],
                0.05,
            ),
            (
                [  # sarcasm — harder than the category suggests
                    "Classify the sentiment: Oh great, another {product} that "
                    "dies the day the warranty ends. Fantastic engineering.",
                    "Positive or negative? Oh great, another {product} that dies "
                    "the day the warranty ends. Fantastic engineering.",
                ],
                0.35,
            ),
        ],
    ),
    (
        "ner",
        0.15,
        [
            (
                [
                    "Extract all named entities from this text: {name} from "
                    "{company} met investors in {city} on Tuesday.",
                    "List the people, organizations and locations mentioned: "
                    "{name} from {company} met investors in {city} on Tuesday.",
                ],
                0.0,
            ),
            (
                [
                    "Identify every person, organization and location: After the "
                    "{city} summit, {name} of {company} flew to {city2} with "
                    "{name2} to brief the {company2} board.",
                    "Extract the named entities (people, orgs, places): After the "
                    "{city} summit, {name} of {company} flew to {city2} with "
                    "{name2} to brief the {company2} board.",
                ],
                0.15,
            ),
        ],
    ),
    (
        "summarize",
        0.22,
        [
            (
                [
                    "Summarize in one sentence: The council approved a plan for "
                    "{topic} after months of debate; supporters cite long-term "
                    "benefits while opponents question the budget.",
                    "Give a one-sentence summary: The council approved a plan "
                    "for {topic} after months of debate; supporters cite "
                    "long-term benefits while opponents question the budget.",
                ],
                0.0,
            ),
            (
                [
                    "Summarize the key points: Researchers studying {topic} "
                    "reported three findings. First, effects appear earlier than "
                    "assumed. Second, the impact varies by region. Third, "
                    "mitigation is cheaper than repair. Critics note the sample "
                    "was small and the funding source is contested; the authors "
                    "acknowledge both limits but defend the conclusions.",
                    "Condense this into two sentences: Researchers studying "
                    "{topic} reported three findings: effects appear earlier "
                    "than assumed, impact varies by region, and mitigation is "
                    "cheaper than repair. Critics note the small sample and "
                    "contested funding; the authors defend the conclusions.",
                ],
                0.2,
            ),
        ],
    ),
    (
        "math",
        0.72,
        [
            (
                [  # EXCEPTION: trivial arithmetic — keyword rules over-escalate
                    "What is {small_a} + {small_b}?",
                    "Calculate {small_a} + {small_b}.",
                    "Compute the sum of {small_a} and {small_b}.",
                ],
                -0.55,
            ),
            (
                [
                    "A shop sells pens for {a} dollars and pads for {b} dollars. "
                    "{name} bought {c} pens and some pads for {total} dollars. "
                    "How many pads did {name} buy?",
                    "Pens cost {a} dollars and pads cost {b} dollars. {name} "
                    "spent {total} dollars on {c} pens and some pads. How many "
                    "pads is that?",
                ],
                0.1,
            ),
            (
                [
                    "Solve for x: {a}x + {b} = {c}x - {total}, then verify the "
                    "solution and explain each algebraic step.",
                    "Find x if {a}x + {b} = {c}x - {total}; verify the answer "
                    "and justify every step.",
                ],
                0.2,
            ),
        ],
    ),
    (
        "code_gen",
        0.68,
        [
            (
                [  # EXCEPTION: one-liner — small models handle it
                    "Write a Python one-liner that reverses the string s.",
                    "Give a single line of Python that reverses a string s.",
                ],
                -0.45,
            ),
            (
                [
                    "Write a Python function that merges two sorted lists into "
                    "one sorted list without using the built-in sort.",
                    "Implement a Python function merging two pre-sorted lists "
                    "into a single sorted list; do not call sort().",
                ],
                0.05,
            ),
            (
                [
                    "Implement an LRU cache class in Python with O(1) get and "
                    "put, and explain the data-structure choice.",
                    "Write a Python LRU cache with constant-time get and put "
                    "operations, explaining the underlying data structures.",
                ],
                0.25,
            ),
        ],
    ),
    (
        "code_debug",
        0.70,
        [
            (
                [
                    "Debug this Python: def mean(xs): return sum(xs)/len(xs) — "
                    "it crashes on empty input. Fix it.",
                    "This Python crashes on empty lists: def mean(xs): return "
                    "sum(xs)/len(xs). Provide the fix.",
                ],
                -0.15,
            ),
            (
                [
                    "Fix the bug: def second_largest(xs): return sorted(xs)[-2] "
                    "— wrong on duplicates and crashes on short lists. Explain "
                    "the failure modes and correct them.",
                    "Debug def second_largest(xs): return sorted(xs)[-2]; it "
                    "mishandles duplicates and short lists. Explain and fix.",
                ],
                0.15,
            ),
        ],
    ),
    (
        "logic",
        0.75,
        [
            (
                [
                    "If all bloops are razzies and all razzies are lazzies, are "
                    "all bloops definitely lazzies? Answer yes or no.",
                    "All bloops are razzies; all razzies are lazzies. Must all "
                    "bloops be lazzies? Yes or no.",
                ],
                -0.25,
            ),
            (
                [
                    "{name} says {name2} is lying; {name2} says {name} and "
                    "{name3} are both lying. Exactly one person tells the "
                    "truth. Who is it? Deduce step by step.",
                    "Exactly one of {name}, {name2}, {name3} is truthful. "
                    "{name} claims {name2} lies; {name2} claims both {name} and "
                    "{name3} lie. Determine the truth-teller, reasoning it out.",
                ],
                0.15,
            ),
        ],
    ),
]

_COUNTRY_OF = {
    "Paris": "France", "Nairobi": "Kenya", "Osaka": "Japan", "Lima": "Peru",
    "Oslo": "Norway", "Hanoi": "Vietnam", "Porto": "Portugal", "Quebec": "Canada",
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _fill(template: str, rng: random.Random) -> str:
    city = rng.choice(_CITIES)
    city2 = rng.choice([c for c in _CITIES if c != city])
    name = rng.choice(_NAMES)
    name2 = rng.choice([n for n in _NAMES if n != name])
    name3 = rng.choice([n for n in _NAMES if n not in (name, name2)])
    company = rng.choice(_COMPANIES)
    a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
    return template.format(
        topic=rng.choice(_TOPICS),
        name=name, name2=name2, name3=name3,
        city=city, city2=city2,
        city_country=_COUNTRY_OF[city],
        company=company,
        company2=rng.choice([x for x in _COMPANIES if x != company]),
        product=rng.choice(_PRODUCTS),
        obscure=rng.choice(_OBSCURE),
        small_a=rng.randint(2, 12), small_b=rng.randint(2, 12),
        a=a, b=b, c=c, total=a * c + b * rng.randint(1, 6),
    )


def generate(n: int, seed: int) -> OutcomeDataset:
    rng = random.Random(seed)
    flat = []  # (category, base_diff, variants, modifier)
    for category, base, groups in _TEMPLATES:
        for variants, modifier in groups:
            flat.append((category, base, variants, modifier))

    records: List[OutcomeRecord] = []
    group_counter: Dict[str, int] = {}
    seen_prompts: set = set()
    while len(records) < n:
        category, base, variants, modifier = rng.choice(flat)
        group_counter[variants[0]] = group_counter.get(variants[0], 0) + 1
        # hashlib, not hash(): str hash is randomized per process and would
        # make group ids (and therefore splits) non-reproducible.
        template_key = hashlib.sha1(variants[0].encode()).hexdigest()[:8]
        group_id = f"{category}-{template_key}-{group_counter[variants[0]]}"
        # One slot-filled instantiation per group; each paraphrase variant of
        # it becomes its own record in the SAME group.
        seed_rng = random.Random(rng.random())
        n_variants = rng.randint(1, len(variants))
        chosen = rng.sample(variants, n_variants)
        for variant in chosen:
            prompt = _fill(variant, random.Random(seed_rng.random() * 1e9 // 1))
            # The slot pools are small, so the same filled prompt CAN recur
            # in a different group — which the split machinery rightly
            # rejects as a leakage bug. Skip repeats instead of emitting them.
            norm = normalize_prompt(prompt)
            if norm in seen_prompts:
                continue
            seen_prompts.add(norm)
            words = len(prompt.split())
            difficulty = _clip01(
                base
                + modifier
                + 0.002 * max(0, words - 30)  # long prompts are a bit harder
                + rng.gauss(0, 0.06)
            )
            local_quality = round(_clip01(1.0 - difficulty + rng.gauss(0, 0.12)), 3)
            remote_quality = round(
                _clip01(0.97 - 0.18 * difficulty + rng.gauss(0, 0.04)), 3
            )
            local_ok = local_quality >= QUALITY_THRESHOLD

            prompt_tokens = int(words * 1.35) + 4
            local_out = {
                "factual": 25, "sentiment": 8, "ner": 30, "summarize": 45,
                "math": 90, "code_gen": 160, "code_debug": 140, "logic": 110,
            }[category] + rng.randint(-5, 25)
            remote_out = int(local_out * rng.uniform(0.8, 1.4)) + 10

            # Post-gen signals: one latent correctness signal + noise, with a
            # confident-wrong slice (the documented real-world failure mode).
            latent = 0.45 + 0.5 * local_quality + rng.gauss(0, 0.07)
            if not local_ok and rng.random() < 0.10:
                latent = rng.uniform(0.85, 0.97)  # confident and wrong
            mean_conf = round(_clip01(latent), 3)
            min_conf = round(_clip01(latent * rng.uniform(0.3, 0.6)), 3)
            low_frac = round(_clip01(0.85 - 0.8 * latent + rng.gauss(0, 0.05)), 3)

            problems: List[str] = []
            if local_quality < 0.2 and rng.random() < 0.35:
                problems.append(
                    rng.choice(["hedging_or_refusal", "degenerate_repetition"])
                )

            records.append(
                OutcomeRecord(
                    task_id=f"toy-{len(records):04d}",
                    prompt=prompt,
                    category=category,
                    source="toy-generator",
                    group_id=group_id,
                    metadata={},
                    local_quality=local_quality,
                    remote_quality=remote_quality,
                    local_ok=local_ok,
                    local_prompt_tokens=prompt_tokens,
                    local_completion_tokens=max(3, local_out),
                    remote_prompt_tokens=prompt_tokens,
                    remote_completion_tokens=max(8, remote_out),
                    local_cost=round(
                        (prompt_tokens + max(3, local_out)) * LOCAL_COST_PER_TOKEN, 9
                    ),
                    remote_cost=round(
                        prompt_tokens * REMOTE_IN_PER_TOKEN
                        + max(8, remote_out) * REMOTE_OUT_PER_TOKEN,
                        9,
                    ),
                    local_latency_s=round(0.2 + max(3, local_out) / 20.0, 3),
                    remote_latency_s=round(1.2 + max(8, remote_out) / 80.0, 3),
                    local_confidence=mean_conf,
                    local_min_token_prob=min_conf,
                    local_low_token_frac=low_frac,
                    post_check_problems=problems,
                )
            )
            if len(records) >= n:
                break

    meta = {
        "local_model": "toy-local-1.5b (synthetic)",
        "remote_model": "toy-remote-frontier (synthetic)",
        "quality_threshold": QUALITY_THRESHOLD,
        "cost_unit": "usd",
        "seed": seed,
        "notes": (
            "SYNTHETIC toy data from scripts/make_toy_dataset.py. Costs are "
            f"invented rates: local {LOCAL_COST_PER_TOKEN * 1e6:.2f}/M tok "
            f"(amortised compute), remote {REMOTE_IN_PER_TOKEN * 1e6:.2f}/M in "
            f"+ {REMOTE_OUT_PER_TOKEN * 1e6:.2f}/M out. Results on this data "
            "validate the PIPELINE, not real routing performance."
        ),
    }
    return OutcomeDataset(meta=meta, records=records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/toy_dataset.json")
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dataset = generate(args.n, args.seed)
    save_dataset(dataset, args.out)
    ok = sum(r.local_ok for r in dataset.records)
    groups = len({r.group_id for r in dataset.records})
    print(
        f"wrote {len(dataset.records)} records ({groups} groups, "
        f"{ok / len(dataset.records):.0%} local_ok) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
