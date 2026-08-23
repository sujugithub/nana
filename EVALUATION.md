# Evaluation methodology

How we will demonstrate that routing is worth doing. This is the backbone of
the final report: if the numbers here are solid, the project stands up; if they
are hand-waved, nothing else rescues it.

> **Status (2026-08-21).** The evaluator implementing this methodology
> exists — `python3 -m evaluation.run` compares all-cheap, all-strong,
> seeded random routing, the heuristic rules, every learned candidate, the
> selected learned router, and the oracle, with bootstrap CIs, per-category
> results, workload mixtures, and a Pareto chart.
>
> It has been run once on **real** data: a 600-outcome Qwen 2.5 1.5B →
> DeepSeek V4 Pro pilot (`reports/eval_qwen_pro_600/`). The learned router
> leads the heuristic and the random line on point estimates but **no
> quality comparison's confidence interval clears zero at n = 90**, and its
> unsafe-local rate is worse than the heuristic's. Full numbers and caveats:
> `docs/HANDOFF.md` §3 and `docs/LEARNED_ROUTING.md`.
>
> What is still missing against the plan below: **scale** (300–500 items per
> category), **five of the eight categories** (the pilot covers math,
> knowledge and reasoning only), **repeated seeds**, and the LLM judge with
> its human validation and verbosity-bias check.

## What we are claiming

That a confidence-gated router reaches a **better accuracy-versus-cost
trade-off** than using any single model for everything — and, critically, that
it beats spending the same money *randomly*.

There is no single headline number. The result is a **curve**.

## The four baselines

All four are required. The third is the one most often omitted, and it is the
one that decides whether the router has learned anything at all.

| Baseline | What it is | Why it matters |
| --- | --- | --- |
| **All-cheap** | Every task on the configured cheap model | The low-cost floor |
| **All-strong** | Every task on the strong model | The expensive “just use one model” case |
| **Random routing at rate p** | Send a random p% strong, sweeping p from 0 to 1 | Traces a line between the corners. **If our router does not sit clearly above it, the learned signal is worthless.** |
| **Oracle routing** | Route strong *only* when cheap would have failed, using ground truth | Cheating by construction: the theoretical routing ceiling. |

Then sweep our own `CONFIDENCE_THRESHOLD` to trace our curve and plot all five
together: accuracy on one axis, cost on the other. Our line should sit above
random and below oracle.

Report-ready sentences fall straight out of that plot, e.g. *"at 40% of
all-strong cost we retain 97% of its accuracy"*.

## Does the difficulty score actually predict anything?

Separate from system performance, test the signal itself:

1. Run every task through **both** models.
2. Grade both.
3. Label each task `cheap_correct` or `cheap_wrong` — this is the ground truth
   the router is trying to predict.
4. Measure **AUC** of the difficulty score against that label.

AUC 0.5 means the score is worthless (indistinguishable from random ranking);
0.8+ means it genuinely detects difficulty. Do this separately for the
pre-route heuristic score and, only in `local_remote`, `local_confidence`.

This also produces the labelled dataset needed to train a learned router —
which now exists: `routing/train.py` reports exactly these AUCs (heuristic
reference vs learned candidates, plus the three post-generation confidence
statistics) on every training run.

## Grading: use code, not judgment, wherever possible

A judge is a source of noise and bias we are choosing to introduce. Seven of
the eight task categories can be verified programmatically, which makes those
numbers unarguable:

| Category | Dataset | Grading |
| --- | --- | --- |
| Factual knowledge | MMLU / TriviaQA | Multiple-choice match / exact-match + F1 |
| Mathematical reasoning | GSM8K | Extract final number, exact match |
| Sentiment classification | SST-2 | Label match |
| Named entity recognition | CoNLL-2003 | Span-level F1 |
| Logical reasoning | LogiQA / BBH | Multiple-choice match |
| Code generation | HumanEval / MBPP | **Execute the unit tests** |
| Code debugging | Bug-injected HumanEval solutions | Execute the unit tests |
| Summarisation | XSum / CNN-DM | LLM judge (the only one that needs it) |

## Judge validation (for summarisation only)

Three requirements, all cheap and all things a panel will ask about:

- **Judge with a model that is not one of the routing tiers.** Judging with a
  model from the same family as a tier introduces self-preference bias.
- **Validate against humans.** Six people can label a ~200-item sample in an
  afternoon. Report inter-annotator agreement, and report judge-versus-human
  agreement (Cohen's κ). *"Our judge agrees with human labels at κ = 0.7"* is
  what makes an evaluation credible rather than hopeful.
- **Test for verbosity bias explicitly.** LLM judges systematically prefer
  longer answers, and this system deliberately produces **shorter** ones — a
  concise-answer prompt cut remote output by more than half. A length-biased
  judge would punish exactly the thing we are optimising. Correlate answer
  length against judge score and report it; a strong correlation means the
  judge is measuring verbosity, not quality, and everything downstream is
  suspect.

Ask the judge for a binary or ternary verdict against an explicit rubric, not a
1–10 score — numeric scores from LLMs are noisy and poorly anchored. Fix
temperature at 0 for reproducibility.

## Statistics

- **Paired comparisons.** Every system sees the same queries, so use paired
  tests — far more statistical power, free by construction.
- **Bootstrap confidence intervals** over the test set; report them on the
  Pareto curve, not just point estimates.
- **Report n.** Sampling 300–500 items per category is statistically adequate
  and keeps run times sane.

## The ablation that shows we understand our own system

**Routing only pays on mixed workloads.** If every query is hard, everything
escalates and we have added latency for nothing; if every query is easy, the
strong tier is never needed. Run the evaluation across several difficulty
mixtures and show how the advantage grows and shrinks.

Demonstrating *when our own system is useless* is what separates a serious
project from a sales pitch.

## Honest accounting

Count the cascade's overhead. When a gate escalates, we paid for a full cheap
generation and discarded it. In `local_remote` that is local compute; in
`remote_pair` it is a billed Flash call. Include either in the cost model and
never describe the two-remote demo as local or free.

Which is the deeper point: the cost model itself must be **real**. The
earlier competition scored local tokens as zero. Local inference costs
compute time, energy, and hardware. Whatever we optimise — dollars, latency,
or energy — must be something we can actually measure. See `ROADMAP.md`,
workstream 4.
