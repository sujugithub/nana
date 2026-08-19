# Learned routing

How the learned pre-router works, how to build a dataset for it, train it,
evaluate it, and turn it on. Written so a teammate can reproduce every step
without reading the code first.

## What it is

The heuristic router (`confidence.py`) guesses difficulty from keywords. The
learned router replaces that guess with a trained estimate of

    P(the local model produces an acceptable answer | prompt)

trained on *observed outcomes* of this project's actual local and remote
models — not on anyone's idea of what "looks hard". Route local when that
probability clears an operating threshold chosen from data; otherwise remote.

Everything downstream is unchanged: a local answer still has to survive
`router.post_check` and the token-probability escalation gate, and every
failure path (escalation failure, remote failure) behaves exactly as before.

```
prompt ──▶ pre-router (heuristic OR learned) ──▶ local or remote
             local ──▶ post_check + confidence gate ──▶ ok, or escalate remote
```

Pre-routing and post-generation confidence stay strictly separate: the
pre-router's features are computable from the prompt alone, before any model
runs. `routing/dataset.py` marks the post-generation fields and
`tests/test_features.py` fails if `routing/features.py` ever references one.

## Architecture and why

Two candidates are trained and compared on every run:

| Candidate | What it is |
| --- | --- |
| `logreg_tfidf` | TF-IDF (word 1-2grams + char 3-5grams) + 23 engineered features → logistic regression. The simple baseline that must always exist. |
| `gbdt_svd` | Same features → TruncatedSVD(64) → histogram gradient boosting. The "stronger" candidate; it wins only if the validation operating point is actually better. |

The engineered features include the heuristic router's own keyword signals —
the hand-tuned patterns encode real domain knowledge, and letting the model
weigh them beats trusting the hand-set 0.75 penalties.

Why not embeddings or a fine-tuned transformer? Three reasons:
1. **Dataset size.** Real collection for this project yields hundreds to a
   few thousand graded rows. TF-IDF + linear/boosted models are the right
   capacity for that; a transformer is not, and embedding models add a
   download + inference dependency to every routed request.
2. **Router cost discipline.** The router must stay much cheaper and faster
   than the models it routes between. This one is microseconds on CPU.
3. **Evidence.** LLMRouterBench (ACL Findings 2026) reports that under
   standardized evaluation many sophisticated routers — commercial ones
   included — fail to reliably beat simple baselines, and that embedding
   backbones have limited utility. Start simple, measure, escalate only on
   measured wins. The `gbdt_svd` candidate is exactly that escalation path,
   and the training report shows whether it earned its keep.

Probabilities are **calibrated** (sigmoid or isotonic, chosen by validation
Brier score, fitted with group-aware CV on training data), because the
operating threshold is chosen on the probability scale and an uncalibrated
score would make the chosen policy meaningless.

### Research ideas used, with attribution

- **RouteLLM** (Ong et al., 2024, arXiv:2406.18665; Apache-2.0,
  github.com/lm-sys/routellm): routing as a learned binary preference between
  a weak and strong model, with the threshold *calibrated* to hit a target
  cost/quality point rather than fixed at 0.5. Our `remote_rate` policy is
  their calibration idea; the win-rate framing became our `local_ok` label.
  No code was copied.
- **HybridLLM** (Ding et al., 2024, arXiv:2404.14618): train the router on
  the *quality gap* between models rather than intrinsic "difficulty" labels.
  Our labels are derived from graded local vs remote quality with an explicit
  threshold.
- **FrugalGPT** (Chen, Zaharia & Zou, 2023, arXiv:2305.05176): the cascade —
  a cheap model plus a learned scorer that decides accept-or-escalate. Our
  post-generation gate is that scorer; the honest-accounting rule (a
  discarded local generation still costs compute and latency) comes from
  their cost model.
- **RouterBench** (Hu et al., 2024, github.com/withmartian/routerbench):
  evaluate routers as cost-quality *curves* against non-learned baselines,
  never as single accuracy numbers. Our evaluator's sweeps, random-routing
  line, and oracle ceiling follow this design.
- **LLMRouterBench** (ACL Findings 2026): standardized comparison shows
  simple baselines are hard to beat and oracle gaps are dominated by recall
  failures — the reason this project *requires* the random-line and
  heuristic comparisons before believing the learned router.

## The outcome dataset

One JSON file, schema-versioned (`schema_version: "1.0"`), validated on
every load. Full field list in `routing/dataset.py`. The essentials per
record: the prompt, safe pre-route metadata, a `group_id` tying related or
paraphrased prompts together, graded `local_quality` and `remote_quality`
(0..1), the label `local_ok` (which MUST equal
`local_quality >= meta.quality_threshold` — the validator rejects
contradictions), token counts, measured costs and latencies for both models,
and the local model's post-generation confidence statistics.

### Leakage rules (enforced, not advisory)

- Splits are **group-aware**: a `group_id` lands in exactly one of
  train/val/test. Paraphrases of a training prompt can never inflate test.
- Identical normalized prompts in different groups are a **hard error**.
- Near-duplicates (char-shingle Jaccard ≥ 0.85) in different groups get
  their groups merged, loudly.
- TF-IDF and scalers are fitted inside the sklearn pipeline on training
  data only; the test split is written to `splits.json` at training time
  and spent once, by the evaluator.
- Seed, ratios, and the full split assignment are recorded in the training
  report.

## Commands

### 0) Toy end-to-end loop (offline, free, ~1 minute)

```bash
make toy-data     # data/toy_dataset.json  (synthetic — see caveat below)
make train        # artifacts/router.joblib + reports/train/
make evaluate     # reports/eval/results.json + pareto.html
make test         # wiring harness + 103-test learned-routing suite
```

**Caveat:** toy numbers validate the *pipeline*, not the router. Every
quality/cost value in the toy dataset is synthesized
(`scripts/make_toy_dataset.py` documents how).

### 1) Collect a real dataset

Write a task file with references and graders (format in
`scripts/collect_outcomes.py`), then:

```bash
# free pipeline test first:
python3 scripts/collect_outcomes.py --tasks tasks/graded_tasks.json \
    --out data/collected.json --mock

# the real thing — one PAID remote call per task, so consent is explicit:
python3 scripts/collect_outcomes.py --tasks tasks/graded_tasks.json \
    --out data/collected.json --run-paid-calls
```

It runs every task through BOTH models, grades programmatically (exact /
numeric / contains / choice), records tokens, latency, and the three
post-generation confidence statistics, appends after every task (interrupted
runs resume without re-paying), and leaves ungraded tasks in a work file for
external grading (`--grades verdicts.json` merges them back).

### 2) Train

```bash
python3 -m routing.train --dataset data/collected.json \
    --out artifacts/router.joblib --report reports/train --seed 7
```

Produces:
- `artifacts/router.joblib` — the full pipeline + calibrator + threshold +
  every version string, loaded once at runtime.
- `reports/train/report.json` — machine-readable: per-candidate validation
  metrics (ROC-AUC, PR-AUC both classes, Brier, ECE, confusion), threshold
  sweeps, chosen calibration method, split assignment, dataset hash, seed.
- `reports/train/summary.txt` — the human version.
- `reports/train/candidate_*.joblib` — every candidate, for the evaluator.

### 3) Choose the threshold policy

0.5 is never assumed. The default policy is `quality_floor` (cheapest
threshold retaining ≥97% of all-remote quality on validation). Others:

```bash
# highest local share with unsafe-local rate ≤ 2%:
python3 -m routing.train --dataset data/collected.json \
    --policy max_unsafe --max-unsafe-rate 0.02 ...

# maximize quality − cost_weight·mean_cost − latency_weight·mean_latency:
python3 -m routing.train --dataset data/collected.json \
    --policy utility --cost-weight 5.0 --latency-weight 0.01 ...

# hit a spend budget: ~30% of tasks remote (RouteLLM-style):
python3 -m routing.train --dataset data/collected.json \
    --policy remote_rate --target-remote-rate 0.3 ...
```

An **unsafe-local** decision (router chose local; the local answer was
graded a failure) is the error that matters most; every policy reports it
and `max_unsafe` optimizes against it directly.

### 4) Evaluate — the one-shot test number

```bash
python3 -m evaluation.run --dataset data/collected.json \
    --artifact artifacts/router.joblib --train-report reports/train \
    --report reports/eval
```

Compares all-local, all-remote, seeded random routing (50 trials × 21 remote
rates), the heuristic rules at their deployed threshold, every trained
candidate, the selected router, and the oracle — with bootstrap CIs,
per-category breakdowns, easy-heavy/hard-heavy workload mixtures, and a
self-contained `pareto.html`. The verdict block answers the two questions
that decide whether the router earned its complexity: does it clear the
random line at matched spend, and does it beat the heuristic paired on the
same tasks? **If it does not, that is the result** — report it and diagnose;
do not tune on the test split (it has been spent).

Note on the oracle: it is the *cost-optimal never-unsafe* policy (remote
exactly when local would fail). Its mean quality can sit below all-remote,
because it happily keeps acceptable-but-mediocre local answers. Read its
column as the cost ceiling at zero unsafe rate, not a quality ceiling.

### 5) Turn it on

```bash
ROUTER_MODE=learned python3 main.py --tasks tasks/sample_tasks.json
ROUTER_MODE=auto    python3 main.py ...   # fall back to heuristic, loudly
```

- `learned` fails fast and clearly when `ROUTER_ARTIFACT` (default
  `artifacts/router.joblib`) is missing or incompatible.
- `auto` falls back to the heuristic with a stderr warning.
- Every usage-log line records `router`, `artifact_version`, `p_local`, and
  the active threshold, so any routed task is traceable to the exact policy
  that routed it.
- Score direction is unchanged everywhere: **higher always means "local is
  safer"**.

## Post-generation gate

`local_model.py` now computes three statistics from the same logits at zero
extra compute: mean token probability (the old signal), minimum token
probability, and the fraction of low-confidence tokens. All three are
logged. The runtime gate uses `LOCAL_CONF_STAT` (`mean` by default), and the
training report's post-gen AUC comparison is the evidence for changing it —
on real data, check `reports/train/report.json → postgen.auc_by_stat` and
switch only if another statistic clearly separates failures better.

## Reproducibility

Same dataset + same seed ⇒ same splits, same hyperparameter search, same
calibration, same threshold, same predictions (asserted by
`tests/test_train_artifact.py::test_reproducible_given_seed`). The artifact
records dataset hash, seed, schema/feature/artifact-format versions and the
sklearn version; the evaluator warns when the dataset hash does not match
the artifact.

## Security: artifacts are executables

`artifacts/*.joblib` are pickle-based. **Loading one executes arbitrary code
embedded in the file.** Load only artifacts you or your pipeline produced;
never one downloaded from an untrusted source. This project never
auto-downloads artifacts, and the artifact loader refuses files whose format
or feature version does not match the code.

## Limitations

- **Toy results are toy.** The generator invents outcomes; the only claims
  it supports are "the pipeline runs end-to-end" and "the machinery can
  recover a signal that exists".
- The router only knows prompts resembling its training distribution.
  Out-of-distribution prompts get calibrated-but-extrapolated probabilities;
  the post-generation cascade is the safety net there.
- `local_ok` binarizes quality at one threshold; information near the
  boundary is lost. A regression-on-quality-gap variant is a natural
  extension (HybridLLM §4 discusses soft labels).
- Random-line comparison interpolates between simulated remote rates; with
  small test sets the CI bands are wide — collect ≥300 rows per category
  (EVALUATION.md) before drawing conclusions.
- The artifact is tied to the model pair it was trained on. Swap either
  model → retrain; the artifact records both names so the mismatch is
  visible.
