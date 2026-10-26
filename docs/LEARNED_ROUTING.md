# Learned routing

How the learned pre-router works, how to build a dataset for it, train it,
evaluate it, and turn it on. Written so a teammate can reproduce every step
without reading the code first.

## What it is

The heuristic router (`confidence.py`) guesses difficulty from keywords. The
learned router replaces that guess with a trained estimate of

    P(the cheap-tier model produces an acceptable answer | prompt)

trained on *observed outcomes* of the exact cheap/strong model pair — not on
anyone's idea of what “looks hard.” Route cheap when that probability clears
an operating threshold chosen from data; otherwise route strong.

Everything downstream is unchanged: a cheap answer survives
`router.post_check`; when it came from the local backend it also passes the
token-probability gate. Failure paths use the strong/cheap fallback cascade.

```
prompt ──▶ pre-router (heuristic OR learned) ──▶ cheap or strong
             cheap ──▶ post_check (+ local confidence gate) ──▶ ok/escalate
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
paraphrased prompts together, graded outcomes, token counts, measured costs,
and latencies for both models. Schema 1.0 retains the names `local_*`,
`remote_*`, and `local_ok` for artifact compatibility; for newly collected
data those mean **cheap**, **strong**, and **cheap_ok**. Metadata records
`tier_mode`, the exact models, and `tier_schema`. Local post-generation
confidence fields are populated only when the cheap provider is local.

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

### Current real Qwen/DeepSeek-Pro pilot

The local demo artifact is trained on 600 automatically graded outcomes for
`Qwen/Qwen2.5-1.5B-Instruct` (local cheap tier) versus
`accounts/fireworks/models/deepseek-v4-pro` (strong tier): 200 GSM8K, 200
MMLU, and 200 BIG-Bench Hard prompts. The group-aware split is 420/90/90 and
the exact combined dataset hash is `30c51583d6c8ab07`.

On the untouched 90-row test split, the selected logistic TF-IDF router sent
60% of tasks to Pro and 40% to local, and reached 0.789 routed quality
against 0.756 for the heuristic, 0.707 for seeded random routing at the same
remote rate, 0.411 for all-local and 0.900 for all-remote (oracle 0.956).

**What that does and does not establish.** Read these caveats before quoting
any number above:

- The +0.082 advantage over random at a matched remote rate is a **point
  estimate whose bootstrap CI does not clear the random line**
  (`CI-clears-random: False`). Encouraging, not demonstrated.
- Against the heuristic the quality difference is **+0.033 with CI
  [-0.056, +0.122]** — it includes zero. The cost reduction is the one
  comparison whose CI excludes zero.
- The learned router's **unsafe-local rate is worse** than the heuristic's
  (18.9% vs 15.6%): it keeps more work on the cheap tier and pays in risk.
  Retrain under the `max_unsafe` policy if safety is the priority.
- Test ROC-AUC is **0.645** against **0.820** on validation — a real
  generalisation drop at n = 90.
- Per category the learned policy is largely "send math to Pro" (math 1.00
  at 96% remote; local math quality is 0.179). Correct here, but shallow —
  and the heuristic reaches the same conclusion on math.

Treat this as **real preliminary pilot evidence**, not a final statistical
claim. The honest one-line summary is: at n = 90 the learned router is not
yet statistically distinguishable from the heuristic or from random at
matched spend. More data and repeated seeds are required — see
[`HANDOFF.md`](HANDOFF.md) §6.

Reproducible local outputs:

```text
data/real_qwen_pro_600.json
artifacts/router_qwen_pro_600.joblib
reports/train_qwen_pro_600/
reports/eval_qwen_pro_600/
```

### 1) Collect a real dataset

Write a task file with references and graders (format in
`scripts/collect_outcomes.py`), then:

```bash
# free pipeline test first:
python3 scripts/collect_outcomes.py --tasks tasks/graded_tasks.json \
    --out data/collected.json --mock

# the real thing — explicit consent; remote_pair makes TWO paid calls/task,
# local_remote makes ONE paid call/task:
python3 scripts/collect_outcomes.py --tasks tasks/graded_tasks.json \
    --out data/collected.json --run-paid-calls --max-api-cost-usd <approved-budget>
```

Before the real command, verify `TIER_MODE`, `CHEAP_MODEL_NAME`, and
`STRONG_MODEL_NAME`. It runs every task through BOTH tiers, grades programmatically (exact /
numeric / contains / choice), records tokens, latency, and the three
post-generation confidence statistics, appends after every task (interrupted
runs resume without re-paying), and leaves ungraded tasks in a work file for
external grading (`--grades verdicts.json` merges them back).

For a local Ollama tier, pass `--local-context-tokens N` (or set
`LOCAL_CONTEXT_TOKENS`): without an explicit context the runtime's own
default applies, is not recorded, and the resulting router is
unverifiable. The collector writes four files next to `--out`:

| File | Contents |
|---|---|
| `<out>.manifest.json` | `compatibility` (pair identity, label identity, canonical task-file hash) — compared on every resume; `provenance` (commit, hardware, Python/dependency/runtime versions, cost rates, budget, timestamps) — informational only; one entry per run |
| `<out>.work.jsonl` | one row per attempt: `run_id`, `attempt_id`, `attempt_number`, `status`, `error`, raw and stripped answers, finish reasons, usage source, backend timings, grades and flags |
| `<out>` | the dataset: completed and graded tasks only, with `pair_identity`, `label_identity` and the accounting in `meta` |
| `<out>.summary.json` | attempted / completed / infrastructure-failed / graded / awaiting-review counts, failed attempts by tier, grade flags, and every rate with its numerator and denominator |

Infrastructure failures (timeouts, HTTP errors, a stopped runtime) are
recorded as failed attempts, never as wrong answers; if the cheap tier fails
the strong tier is not called. A resumed run retries failed tasks with a new
attempt ID and keeps the earlier failure rows. `--max-consecutive-failures`
(default 3) stops a run during an outage. A resume is refused when the
deployment identity, label definition or task file differs from the
manifest, or when a work file has no manifest.

Fingerprints: `file_sha256` is the exact bytes (use it to verify a
transferred file; Git line-ending conversion changes it). The canonical
fingerprint (`routing.identity.canonical_json_sha256`) hashes parsed JSON
with sorted keys, compact separators and ASCII escapes, so file layout and
CRLF/LF between tokens do not matter — but nothing inside strings is
normalised: any change to a prompt or answer, including `\r\n` vs `\n` or
Unicode composition, changes it. Pipeline files are written as UTF-8 with LF
line endings on every OS.

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

An **unsafe-cheap** decision (router chose cheap; its answer was graded a
failure) is the error that matters most; the schema/report currently retains
the historical label `unsafe_local`. Every policy reports it
and `max_unsafe` optimizes against it directly.

### 4) Evaluate — the one-shot test number

```bash
python3 -m evaluation.run --dataset data/collected.json \
    --artifact artifacts/router.joblib --train-report reports/train \
    --report reports/eval
```

Compares all-cheap, all-strong, seeded random routing (50 trials × 21 strong
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
- Every usage-log line records `router`, `artifact_version`, `p_cheap_ok`
  (and the legacy `p_local` alias),
  the active threshold, so any routed task is traceable to the exact policy
  that routed it.
- Score direction is unchanged everywhere: **higher always means “the cheap
  tier is safer.”**

### Compatibility rules: retrain vs revalidate

A router is evidence only for the deployment whose outcomes trained it. The
artifact carries that deployment's `pair_identity` (from the collector
manifest); `routing/identity.py` compares it with the running configuration
when the router loads, and the web UI shows the result.

The generation fields are not assumed: they are read from the request each
client would actually send (`ollama_client.build_chat_payload`,
`remote_client.build_chat_payload` — the same builders generation uses),
rendered for a fixed placeholder prompt. A change to the system prompt,
message construction, temperature, thinking mode, tools or any other request
field changes `request_sha256`; a change to the output cap or context window
changes the revalidate fields.

| Change | Class | Why | Current action |
|---|---|---|---|
| Tier mode; cheap or strong provider/model; cheap backend, model digest, quantisation, chat template or model parameters (`/api/show`); anything in the rendered request (system prompt, message construction, temperature, thinking mode, tools) | **Retrain** | The models' answers, and so the labels, change | Collect new outcomes, retrain, re-select the threshold, evaluate |
| Grader version or quality threshold (label identity) | **Retrain** | The labels change for the same answers | Re-grade (raw answers are kept) and retrain; refuse to merge across label identities |
| Cheap/strong output-token caps; local context window | **Revalidate** | Only long inputs/outputs can change; whether they did must be measured, not assumed | Refused at runtime. See "No revalidation-acceptance workflow" below |
| Local runtime (Ollama) version | Informational, surfaced | Backend upgrades can change templating, kernels or runners | Recorded as provenance; not automatically compared, but changes may affect outputs and require review. A difference raises a visible notice (stderr, web status); it does not block |
| Hardware, OS, timestamps, repository commit, Python/dependency versions, cost rates | Informational | Needed to reproduce and interpret results | Recorded as provenance; not automatically compared, but changes may affect outputs and require review |

**No revalidation-acceptance workflow exists.** "Revalidation required"
means the old artifact cannot be used with the changed settings: the runtime
refuses it exactly as it refuses a retrain-class mismatch. Nothing in the
code records a passed revalidation. The supported route to a usable artifact
is to collect outcomes under the new settings and train a replacement. That
is a statement about this codebase, not a claim that retraining is
necessary in principle; a future, separately approved workflow could
re-collect a sample, compare it with the old outcomes and record the result.

**Runtime versions.** The final experiment must freeze the tested local
runtime version (it is recorded per run in the manifest). A later version
change requires review and the relevant checks; it is neither proof that
outputs changed nor proof that the old evaluation remains valid, and an
upgraded deployment must not be described as verified until those checks
are done.

**Ollama metadata.** `model_identity` hashes the `/api/show` template exactly
as returned, and the parameters exactly as returned (text) or canonicalised
with sorted keys (structured) — no string inside is altered. A field the
response did not contain is recorded as `absent_in_runtime_response`; a
field that could not be read is `None` (unknown). Identities recorded before
these fields existed (identity version "1") are `unverifiable`, never
guessed compatible. Follow-up: the probe runs on every `LearnedRouter`
construction (the web UI builds one per hybrid request: three local HTTP
calls); this patch does not change that.

Statuses: `compatible` (accepted, with any notices shown), `retrain_required`
and `revalidation_required` (always refused, including in mock and test
use), `unverifiable` (an identity field is unknown on either side, an older
identity version, a names-only artifact from before manifests, or an active
request-scoped override), and `synthetic` (mock/toy training data).
`unverifiable` and `synthetic` artifacts are accepted only in mock mode or
with `LearnedRouter(..., allow_unverified=True)` in tests, with a warning;
production learned routing refuses them. `ROUTER_MODE=learned` then fails
at startup; `ROUTER_MODE=auto` falls back to the heuristic with a stderr
warning naming the status, and every decision is logged as `heuristic`.

**Batch routing is the initial validation target; learned Chat routing is
pending.** Chat, agent and tool requests apply request-scoped overrides
(conversation messages and style prompt, a 4,096-token output limit, context
profiles, tools; `chat/service.py`, `generation_context.py`). Outcomes are
collected without them, so whenever any override is active the identity is
`unverifiable` and learned routing is refused for that request. Heuristic
Chat is unchanged. Chat already maps `ROUTER_MODE=auto` to the heuristic
router; each hybrid Chat API response now says so (`routing.router_requested`
and `routing.router_note`), and `routing.router` reports `heuristic`.

**Finish status vs correctness.** Finish reasons are recorded per attempt
and summarised separately (`accounting.finish_reasons`). A reply with no
usable answer — empty, a refusal, or cut off at the output cap — is graded
0.0 with that flag: unsuccessful under the fixed generation budget, not
evidence that the model lacks the knowledge.

**Accounting.** `tasks_*` count unique tasks; `attempts_*` and
`strong_calls` count attempt rows (a task retried on resume adds attempts,
not tasks). Reported: attempted, completed, infrastructure-failed, usable
paired outcomes (both tiers graded — the only rows that become training
labels), unresolved reviews in total and per tier, failed attempts by tier,
strong calls completed / failed / skipped with the skip reason
(`cheap_tier_failed`: the paid strong call is not made when the cheap tier
fails), grade flags and finish reasons. Quality means use
`usable_paired_outcomes` as the denominator.

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
