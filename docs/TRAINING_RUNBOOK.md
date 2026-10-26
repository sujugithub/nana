# Training runbook — outcome collection for the new model pair

Status (2026-10-26): **preparation only.** Paid collection is blocked until
the owner approves it, the actual model-pair configuration is confirmed
(Step A), and retry/resume spending controls are settled. Nothing here has
been run against a real model.

Related: `docs/LEARNED_ROUTING.md` (pipeline, compatibility rules),
`docs/AGENT_HANDOFF.md` (progress), `tasks/pair_v2/manifest.json`
(task provenance).

## 1. Who runs what

| Work | Where | Needs |
|---|---|---|
| Steps A–B (identity report, offline tests, mock dry run) | Inference machine (teammate) | Local model installed; no Fireworks calls |
| Step C (pilot collection) | Inference machine | Owner approval; Fireworks key on that machine |
| Step D (verification, review queue) | Either machine | Collected files |
| Human grading of review cases | Owner/team | Raw answers from the work file |
| Training and evaluation (later stages) | Any machine | Verified dataset |

All Git operations are done by the owner or teammate, not by an agent.

## 2. Deployment values

| Value | Status |
|---|---|
| Local model tag (reported as Qwen3.5-27B) | **Pending** — the checkout only contains `qwen3.8:27b` (verified at 2,048 context). Step A records the exact tag |
| Digest, quantisation, template and parameter hashes | **Pending** — recorded by Step A |
| Local backend | **Pending** — expected `ollama`; confirm in Step A |
| Local context window (`LOCAL_CONTEXT_TOKENS`) | **Pending** — must be set explicitly; no evidence beyond 2,048 tokens for the 27B model (see §8) |
| Local output cap (`LOCAL_MAX_NEW_TOKENS`) | Default 512 in code — confirm before the pilot |
| Thinking mode | `think: false` (sent by `ollama_client.build_chat_payload`) |
| Temperature | 0 for both tiers |
| Strong model | **Pending** — code default `accounts/fireworks/models/gpt-oss-120b`; confirm on the inference machine |
| Strong output cap (`STRONG_MAX_TOKENS`) | Default 4096 — confirm |
| Fireworks prices | **Pending verification** — repository values are $0.15 in / $0.60 out per million tokens for the strong tier; unverified |
| Ollama runtime version | **Pending** — recorded by Step A; freeze it for the final experiment |

## 3. Step A — identity report and offline checks (no paid calls)

On the inference machine, from the repository root, on the branch the owner
names:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install requests certifi numpy scikit-learn joblib fastapi httpx pypdf Pillow pypdfium2
.venv/bin/python -m unittest discover -s tests          # report the result line
```

Set the deployment (values are Pending — fill in the confirmed ones). Use the
address of the runtime that actually serves the model:

```sh
export TIER_MODE=local_remote
export LOCAL_BACKEND=ollama
export LOCAL_MODEL_NAME='<pending: exact Ollama tag>'
export OLLAMA_BASE_URL='http://127.0.0.1:<port>'
export LOCAL_CONTEXT_TOKENS='<pending: explicit num_ctx>'
export STRONG_MODEL_NAME='<pending: exact Fireworks model id>'
```

Record the identity. This makes three local Ollama metadata requests
(`/api/show`, `/api/tags`, `/api/version`) and no generation or Fireworks
call:

```sh
.venv/bin/python scripts/check_collection.py identity --json-out identity.json
```

Exit code 0 means every identity field is known; 1 lists `unknown_fields`
(and `probe_error` if Ollama could not be reached). Send `identity.json` to
the owner. It contains model names, hashes, the Ollama host name, numeric
settings, and only whether a Fireworks key is set — never the key.

## 4. Step B — mock dry run (no models, no network, no cost)

```sh
.venv/bin/python scripts/collect_outcomes.py --tasks tasks/pair_v2/pilot.json \
  --out data/pilot_mock.json --mock
.venv/bin/python scripts/check_collection.py verify --out data/pilot_mock.json \
  --tasks tasks/pair_v2/pilot.json
```

Expected: 60 attempted and completed, 0 usable paired outcomes, 60 awaiting
review (mock answers contain no option letter or number), exit 1 from the
collector ("no graded records"), `consistent` from `verify`, and a
`synthetic` note. Delete `data/pilot_mock.json*` afterwards.

## 5. Step C — pilot collection (BLOCKED until approved)

Purpose: check collection, grading and accounting on 60 tasks (20 GSM8K, 20
MMLU, 5 from each of four BBH subtasks). It is **not** a statistical result.

Command for the approval request (do not run before approval):

```sh
.venv/bin/python scripts/collect_outcomes.py --tasks tasks/pair_v2/pilot.json \
  --out data/pilot_v2.json --run-paid-calls --max-api-cost-usd '<approved budget>' \
  --local-context-tokens "$LOCAL_CONTEXT_TOKENS" --max-consecutive-failures 3
```

Facts for the approval request:

| Item | Value |
|---|---|
| Models | Local tier: Pending (Step A). Strong tier: Pending Fireworks model |
| Data sent externally | The 60 pilot prompts (public benchmark items) and the system prompt, to Fireworks only |
| Calls | 60 local generations; 60 Fireworks calls (one attempt per task), plus any transport retries inside `remote_client` (429/5xx/connection errors, up to `MAX_RETRIES=3`) |
| Configured cost ceiling | $0.1544 at the repository's unverified rates, assuming every reply uses the full 4,096-token cap. Excludes transport retries and repeat attempts on resume; not a worst case. Verify prices first |
| Runtime | Pending — not measured for these prompts on the 27B model |
| Output | `data/pilot_v2.json` plus `.manifest.json`, `.work.jsonl`, `.summary.json` (all Git-ignored) |
| Resume | Rerun the same command: completed tasks are skipped, failed ones retried with a new attempt ID. Refused if the deployment, label definition or task file changed |
| Stops | Budget check fails before any call; 3 consecutive failed attempts (exit 3); Ctrl-C (rows already written are kept) |

The cheap tier runs first; if it fails, the Fireworks call is skipped and
recorded (`strong_skipped_reason: cheap_tier_failed`).

Larger runs, for later approval: `development_remaining.json` has 390 tasks
(configured ceiling $1.0018); `final_test.json` has 150 (configured ceiling
$0.3862, reserved — see §7).

## 6. Step D — verification, review and transfer

```sh
.venv/bin/python scripts/check_collection.py verify --out data/pilot_v2.json \
  --tasks tasks/pair_v2/pilot.json --write-checksums data/pilot_v2.sha256
```

Check `data/pilot_v2.json.summary.json`: attempted, completed,
infrastructure failures, usable paired outcomes, unresolved reviews per
tier, strong calls skipped, and finish reasons. Every quality figure states
its denominator (`usable_paired_outcomes`).

Review queue: work rows whose `local_grade` or `remote_grade` has
`status: needs_review`. Reviewers read the raw answers
(`local_answer_raw`, `remote_answer_raw`) and record verdicts as
`{"task_id": {"cheap": 0 or 1, "strong": 0 or 1}}` (a side may be omitted
when only the other needed review), then:

```sh
.venv/bin/python scripts/collect_outcomes.py --tasks tasks/pair_v2/pilot.json \
  --out data/pilot_v2.json --finalize-only --grades data/pilot_v2.verdicts.json
```

Document who reviewed which cases. Do not regrade cases because the result
looks unfavourable.

Transfer to the owner, through a private channel the team controls (for
example a privately shared university drive folder — not a public repository,
issue, or chat with public links):

- `data/pilot_v2.json`, `.manifest.json`, `.work.jsonl`, `.summary.json`,
  `data/pilot_v2.sha256`, and `identity.json`.

On receipt: `sha256sum -c pilot_v2.sha256` (from the folder holding the
files) and the `verify` command above. Exact checksums match across macOS and
Windows because the pipeline writes UTF-8 with LF line endings.

Never transfer or commit: `.env`, API keys, OAuth tokens, `logs/`, chat
databases, uploads or workspace data, model weights, local runtimes, `.venv/`,
or personal absolute paths. Do not commit `data/` outputs (Git-ignored).

## 7. Final-test isolation

`tasks/pair_v2/final_test.json` (150 tasks) is reserved:

- Do not collect, inspect, grade or tune on it during development.
- Collect it only after the classifier, threshold, grading, split and model
  pair are frozen, with separate owner approval, under the same identity.
- Evaluate it once. Report the result whatever it is.

It excludes the 90 tasks of the previous pair's (reconstructed) test split
and every near-duplicate group touching development. It contains no
`bbh-boolean_expressions` tasks: all 50 share an instruction template and
form one inseparable near-duplicate group, which stays in development.

Open requirement for the training stage: `routing/train.py` currently makes
its own 70/15/15 split of one dataset, and `evaluation/run.py` evaluates that
split's test part. Training on development outcomes and evaluating once on a
separately collected final-test dataset needs a small, separately approved
change.

## 8. Long-context validation — separate requirement

No claim about large contexts (for example 100K tokens) may be made from
this pilot. The pilot prompts are short; the repository's only 27B evidence
is at a 2,048-token context, with heavy swap growth on a 24 GB machine.
Before any long-context claim, a separate, approved validation must:

- fix the context window explicitly and record it (identity + manifest);
- use tasks whose required information sits at known positions across the
  window, with deterministic grading;
- record memory use, load and generation timings, truncation and failures;
- compare against the short-context results on the same model.

Until then, the router is validated only for the recorded context window.

## 9. Task sources and licences

Verified 2026-10-26 from each repository's LICENSE file:

| Source | Licence |
|---|---|
| GSM8K — github.com/openai/grade-school-math | MIT, Copyright (c) 2021 OpenAI |
| MMLU — github.com/hendrycks/test | MIT, Copyright (c) 2020 Dan Hendrycks |
| BIG-Bench Hard — github.com/suzgunmirac/BIG-Bench-Hard | MIT, Copyright (c) 2022 suzgunmirac |
| BIG-bench (BBH's upstream tasks) — github.com/google/BIG-bench | Apache-2.0: keep its attribution and notice when redistributing BBH items |

These public benchmarks may be in either model's training data, which can
inflate absolute accuracy; the study compares the two tiers on identical
prompts. All 600 prompts were also used with the previous pair.
