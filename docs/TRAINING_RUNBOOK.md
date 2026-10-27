# Training runbook — outcome collection for the new model pair

Status (2026-10-28): Steps A–B done on the inference machine; model pair
recorded (§2); context frozen at 2,048; prices verified; cumulative
spending control in place. **The paid pilot (Step C) is still blocked until
the owner approves the exact command and budget.** Nothing here has been run
against a real model.

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

Step A identity report received 2026-10-28 (`identity.json`, generated
2026-10-27 on the inference machine: macOS 26.6 arm64, Python 3.14.8). It
has no unknown fields, and its rendered request hashes for both tiers equal
those produced by commit `1772fdd`, so the machine runs the same request
code and system prompts.

| Value | Status |
|---|---|
| Local model | **Recorded:** exact Ollama tag `qwen3.8:27b` (teammate: do not rename it); Ollama reports family `qwen3_5`, 27.8B parameters, safetensors format. Reports should name the exact tag and cite the family as Ollama metadata |
| Digest | `47951e4a666180bc2aedfc93c169472f2de783696fe196efbf063f5730ecf163` |
| Quantisation | `nvfp4` |
| Template / parameters SHA-256 | `b507b9c2…fb315` / `3c1eabbc…00980` (full values in `identity.json`) |
| Local backend | `ollama`, loopback host |
| Local context window | **Frozen at 2,048** for this study (owner decision 2026-10-28; the only tested value). Not a large-context configuration (see §8). Estimated (chars/4, not a tokenizer count) largest pilot request ≈ 391 system + 572 prompt + 512 output ≈ 1,475 tokens |
| Local output cap | 512 |
| Thinking mode / temperature | `think: false` / 0 |
| Strong model | `accounts/fireworks/models/gpt-oss-120b`, output cap 4096, temperature 0 |
| Ollama runtime version | 0.40.0 — freeze for the final experiment |
| Fireworks key on the inference machine | **Not set** (reported as unset) — required before Step C; set privately in the ignored `.env` |
| Fireworks prices | **Verified 2026-10-28** on fireworks.ai/models/fireworks/gpt-oss-120b: $0.15 input / $0.015 cached input / $0.60 output per million tokens (131k context). Equal to the repository rates; the ceiling charges all input at $0.15 (never the cached rate) |
| Offline tests on the inference machine | **Received** (teammate report, 2026-10-27): harness passed; 329 tests, OK (the 4 Windows-only failures pass on macOS) |
| Step B mock dry run | **Received:** 60/60 completed, 0 usable, 60 awaiting review per tier, collector exit 1, `verify` consistent — as expected |
| Source used | Commit `1772fdd` exported without `.git` (read-only blob export), so a manifest written there records `repository_commit: null`. For Step C, run from a real checkout of the approved commit so the manifest records it |

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

Run from a real Git checkout of the approved commit (so the manifest
records it), with the Step A exports set and a Fireworks key in the ignored
`.env`. Command for the approval request (do not run before approval):

```sh
.venv/bin/python scripts/collect_outcomes.py --tasks tasks/pair_v2/pilot.json \
  --out data/pilot_v2.json --run-paid-calls --max-api-cost-usd 0.50 \
  --local-context-tokens 2048 --max-consecutive-failures 3
```

`--max-api-cost-usd` is the **total budget for this `--out` across every
run and resume**. Each attempt records its spend; a call that failed after
being sent, or whose usage the provider omitted, is charged at its full
per-call ceiling. A run is refused when already-charged + the ceiling for
unfinished tasks exceeds the budget, and stops (exit 4) before a task that
could cross it.

Facts for the approval request:

| Item | Value |
|---|---|
| Models | Cheap: local Ollama `qwen3.8:27b` (digest `47951e4a…ecf163`, nvfp4, Ollama 0.40.0), context 2,048, output cap 512, think off, temperature 0. Strong: Fireworks `accounts/fireworks/models/gpt-oss-120b`, output cap 4,096, temperature 0 |
| Data sent externally | The 60 pilot prompts (public GSM8K/MMLU/BBH items) and the base system prompt, to Fireworks only. No chat data, uploads or credentials |
| Calls | 60 local generations (free; local compute); 60 Fireworks calls, one attempt per task. Transport retries inside `remote_client` (429/5xx/connection errors, up to 3) are possible and assumed unbilled — not verified |
| Cost | Configured ceiling $0.1544 at verified prices, assuming every reply uses the full 4,096-token cap and every input byte is a token. Typical spend should be far lower (the previous 60-task 4B pilot spent $0.0075), but that is not a prediction for this pair |
| Budget ceiling | Proposed total $0.50: the full ceiling plus room to retry failed tasks on resume (failed calls are charged at their ceiling) |
| Runtime | Not measured for these prompts. Earlier smoke tests on this model: ~29 s cold load; ~7 output tokens/s (25 tokens in 3.5 s; 45 in 7.1 s). At 512 output tokens a local answer could take ~75 s; a rough pilot range is 20–90 minutes |
| Known risk | The Ollama request timeout is 120 s: a cold model load plus a long answer can time out and be recorded as an infrastructure failure. Warm the model first. Earlier 27B trials showed heavy swap growth and one runtime outage on the 24 GB Mac — keep other large apps closed |
| Output | `data/pilot_v2.json` plus `.manifest.json`, `.work.jsonl`, `.summary.json` (all Git-ignored) |
| Resume | Rerun the same command: completed tasks are skipped, failed ones retried with a new attempt ID; refused if the deployment, label definition or task file changed, or if the cumulative budget would be exceeded |
| Stops | Budget refusal before any call; 3 consecutive failed attempts (exit 3); budget reached mid-run (exit 4); Ctrl-C (rows already written are kept) |

The cheap tier runs first; if it fails, the Fireworks call is skipped and
recorded (`strong_skipped_reason: cheap_tier_failed`).

Larger runs, for later approval: `development_remaining.json` has 390 tasks
(configured ceiling $1.0018 at the verified prices); `final_test.json` has
150 (configured ceiling $0.3862, reserved — see §7). Use a separate `--out`
and budget for each.

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
