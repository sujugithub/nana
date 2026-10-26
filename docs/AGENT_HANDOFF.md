# Agent handoff — learned router for the new local model pair

Durable rules: `AGENTS.md`. This file records current task progress only.
Facts only; no secrets.

## 1. Timestamp

2026-10-26 16:00 Australia/Sydney (AEDT). Milestone: S3 (pilot preparation,
licences, final-test isolation, runbook) complete; awaiting the S3 manual
Git checkpoint.

## 2. Current objective

Prepare, offline, the router-training and evaluation pipeline for the team's
new local model (reported as Qwen3.5-27B) paired with a Fireworks strong tier:
grading fixes (S1), reproducibility and pair-identity safeguards (S2),
pilot collection preparation (S3). The routing classifier is trained; the
answering models are not.

## 3. Acceptance criteria

- S0: project-local environment; baseline reported with passes, failures,
  skips, and setup errors separated. **Done.**
- S1: grading regression tests; explicit final-answer extraction; review
  status for contradictory/ambiguous answers; raw answers preserved; flags
  separate from correctness. **Done.**
- S2 (offline only): completion metadata; run/attempt IDs; raw-answer
  preservation; versioned manifest; identity-checked resume; infrastructure
  failures recorded and reported; identity separated from provenance;
  retrain-vs-revalidate rules documented then enforced; synthetic artifacts
  restricted to mock/test use; explicit UTF-8; exact and canonical
  fingerprints; accounting with denominators; mock-only tests. **Done**
  (sections 5 and 10).
- S3: GSM8K/MMLU/BBH pilot workflow (~60 tasks, for checking collection and
  grading only); licences verified individually; final-test tasks isolated;
  runbook with unknown deployment values marked Pending; long-context
  validation documented as a separate requirement. **Done** (section 5).
- Reporting rule for all stages: report all attempted tasks,
  completion/failure rates, grading coverage, and the denominator of every
  quality metric.

## 4. Branch and starting commit

- Branch: `feature/router-offline-prep`, tracking
  `origin/feature/router-offline-prep`.
- Base: `8b44cf6` (= `origin/main`).
- `b52d54b` "Add review-aware grading and regression tests" contains S0
  documents and S1 together (one commit, pushed by the owner).
- `79137c0` "Record attempt-level outcomes with verified pair identity"
  contains S2 + correction (26 files, pushed by the owner).
- S3 is uncommitted on top of `79137c0`.

## 5. Completed work

### S0
`.venv` (Python 3.14.4) with minimal offline dependencies; baseline recorded;
`AGENTS.md` and this file created.

### S1
`routing/grading.py`: `grade_answer(...) -> GradeResult` (status, score,
flags, extracted, method); explicit final answers first; contradictory /
ambiguous answers -> `needs_review`; empty/refusal/truncated/thinking flags;
grading copy only. Collector stores grades beside raw answers; review cases
use the `-1` manual-grade path; one-sided verdicts accepted.

### S2 (offline portions)

1. **Completion metadata** (`schemas.py`): optional `finish_reason`,
   `usage_source` (`provider` | `estimated` | `tokenizer` | `mock` | ""),
   `timings` (seconds), `raw_text` (before `.strip()`/tool-call text). All
   default to "not reported"; existing callers unchanged.
   - `ollama_client.py`: `done_reason`, provider counts, load/prompt/eval/
     total durations, raw text, for both plain and streaming calls. New
     `model_identity(model)` reads digest (`/api/tags`), quantisation/format/
     family/parameter size (`/api/show`) and runtime version (`/api/version`);
     missing facts are None, never guessed.
   - `remote_client.py`: `finish_reason`, `usage_source` (`estimated` when
     the provider omitted usage), raw text; streaming path too.
   - `local_model.py` (transformers): `finish_reason` inferred (`length`
     when the token cap was reached), `usage_source="tokenizer"`, raw text.
   - `config.py`: new opt-in `LOCAL_CONTEXT_TOKENS` (default 0 = unchanged
     behaviour). Chat's per-request context limits still take priority; no
     context safeguard was removed.
2. **Collector** (`scripts/collect_outcomes.py`): one JSONL row per attempt
   with `run_id`, `attempt_id`, `attempt_number`, `status`, `error`
   (`tier`/`type`/message truncated to 500 chars), raw and stripped answers,
   finish reasons, usage source, timings, grades. Exceptions are recorded
   as failed attempts, not wrong answers; a cheap-tier failure skips the
   paid strong call; `--max-consecutive-failures` (default 3) stops a run
   with exit code 3. Resume skips completed tasks, retries failed ones, and
   keeps the earlier failure rows. `--local-context-tokens` sets an explicit
   Ollama context for collection. Cost-guard wording no longer says
   "worst case"; it states what the ceiling excludes (transport retries,
   repeat attempts on resume).
3. **Manifest** (`<out>.manifest.json`, version 1): `compatibility`
   (pair identity, label identity = grader version + quality threshold,
   canonical task-file hash) and `provenance` (commit, dirty flag,
   repo-relative task path, exact task-file hash, mock flag, Python,
   platform, CPU, dependency versions, Ollama runtime facts, cost rates,
   budget, timestamps). Resume is refused if any compatibility value
   differs (message names the differing fields) or if a work file has no
   manifest. Provenance differences never block. Git is queried read-only
   with `--no-optional-locks`.
4. **Identity and compatibility** (`routing/identity.py`): retrain fields,
   revalidate fields, informational provenance; statuses `compatible`,
   `retrain_required`, `revalidation_required`, `unverifiable`, `synthetic`.
   Rules documented in `docs/LEARNED_ROUTING.md` ("Compatibility rules")
   before enforcement:
   - `LearnedRouter` accepts only `compatible`; `unverifiable`/`synthetic`
     only in mock mode or with `allow_unverified=True` (tests), with a
     warning; retrain/revalidate always refused. Ollama facts are probed at
     router construction (real mode only).
   - `webui/service.py`: `artifact_status` adds `verified` and
     `compatibility`; real-mode learned routing requires `verified`.
     `pair_match` keeps its existing meaning. `webui/static/index.html`
     shows "trained for exactly this pair" only when `verified`.
   - `GRADER_VERSION = "2"` in `routing/grading.py` ("1" = pre-S1 grader).
   - `scripts/merge_outcome_datasets.py` refuses to merge different
     `pair_identity` or `label_identity`; records canonical hashes.
5. **UTF-8**: explicit `encoding="utf-8"` on every text `open()` in
   `routing/`, `evaluation/run.py`, the collector, `build_real_pilot.py`,
   `calibrate.py`; writes use `newline="\n"` so files are byte-identical
   across OSes. The real 600-row dataset now loads without `PYTHONUTF8=1`.
6. **Fingerprints**: `file_sha256` (exact bytes) and
   `canonical_json_sha256` / `canonical_file_sha256` (parsed JSON, sorted
   keys, compact, ASCII escapes; string contents never normalised).
7. **Accounting** (`<out>.summary.json` and dataset `meta.collection`):
   tasks in file / not attempted / attempted / completed / failed
   (infrastructure) / graded / awaiting review; attempts total/failed; failed
   attempts by tier; grade flags per tier; completion rate,
   infrastructure-failure rate and grading coverage each with numerator and
   denominator; quality means with `denominator: tasks_graded` and `n`.

### S2 correction (owner-approved scope A–H)

- **A Accounting:** `strong_call` (`completed`/`failed`/`skipped`) and
  `strong_skipped_reason` (`cheap_tier_failed`) on every attempt row;
  summary adds `usable_paired_outcomes` (both tiers graded; the only rows
  that become labels), `unresolved_review_by_tier`, `strong_calls` (attempt
  counts with `skipped_by_reason`), `finish_reasons`. Quality denominator is
  `usable_paired_outcomes`. Task counts are unique IDs; attempt counts are
  rows (checked across a resume).
- **B Request-scoped chat identity:** `routing.identity.request_scoped_overrides()`
  detects Chat/agent overrides (messages, output limit, local context,
  tools); any active override makes the identity `unverifiable`, so
  learned routing is refused on that path (mock/test exemptions only).
  Heuristic Chat unchanged. Chat's existing `auto` -> heuristic mapping is
  now labelled in each hybrid API response (`routing.router_requested`,
  `routing.router_note`); the Chat UI does not render routing details and
  was not changed.
- **C Enforcement tests:** matrix over compatible / retrain / revalidation /
  unverifiable (unknown field, names-only) / synthetic for `LearnedRouter`,
  `build_router` learned and auto, and web artifact status (real and mock).
- **D Docs:** "No revalidation-acceptance workflow exists"; recollection +
  retraining is the supported route to a replacement artifact (not claimed
  necessary in principle); informational wording replaced with "Recorded as
  provenance; not automatically compared, but changes may affect outputs
  and require review."
- **E Effective settings:** `ollama_client.build_chat_payload` and
  `remote_client.build_chat_payload` are the single request builders used
  by generation (plain and streaming) and by the identity, which renders a
  placeholder request and records `request_sha256` (everything sent except
  the user prompt, output cap, context window and transport-only keys),
  system-prompt hash, temperature, think, output cap, context window.
  Tests show a changed system prompt or a changed request builder ->
  `retrain_required`; changed context -> `revalidation_required`;
  overrides -> `unverifiable`; generation sends exactly the builder output.
- **F Ollama template/parameters:** `model_identity` adds
  `template_sha256` / `parameters_sha256` from the existing `/api/show`
  call (no extra request): text hashed exactly as returned, structured
  values canonicalised (sorted keys, strings untouched), missing field ->
  `absent_in_runtime_response`, unreadable -> `None`. Retrain-class fields.
  `IDENTITY_VERSION` "2"; version "1" identities are `unverifiable`.
- **G Runtime version:** `informational.cheap_runtime_version` in the
  identity; a difference produces a notice (stderr from `LearnedRouter`,
  `notices` in web status; the demo UI shows a warning instead of "exactly
  this pair") and never blocks. Resume ignores informational differences
  with a warning; each run records `local_runtime_version`.
- **H Provenance tests:** manifest provenance and run-entry fields asserted.
- **Truncation policy:** docstrings/docs state that no usable answer =
  unsuccessful under the fixed generation budget, not proof of missing
  knowledge; finish reasons summarised separately. Grading semantics
  unchanged.

### S3 (pilot preparation; offline, no downloads, no model calls)

- **Licences verified individually** (2026-10-26, each repository's LICENSE
  file, URL-only fetches): GSM8K MIT (c) 2021 OpenAI; MMLU MIT (c) 2020 Dan
  Hendrycks; BIG-Bench Hard MIT (c) 2022 suzgunmirac; **BBH's upstream
  BIG-bench is Apache-2.0** (attribution/notice obligation). The historical
  task manifests' "all MIT" is incomplete; left unchanged as history.
- **No downloads:** `scripts/build_real_pilot.py` needs raw dataset copies,
  so S3 partitions the 600 already-tracked graded tasks instead.
- **`scripts/partition_tasks.py`** (deterministic, seed 26) ->
  `tasks/pair_v2/`: `pilot.json` 60 (20 GSM8K, 20 MMLU, 5 per BBH
  subtask), `development_remaining.json` 390, `final_test.json` 150
  (50 per category: GSM8K 50, MMLU 50, causal 17, date 17, logical 16),
  `manifest.json` (licences, exposure, contamination, quotas, fingerprints).
  Refuses to overwrite without `--force`.
  - Effective groups = declared groups merged across near-duplicates
    (same rule/threshold as training): 523 groups. Final test uses whole
    groups only; none straddles final test and development.
  - Previous pair's test split reconstructed (seed 7, 70/15/15 -> 420/90/90,
    matching the documented sizes; original `splits.json` unavailable):
    its 90 tasks are excluded from the final test.
  - Finding: all 50 `bbh-boolean_expressions` prompts form ONE
    near-duplicate group (shared instruction template), so the subtask is
    absent from the final test; the pilot samples development tasks
    individually (pilot vs development is not an isolation boundary).
- **`scripts/check_collection.py`**: `identity` (deployment identity report
  for the inference machine; three local Ollama metadata calls, no
  generation; key reported only as set/unset; exit 1 if any field unknown)
  and `verify` (manifest/rows/dataset/summary consistency, task-file hash,
  exact + canonical checksums, `--write-checksums` in `sha256sum -c` format).
- **`docs/TRAINING_RUNBOOK.md`**: roles, Pending deployment values, Steps
  A–D (identity, offline tests, mock dry run, BLOCKED pilot command with
  approval facts, verification/review/transfer), final-test isolation,
  long-context validation as a separate requirement, licences. Configured
  ceilings at the repository's unverified rates: pilot $0.1544,
  development $1.0018, final test $0.3862 (one attempt per task; excludes
  transport retries and repeat attempts; not a worst case).
- Runbook Step B executed here in mock mode (`local_remote`, ollama backend
  configured, scratchpad output): 60/60 completed, 0 usable, 60 awaiting
  review, collector exit 1, `verify` consistent — matches the runbook.
- Recorded S4 requirement: `routing/train.py` splits one dataset 70/15/15
  and `evaluation/run.py` evaluates that split's test part; training on
  development outcomes and evaluating once on a separately collected final
  test needs a separately approved change.

Existing tests changed (no assertion weakened):
- `tests/test_learned_router.py`: two synthetic-artifact constructions pass
  `allow_unverified=True`; `test_learned_mode_loads_artifact` now uses a
  fully compatible `remote_pair` identity and also asserts
  `compatibility.ok` (exercises the production path).
- `tests/test_runtime.py`: one synthetic construction passes
  `allow_unverified=True`.
- `tests/test_real_pilot.py`: five `open()` calls gained
  `encoding="utf-8"` (failed on Windows without `PYTHONUTF8=1`).
- S2's own new tests, adjusted in the correction: identity fixtures gained
  template/parameter/runtime facts (missing values are now correctly
  `unverifiable`); the quality-denominator assertion follows the rename to
  `usable_paired_outcomes`.

## 6. Partially completed work

None. S4 not started.

## 7. Changed files

S3 (uncommitted, all new): `scripts/partition_tasks.py`,
`scripts/check_collection.py`, `tests/test_partition.py` (9 tests),
`docs/TRAINING_RUNBOOK.md`, `tasks/pair_v2/pilot.json`,
`tasks/pair_v2/development_remaining.json`, `tasks/pair_v2/final_test.json`,
`tasks/pair_v2/manifest.json`, plus this file. No existing file modified.

S2 (committed in `79137c0`):

New: `routing/identity.py`, `tests/test_identity.py` (25 tests),
`tests/test_collection.py` (20 tests).

Modified: `chat/service.py` (correction B: auto-fallback note only),
`schemas.py`, `config.py`, `.env.example`, `ollama_client.py`,
`remote_client.py`, `local_model.py`, `routing/grading.py`,
`routing/learned_router.py`, `routing/dataset.py`, `routing/train.py`,
`evaluation/run.py`, `scripts/collect_outcomes.py`,
`scripts/build_real_pilot.py`, `scripts/calibrate.py`,
`scripts/merge_outcome_datasets.py`, `webui/service.py`,
`webui/static/index.html`, `docs/LEARNED_ROUTING.md`,
`tests/test_learned_router.py`, `tests/test_runtime.py`,
`tests/test_real_pilot.py`, `docs/AGENT_HANDOFF.md`.

`routing/train.py`, `evaluation/run.py`, `scripts/build_real_pilot.py`,
`scripts/calibrate.py`, `routing/dataset.py`: encoding/newline arguments
only. `routing/train.py` already copies `dataset.meta` into the artifact, so
`pair_identity` flows to artifacts without a training change.

## 8. Pre-existing unrelated changes

- Untracked file in the repository root whose name begins `eline` followed
  by a private-use character (17,924 bytes, 2026-10-26 10:57), containing
  coloured `git diff --cached` output. Not created by the agent; still
  present. Do not stage it.

## 9. Commands executed

S0 environment (Git Bash, Windows 11):

```sh
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install --only-binary=:all: \
  requests==2.34.2 certifi==2026.7.22 pypdf==6.19.0 Pillow==12.3.0 \
  pypdfium2==5.14.0 numpy scikit-learn joblib fastapi httpx
.venv/Scripts/python.exe -m pip install --only-binary=:all: tzdata
```

Resolved: scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, joblib 1.6.0,
fastapi 0.162.0, starlette 1.10.0, pydantic 2.14.1, httpx 0.28.1,
requests 2.34.2, certifi 2026.7.22, pypdf 6.19.0, Pillow 12.3.0,
pypdfium2 5.14.0, tzdata 2026.5. Not installed: torch, transformers,
playwright.

S2 checks:

```sh
# targeted, UTF-8 mode OFF (proves the encoding fixes)
.venv/Scripts/python.exe -m unittest tests.test_identity tests.test_collection tests.test_grading
PYTHONUTF8=1 .venv/Scripts/python.exe test_harness.py
PYTHONUTF8=1 .venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m unittest discover -s tests -v          # UTF-8 mode off
# mock CLI over the real pilot task file, outputs in the agent scratchpad only
.venv/Scripts/python.exe scripts/collect_outcomes.py --tasks tasks/real_pilot_qwen_pro.json \
  --out <scratch>/pilot-mock.json --mock --tier-mode remote_pair
```

## 10. Actual test results

### S3 (working tree on 79137c0)

- `tests.test_partition`: 9/9 pass (UTF-8 mode off).
- Harness: PASS.
- Full suite: **329 run (320 + 9 new), 325 pass, 4 fail, 0 skipped** in
  both encoding modes; the 4 are the S0 Windows platform failures. **No new
  failures.**

### S2 + correction (working tree on b52d54b)

- Targeted (`test_identity` 25, `test_collection` 20, `test_grading` 25):
  70/70 pass with UTF-8 mode off.
- Harness: PASS.
- Full suite, `PYTHONUTF8=1`: **320 run (306 + 14 new), 316 pass, 4 fail,
  0 skipped**; the 4 are the S0 Windows platform failures. UTF-8 mode off:
  identical. **No new failures.**
- Mock CLI over the 120-task pilot file: 120 attempted, 120 completed,
  0 usable paired outcomes, 120 awaiting review (cheap 120, strong 120),
  0 strong calls skipped; exit 1 (no graded records), as expected.

### S2 before the correction

- New tests: `tests.test_identity` 14, `tests.test_collection` 17; with
  `tests.test_grading`: all pass with UTF-8 mode off.
- Harness: PASS.
- Full suite, `PYTHONUTF8=1`: **306 run (275 + 31 new), 302 pass, 4 fail,
  0 skipped.** The 4 are exactly the S0 Windows platform failures.
- Full suite, UTF-8 mode off: identical (306 / 302 / same 4).
- **No new failures.**
- Mock CLI over the 120-task pilot file: 120 attempted, 120 completed,
  0 infrastructure failures, 0 graded, 120 awaiting review (mock answers
  contain no option/number -> `no_final_answer`, none mis-scored as wrong);
  exit 1 = no graded records, as expected. Manifest marked `synthetic`.
- `data/real_qwen_pro_600.json` loads with UTF-8 mode off; content hash
  `30c51583d6c8ab07` unchanged.

### S1: 275 run, 271 pass, same 4 failures. S0: 250 run, 246 pass, same 4.

Windows platform failures (unchanged, not fixed, owner informed):

| Test | Cause |
|---|---|
| `test_agent_contracts…test_cancellation_stops_a_running_shell` (+ tearDown error) | `workspace/tools.py:202` `os.killpg` absent on Windows |
| `test_schedule_recovery…test_scheduler_has_one_owner_for_workspace` | `workspace/scheduler.py:103` imports `fcntl` |
| `test_workspace…test_mcp_stdio_handshake_and_discovery` | `workspace/mcp.py:53` `select` on pipes |
| `test_workspace…test_oauth_state_and_secrets_are_not_exposed` | expects mode `0o600`; Windows reports `0o666` |

`tzdata` is required on Windows and is not in the requirements files.
Earlier handoffs report 261 tests; commit `8b44cf6` defined 250 (unexplained).

## 11. Dataset, configuration, and artifact paths

- `data/real_qwen_pro_600.json` — Qwen2.5-1.5B-Instruct -> DeepSeek V4 Pro.
  **Not valid for the new pair.** No pair identity (pre-manifest) -> any
  router trained on it is `unverifiable` at runtime.
- `tasks/real_pilot_qwen_pro.json` (120), `tasks/real_extension_fast_qwen_pro.json`
  (480) — model-independent graded tasks.
- `artifacts/`, `reports/`: absent here. No `.env`; no Fireworks access here.

## 12. Fingerprints

Committed (LF) content, `git cat-file -p HEAD:<path> | sha256sum`:

| Path | SHA-256 |
|---|---|
| `data/real_qwen_pro_600.json` | `53d6428edf5669021a8a26228ef0ddbf50aff12d5b882c9432a82e4f3bd6bb52` |
| `tasks/real_pilot_qwen_pro.json` | `1f6f0833be46c98973b948c9d5a62907802ca61f4276bc9a0f0031c721dc5f7d` |
| `tasks/real_extension_fast_qwen_pro.json` | `dad1ace6cc622aeedc6da0e6c161a73ea5950b2e1384a633333ec03beac92175` |
| `tasks/real_pilot_qwen_pro.manifest.json` | `6d57986246df94283a4846411af7de7ceedd6f00fe2b7854277d8ecbdbcf1d32` |
| `tasks/real_extension_fast_qwen_pro.manifest.json` | `9f9c2a245808cdc153ebe5d155f06d9d541adc64e4042379a9d8acd262b56692` |

S3 partition (`tasks/pair_v2/`, as written on Windows with LF endings;
canonical values are in `manifest.json` and checked by tests):

| File | Exact SHA-256 | Canonical SHA-256 |
|---|---|---|
| `pilot.json` | `1de9c59494ec23a1702b26afe12e943fbd05b6f513b4389c6612a24ee71158f8` | `0fc2c6015035baf11b00254fed2f862ccd3338f225f2dbe7858297ab0d3fb17f` |
| `development_remaining.json` | `8ab7eee397174aa2d6655e2893b69f343da0a526e74b1f267c88fe66051224d6` | `5c49307e9fc154985b06a8efc55b92f9a7d7930bbd86b8d45748c1a0969a7468` |
| `final_test.json` | `18eb32672bff302a15ac6fffe0046842f8f8314e22d16b553e42c48dba2901b0` | `bdc681f514bec5d337915618521e9d0840c8f4563620f6a8d9e9f5760f6b89dd` |
| `manifest.json` | `4e4090238cda0b63d5f8031c8c36cc9c0d18fb3706d7949fe4b91d8f912e5bda` | — |

Exact values may change after a Git round trip on a CRLF checkout; the
canonical values do not.

Canonical (line-ending independent):
- `tasks/real_pilot_qwen_pro.json`:
  `89c118518d41e1536894dc3b3a64c535ef2b80cab650eeda9de3cbf8d147c6cc`
- `data/real_qwen_pro_600.json` `content_hash`: `30c51583d6c8ab07`.

This checkout uses CRLF (`core.autocrlf=true`), so raw working-tree hashes
of tracked files differ from the committed values.

## 13. Active or interrupted processes

None.

## 14. Resume instructions

1. Read `AGENTS.md`, then this file.
2. Confirm branch `feature/router-offline-prep`; HEAD `b52d54b` plus the S2
   working-tree changes (or the S2 commit, once the owner makes it).
3. Use `.venv/Scripts/python.exe`. `PYTHONUTF8=1` is no longer required for
   the data pipeline but keeps console output safe.
4. Re-run the suite; expect 325/329 with the four platform failures.

## 15. Pending approvals

- S3 manual Git checkpoint.
- Teammate Step A (identity report, offline tests, mock dry run) — no
  paid calls; needs the owner to coordinate.
- Pilot collection (runbook Step C): paid; needs verified pricing, an
  explicit budget, the confirmed pair, and owner approval.
- S4 change to train on development outcomes and evaluate a separate
  final-test dataset once.
- Paid collection stays blocked until retry/resume spending controls and
  the actual model-pair configuration are settled (owner instruction).
- Not approved: real-model or paid collection, Fireworks calls, model or
  dataset downloads, final training for an unconfirmed pair, final held-out
  evaluation, artifact deployment, broad chat-routing changes, Git writes.

## 16. Blockers and open decisions

- **Pending — teammate evidence:** exact local model identity (tag,
  digest, quantisation, backend), intended context length, generation
  settings. Checkout contains no Qwen3.5-27B configuration; only
  `qwen3.8:27b` (verified at 2,048 context). `ollama_client.model_identity`
  can now record digest/quantisation on the inference machine.
- **Pending — teammate evidence:** large-context capability (none beyond
  2,048 tokens evidenced).
- **Pending:** strong model actually configured on the inference machine.
- No outcomes exist for any current model pair.
- **Learned Chat routing: pending.** Batch routing is the initial
  validation target. Any request-scoped override makes the path
  `unverifiable` and learned routing is refused there (S2 correction B).
- **No revalidation-acceptance workflow** (documented). Revalidate-class
  changes are refused until a replacement artifact is trained.
- Limits: the strong model's revision is not exposed by Fireworks
  (recorded as such); transformers-backend identities are always
  `unverifiable`; that the Ollama digest covers template/parameter layers
  is unverified (both are now recorded separately); the cost ceiling
  excludes transport retries and repeat attempts.
- **Follow-ups (not done, by instruction):** Ollama facts are probed on
  each `LearnedRouter` construction (web UI: one per hybrid request, three
  local HTTP calls); GPU/memory/runner discovery and live Ollama probes
  deferred (H); a recorded revalidation workflow would need separate
  approval.

## 17. Exact next action

Owner performs the S3 checkpoint, then sends runbook Steps A–B to the
teammate and returns `identity.json` plus the test/dry-run results. Agent
work after that: confirm the pair from `identity.json`, prepare the pilot
approval request with verified pricing, and propose the S4 change.

## 18. Manual Git checkpoint due?

Yes — S3 "Pilot preparation and runbook".
