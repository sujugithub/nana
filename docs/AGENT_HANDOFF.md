# Agent handoff — learned router for the new local model pair

Durable rules: `AGENTS.md`. This file records current task progress only.
Facts only; no secrets.

## 1. Timestamp

2026-10-26 11:30 Australia/Sydney (AEDT). Milestone: S1 grading safeguards
complete; awaiting the S1 manual Git checkpoint.

## 2. Current objective

Prepare, offline, the router-training and evaluation pipeline for the team's
new local model (reported as Qwen3.5-27B) paired with a Fireworks strong tier:
grading fixes (S1), reproducibility and pair-identity safeguards (S2),
pilot collection preparation (S3). The routing classifier is trained; the
answering models are not.

## 3. Acceptance criteria

- S0: project-local environment; harness and unit-test baseline reported
  with passes, failures, skips, and setup errors separated. **Done.**
- S1: every demonstrated grading error has a regression test; explicit
  final-answer extraction; contradictory/ambiguous answers flagged for
  review; raw answers preserved unchanged (thinking-block removal only on a
  grading copy); refusal, truncation, empty output, and grading uncertainty
  recorded separately from correctness. **Done** (section 10).
- S2 (offline only): backward-compatible completion metadata; run/attempt
  IDs; raw-answer preservation; run manifest; identity-checked resume;
  infrastructure failures recorded and reported, not silently dropped or
  counted as wrong answers; model/prompt/generation identity separated from
  informational provenance; learned routing refused for incompatible or
  unverifiable artifacts with a visible heuristic fallback; written
  retraining-vs-revalidation rule; tests using mock fixtures only.
- S3: GSM8K/MMLU/BBH pilot workflow (~60 tasks, for checking collection and
  grading only); licences verified individually; final-test tasks isolated;
  runbook with unknown deployment values marked Pending; long-context
  validation documented as a separate requirement.
- Reporting rule for all stages: report all attempted tasks,
  completion/failure rates, grading coverage, and the denominator of every
  quality metric.

## 4. Branch and starting commit

- Branch: `feature/router-offline-prep` (created by the owner).
- Starting commit: `8b44cf6` (= `origin/main` after the owner's fetch at
  2026-10-26 04:45 AEDT). No remote branch is newer.

## 5. Completed work

- Phase 1 audit (read-only).
- S0: `.venv` created; minimal dependencies installed; harness and unit-test
  baseline recorded (section 10).
- Created `AGENTS.md` and this file.
- S1 grading safeguards:
  - New `routing/grading.py` (stdlib only): `grade_answer(grader,
    reference, answer, finish_reason=None) -> GradeResult` with `status`
    (`graded` | `needs_review`), `score`, `flags`, `extracted`, `method`.
  - Explicit final-answer extraction first ("Answer: B", "the answer is
    42", `#### 42`, `\boxed{}`); then a leading standalone answer; then a
    single mentioned option or a trailing final number.
  - Conflicting explicit answers -> `needs_review` + `contradictory`;
    leading answer plus other options mentioned -> `needs_review` +
    `ambiguous`. No negation interpretation is attempted.
  - Flags recorded separately from correctness: `empty`, `refusal`,
    `truncated` (from `finish_reason` "length"/"max_tokens"),
    `thinking_removed`, `unclosed_thinking`, `no_final_answer`.
  - Thinking blocks and markdown emphasis are removed on a grading copy
    only; raw answers are unchanged.
  - `scripts/collect_outcomes.py`: `grade()` keeps its name and float
    interface and delegates (review cases return 0.0). `collect()` uses
    `grade_answer`, stores `local_grade`/`remote_grade` dicts beside the
    raw answers, and routes review cases to the existing `-1` manual-grade
    path (`--grades`). `finalize()` accepts a verdict for one tier only.
    Removed the now-unused `_NUM_RE` and `import re`.
  - `contains` and `exact` semantics are unchanged (substring/equality on
    the grading copy).

## 6. Partially completed work

None. S2 not started.

## 7. Changed files (this task)

- `AGENTS.md` (new; staged by the owner, not yet committed — see 18)
- `docs/AGENT_HANDOFF.md` (new; staged S0 version, S1 edits unstaged)
- `routing/grading.py` (new)
- `tests/test_grading.py` (new, 25 tests)
- `scripts/collect_outcomes.py` (modified)
- `routing/__init__.py` (modified: one docstring line)

`.venv/` is local and Git-ignored.

## 8. Pre-existing unrelated changes

- Untracked file in the repository root whose name begins `eline` followed
  by a private-use character (17,924 bytes, created 2026-10-26 10:57). It
  contains coloured `git diff --cached` output — likely a mangled shell
  redirect. Not created or modified by the agent; the owner decides whether
  to delete it. Do not stage it.

## 9. Commands executed (S0)

Run from the repository root in Git Bash on Windows 11.

```sh
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install --only-binary=:all: \
  requests==2.34.2 certifi==2026.7.22 pypdf==6.19.0 Pillow==12.3.0 \
  pypdfium2==5.14.0 numpy scikit-learn joblib fastapi httpx
.venv/Scripts/python.exe -m pip install --only-binary=:all: tzdata
PYTHONUTF8=1 .venv/Scripts/python.exe test_harness.py
PYTHONUTF8=1 .venv/Scripts/python.exe -m unittest discover -s tests -v
```

Interpreter choice: Python 3.14.4. Python 3.13 and 3.14 both resolved binary
wheels for every required package. 3.14 was chosen because
`requirements-workspace.txt` records 3.14.8 as the version verified on the
team's macOS machine. Pins from `requirements-workspace.txt` were reused.

Resolved versions: scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1,
joblib 1.6.0, fastapi 0.162.0, starlette 1.10.0, pydantic 2.14.1,
httpx 0.28.1, requests 2.34.2, certifi 2026.7.22, pypdf 6.19.0,
Pillow 12.3.0, pypdfium2 5.14.0, tzdata 2026.5.

Not installed (not needed offline): torch, transformers (only the
`transformers` local backend), playwright (optional browser tool),
`scripts/evals/engine_spike.py` dependencies.

S1 commands:

```sh
PYTHONUTF8=1 .venv/Scripts/python.exe -m unittest tests.test_grading -v
PYTHONUTF8=1 .venv/Scripts/python.exe -m unittest tests.test_grading tests.test_real_pilot tests.test_runtime
PYTHONUTF8=1 .venv/Scripts/python.exe test_harness.py
PYTHONUTF8=1 .venv/Scripts/python.exe -m unittest discover -s tests -v
```

## 10. Actual test results

### S1 (working tree on 8b44cf6)

- `tests.test_grading`: 25/25 pass.
- Targeted (`test_grading`, `test_real_pilot`, `test_runtime`): 51/51 pass.
- Harness: PASS.
- Full suite: 275 run (250 baseline + 25 new), **271 pass, 4 fail, 0
  skipped**. The 4 failures are exactly the S0 Windows platform failures
  below. **No new failures.**
- Offline check (not a test file): all 600 references in the two real task
  files self-grade as correct in four answer formats each (raw, `Answer:`,
  `(X)`/`X.` for choice, `#### N` for numeric); 0 rejected.
- The commit-8b44cf6 grader produces the wrong verdict on every regression
  input in `tests/test_grading.py` (article "a" read as option A; "not B"
  read as B; "not True" read as True; "18 dollars over 2 days" read as 2;
  "1/2" unparsed; two different final numbers accepted; "A bakery…" read
  as option A).

### S0 baseline (commit 8b44cf6)

- `test_harness.py`: PASS ("All routing assertions passed").
- Unit tests: 250 run (250 test methods defined in the checkout), **246 pass,
  4 fail, 0 skipped**. All routing, dataset, features, splits, policies,
  training/artifact, evaluation, real-pilot, runtime, webui, and chat test
  modules pass.

Setup failure (resolved): `test_daily_sydney_handles_dst_gap_and_fold` needed
`tzdata` (Windows has no system time-zone database). Passes after installing
it. `tzdata` is not listed in the requirements files.

Platform failures (Windows vs POSIX code paths; not fixed, owner informed):

| Test | Cause |
|---|---|
| `test_agent_contracts…test_cancellation_stops_a_running_shell` (error, plus a tearDown error) | `workspace/tools.py:202` calls `os.killpg`, absent on Windows; the child process keeps the temp directory locked |
| `test_schedule_recovery…test_scheduler_has_one_owner_for_workspace` | `workspace/scheduler.py:103` imports `fcntl` (POSIX only) |
| `test_workspace…test_mcp_stdio_handshake_and_discovery` | `workspace/mcp.py:53` uses `select` on pipes (sockets only on Windows) |
| `test_workspace…test_oauth_state_and_secrets_are_not_exposed` | asserts file mode `0o600`; Windows reports `0o666` |

Unexplained: earlier handoffs report 261 passing tests; this commit defines
250. Not investigated further.

## 11. Dataset, configuration, and artifact paths

- `data/real_qwen_pro_600.json` — real outcomes for Qwen2.5-1.5B-Instruct →
  DeepSeek V4 Pro. **Not valid for the new pair.** Raw answers not stored.
- `tasks/real_pilot_qwen_pro.json` (120) and
  `tasks/real_extension_fast_qwen_pro.json` (480) — model-independent graded
  task files with licence/SHA manifests.
- `artifacts/`, `reports/`: absent on this machine (Git-ignored).
- No `.env` on this machine; no Fireworks access here.
- Configured defaults (`config.py`): `TIER_MODE=local_remote`, strong
  `accounts/fireworks/models/gpt-oss-120b`, `ROUTER_MODE=heuristic`.

## 12. Fingerprints

SHA-256 of committed (LF) content via `git cat-file -p HEAD:<path>`:

| Path | SHA-256 |
|---|---|
| `data/real_qwen_pro_600.json` | `53d6428edf5669021a8a26228ef0ddbf50aff12d5b882c9432a82e4f3bd6bb52` |
| `tasks/real_pilot_qwen_pro.json` | `1f6f0833be46c98973b948c9d5a62907802ca61f4276bc9a0f0031c721dc5f7d` |
| `tasks/real_extension_fast_qwen_pro.json` | `dad1ace6cc622aeedc6da0e6c161a73ea5950b2e1384a633333ec03beac92175` |
| `tasks/real_pilot_qwen_pro.manifest.json` | `6d57986246df94283a4846411af7de7ceedd6f00fe2b7854277d8ecbdbcf1d32` |
| `tasks/real_extension_fast_qwen_pro.manifest.json` | `9f9c2a245808cdc153ebe5d155f06d9d541adc64e4042379a9d8acd262b56692` |

Dataset content hash (`routing.dataset.content_hash`, line-ending
independent): `30c51583d6c8ab07`, 600 records — matches
`docs/LEARNED_ROUTING.md`.

This Windows checkout uses CRLF (`core.autocrlf=true` from the Git for
Windows system config), so raw working-tree hashes differ from the values
above.

## 13. Active or interrupted processes

None.

## 14. Resume instructions

1. Read `AGENTS.md`, then this file.
2. Confirm branch `feature/router-offline-prep` and a clean or expected tree.
3. Use `.venv/Scripts/python.exe` (Windows) with `PYTHONUTF8=1`.
4. Re-run the suite; expect 271/275 with the four platform failures above.

## 15. Pending approvals

- S1 manual Git checkpoint, then S2 start (owner asked to stop at the S1
  checkpoint).
- Not approved: real-model or paid collection, Fireworks calls, model or
  dataset downloads, final training for an unconfirmed pair, final held-out
  evaluation, artifact deployment, broad chat-routing changes, Git writes.

## 16. Blockers

- **Pending — teammate evidence:** exact local model identity (tag, digest,
  quantisation, backend), active context length, and generation settings.
  The checkout contains no Qwen3.5-27B configuration; the only 27B
  reference is the Ollama tag `qwen3.8:27b`, verified at a 2,048-token
  context only. The upgrade is presumed to be machine-local configuration.
- **Pending — teammate evidence:** large-context capability. No evidence in
  the repository supports contexts beyond 2,048 tokens for the 27B model.
- **Pending:** the strong model actually configured on the inference machine.
- No outcomes exist for any current model pair.

Known defects found in S0, relevant to later stages (not yet fixed):

- Data-pipeline `open()` calls without `encoding=` (`routing/dataset.py`,
  `scripts/collect_outcomes.py`, `scripts/build_real_pilot.py`,
  `routing/train.py`, `evaluation/run.py`). Without `PYTHONUTF8=1`,
  `load_dataset` fails on Windows with `UnicodeDecodeError` for the real
  600-row dataset. Proposed for S2.
- Grader false positives/negatives from the Phase 1 audit: fixed in S1.

Limitations carried into S2 (not fixed in S1):

- Truncation is detected only from `finish_reason`, which the clients do
  not yet populate (`getattr(..., "finish_reason", None)` is None today).
  S2 adds it to `Completion`.
- "Raw" answers are what the clients return: `ollama_client.generate` and
  `remote_client` already `.strip()` whitespace, and the remote client
  appends tool-call text. S2 should decide whether to keep the
  pre-normalisation text.
- `finalize()` writes only graded records into the dataset; grade flags
  and review counts live in the `.work.jsonl` rows. S2 reporting must count
  every attempted task and state denominators.
- `exact`/`contains` keep substring/equality semantics (no task file uses
  them today).
- The letter grader treats options A–H; "I" is excluded as a pronoun.

## 17. Exact next action

Owner performs the S1 manual Git checkpoint (and the still-pending S0
documentation commit), then approves S2.

## 18. Manual Git checkpoint due?

Yes — S1 "Pipeline and grading safeguards". Note: at 2026-10-26 11:00 HEAD
was still `8b44cf6` and `origin/feature/router-offline-prep` did not exist;
the S0 documents were staged but not committed.
