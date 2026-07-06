# Task backlog — GitHub-issue-ready

Copy each open block below into a GitHub issue (title = heading, body = the
text under it), or batch-create them with the `gh` commands at the bottom.

Tags: **[NOW]** = doable today · **[KICKOFF]** = needs launch-day info.
Kickoff happened 2026-07-07 (real spec — see HANDOFF.md §0); competition ends
11 July 2026.

---

## 🚀 KICKOFF — real spec landed: DO THESE FIRST (contract compliance)

Without these the submission scores ZERO regardless of answer quality.

### [P0] #9 I/O adapter — `/input/tasks.json` → `/output/results.json`
Harness mode: read `/input/tasks.json` (`[{task_id, prompt}]`), write
`/output/results.json` (`[{task_id, answer}]`, valid JSON). New entrypoint
that runs this instead of the `--tasks`/stdout dev path. Always write a valid
file even on partial failure; exit 0 on success, non-zero on failure.
**Files:** `main.py`, `Dockerfile` (ENTRYPOINT/CMD).

### [P0] #10 Honor `ALLOWED_MODELS` at runtime
Read `ALLOWED_MODELS` (comma-separated) + `FIREWORKS_BASE_URL` from env; the
remote client must pick its model from that list (not the hardcoded
`deepseek-v4-pro`). Fail loudly if none is usable. Verify EVERY remote call
goes through `FIREWORKS_BASE_URL` (bypass = 0 tokens recorded).
**Files:** `config.py`, `remote_client.py`.

### [P0] #11 Concurrency — fit the 10-minute cap
Remote calls run ~27 s each; a sequential loop over N tasks blows 10 min fast.
Parallelise remote calls (asyncio or a thread pool), with a global deadline
guard so we always write results before the cap. **Files:** `main.py`.

### [P1] #12 Small/quantized local model + public image ≤ 10 GB
Pick a local model that fits < 10 GB compressed and loads fast (Q4 GGUF ~1 GB,
or a 0.5–1.5B checkpoint). Build, then push the image to GHCR (public).
**Files:** `Dockerfile`, `config.py`, `requirements.txt`.

### [P1] #13 Conservative accuracy-gate tuning + concise remote output
Accuracy is a pass/fail GATE (below → excluded). Sanity-check the local model
per category, raise the escalation bar until it clears comfortably, THEN
minimise tokens (concise-answer prompting + tight per-task `max_tokens`).
**Files:** `router.py`, `config.py`, `confidence.py`.

---

## ✅ Done (closed 2026-07-04/05 — details in git history + HANDOFF.md §3)

| # | Was | Outcome |
| --- | --- | --- |
| 1 | Prove real local model path | Qwen2.5-1.5B generates real answers, exact tokenizer counts, chat-template + plain branches both work |
| 2 | Prove real Fireworks remote path | Live calls + real `usage` billing; found `llama-v3p3-70b` retired → default now `deepseek-v4-pro`; REMOTE_MAX_TOKENS 4096; clean bad-key error path |
| 3 | Docker build + smoke test | BOTH images build + run mock correctly in-container: CPU 1.66 GB (`torch 2.12.1+cpu`) and ROCm 4.94 GB (`torch 2.9.1+rocm6.4`); logs persist via mount; real API call works in-container. Only the optional model-bake line untested — needs the final model |
| 5 | Threshold calibration tool | `scripts/calibrate.py` — audited line-by-line 2026-07-05, replay math correct |
| — | Draft-and-judge confidence gate | Local model's own mean token logprob (`local_confidence` in the log); below `LOGPROB_CONFIDENCE_THRESHOLD` (default 0.4) → escalate. Live-tested both directions. Limit: catches *uncertainty*, not *confident error* — calibrate at kickoff |

---

## [P1][NOW→KICKOFF] #4 ROCm live test on real AMD hardware — the last unproven path
**Why:** ROCm torch is already the Docker default (`TORCH_INDEX` build arg,
rocm6.4; `make build-cpu` for the CPU image; `make docker-run-gpu` passes
`/dev/kfd` + `/dev/dri`), and the ROCm image now BUILDS clean and runs mock
in-container (4.94 GB, `torch 2.9.1+rocm6.4`). But no real generation has
ever run on an AMD GPU — the dev Macs (arm64) cannot execute ROCm wheels.
This is the only remaining unproven path in the repo.

**Tasks**
- [ ] On AMD Developer Cloud (access at kickoff): rebuild `make build`
      natively on x86 first (the arm64 Mac's emulated build took ~5.5 h —
      do NOT rebuild there) + `make docker-run-gpu`
- [ ] Confirm `_pick_device()` returns `cuda` and a real local generation runs on the GPU
- [ ] Time it vs CPU; decide the submission image (ROCm default vs CPU fallback)

**Done when:** a real local generation completes on an AMD GPU via ROCm.
**Files:** `Dockerfile`, `Makefile`, `local_model.py` (verify only).

---

## [P2][KICKOFF] #6 Decide single-shot router vs multi-step agent
**Why (alignment flag):** current flow is prompt → ONE model call → answer.
The track says "complete tasks autonomously." If the revealed tasks need
decomposition or tool use, we need a loop around `run_task`.

**Tasks**
- [ ] Inspect the revealed task shape at kickoff
- [ ] Decide: is single-shot sufficient?
- [ ] If not, add a task loop / tool-calling layer around `run_task` (route
      each step independently — some steps local, some remote)

**Done when:** the decision is documented, and a loop is added if the tasks
demand it. **Files:** `main.py`, `router.py`. Blocked on the reveal.

---

## [P2][KICKOFF] #7 Task-specific output validator in post_check
**Why:** `router.post_check` currently catches only generic failure modes
(empty/hedge/echo/repetition), and the logprob gate catches uncertainty but
NOT confident error (a wrong bat-and-ball answer scored 0.90). If outputs are
verifiable, a real validator is the single biggest accuracy upgrade — it
turns the escalation cascade from a guess into a check.

**Tasks**
- [ ] Determine the revealed output format (exact-match / JSON schema / unit
      tests / numeric)
- [ ] Add a validator to `post_check` that escalates local answers failing it
- [ ] Extend `_SIGNAL_PATTERNS` in `confidence.py` with task-set keywords;
      fill `FORCE_ROUTE_BY_CATEGORY` if tasks come labeled

**Done when:** wrong-but-fluent local answers get caught and escalated. **Files:**
`router.py`, `confidence.py`. Blocked on the reveal.

---

## [P2][KICKOFF] #8 Calibrate BOTH thresholds from one graded sweep
**Why:** `usage.jsonl` now logs two dials per task — the router's heuristic
`confidence` (pre-route) and the model's `local_confidence` (logprob gate).
`scripts/calibrate.py` currently tunes only the first. One graded sweep can
tune both.

**Tasks**
- [ ] Extend `scripts/calibrate.py`: given `--accuracy grades.json`, plot/rank
      graded-wrong vs graded-right tasks by `local_confidence` and recommend
      the `LOGPROB_CONFIDENCE_THRESHOLD` split point (~30 lines)
- [ ] If confident-wrong answers are common in the real task set, switch the
      confidence statistic from mean to min-token-prob (2-line change in
      `local_model.py`, noted in its comments)

**Done when:** one command recommends both `CONFIDENCE_THRESHOLD` and
`LOGPROB_CONFIDENCE_THRESHOLD` from the same graded run. **Files:**
`scripts/calibrate.py`, maybe `local_model.py`. Blocked on graded data.

---

## Out of repo: pitch website (Aryan)

The "RouteFlow AI" marketing/pitch site is being built separately and is NOT
part of the scored submission (score = tokens + accuracy of the containerized
agent). It exists for the judges' demo/pitch. Keep it in its own repo — do
not add frontend tooling to this one.

---

## Optional: batch-create the open issues

```bash
gh issue create --title "P1 #4 ROCm live test on real AMD hardware" \
  --body "See ISSUES.md #4. make build + make docker-run-gpu on AMD Developer Cloud; confirm cuda device + real generation. Last unproven path."

gh issue create --title "P2 KICKOFF #6 Decide single-shot router vs multi-step agent" \
  --body "See ISSUES.md #6. Inspect revealed task shape; add a task loop around run_task if multi-step work is needed. Files: main.py, router.py."

gh issue create --title "P2 KICKOFF #7 Task-specific output validator in post_check" \
  --body "See ISSUES.md #7. Add a real output validator that escalates failing local answers; extend confidence signals. Files: router.py, confidence.py."

gh issue create --title "P2 KICKOFF #8 Calibrate both thresholds from one graded sweep" \
  --body "See ISSUES.md #8. Extend scripts/calibrate.py to also recommend LOGPROB_CONFIDENCE_THRESHOLD from local_confidence vs grades."
```
