# Task backlog — GitHub-issue-ready

Copy each open block below into a GitHub issue (title = heading, body = the
text under it), or batch-create them with the `gh` commands at the bottom.

Tags: **[NOW]** = doable today · **[KICKOFF]** = needs launch-day info.
Kickoff happened 2026-07-07 (real spec — see HANDOFF.md §0); competition ends
11 July 2026.

---

## 🚀 KICKOFF — real spec landed: DO THESE FIRST (contract compliance)

Without these the submission scores ZERO regardless of answer quality.

### [P1] #12 Small/quantized local model + public image ≤ 10 GB
**Mostly done 2026-07-07** — remaining steps need x86 hardware + GHCR auth:
- [x] Model picked: **Qwen2.5-1.5B-Instruct** (proven checkpoint; accuracy
      gate is existential, so 1.5B > 0.5B; size budget comfortable)
- [x] Bake implemented: `BAKE_MODEL` build arg (default ON), snapshot into
      the image's HF cache, `LOCAL_MODEL_NAME` pinned inside the image
- [x] CPU image built + VALIDATED: **2.78 GB compressed**; harness-mode mock
      run passes with `--network none` (zero-download local path proven)
- [x] Makefile: `ghcr-login` / `push` / `push-cpu` / `image-size` targets
      (registry: `ghcr.io/sujugithub/hybrid-token-routing-agent`)
- [x] CPU image PUSHED + PUBLIC + verified 2026-07-07:
      `ghcr.io/sujugithub/hybrid-token-routing-agent:cpu` (digest e0d02346…),
      anonymous `docker pull --platform linux/amd64` succeeds. NOTE: the
      package visibility flip is under PACKAGE settings, not repo settings —
      three failed attempts hit the repo page first (repo is now public too)
- [x] ROCm image BUILT 2026-07-07 (emulated on the Mac, minutes with layer
      cache): `hybrid-router-agent:latest`, 7.37 GB uncompressed, torch
      2.9.1+rocm6.4, model baked, `LOCAL_MODEL_NAME` pinned
- [ ] ROCm `:latest` GHCR push was IN FLIGHT at session end — verify:
      `docker manifest inspect ghcr.io/sujugithub/hybrid-token-routing-agent:latest`
      (logged out); if absent, `make ghcr-login && make push` from the Mac
      (image exists locally) or rebuild+push on the AMD droplet (faster)
**Files:** `Dockerfile`, `Makefile`.

### [P1] #13 Conservative accuracy-gate tuning + concise remote output
Accuracy is a pass/fail GATE (below → excluded). Sanity-check the local model
per category, raise the escalation bar until it clears comfortably, THEN
minimise tokens.
- [x] Concise-answer `SYSTEM_PROMPT` (both backends) **done 2026-07-07** —
      real A/B: remote billable −58% on the heaviest tasks, and it FIXED the
      one wrong local answer ("mixed" → "Negative"). See HANDOFF §3.
- [ ] Raise `LOGPROB_CONFIDENCE_THRESHOLD` per calibration on revealed
      samples (real-data recommendation so far: 0.697, n=6, pre-dates the
      concise prompt — redo the sweep)
- [ ] Optional: tight per-task `max_tokens` if the revealed set allows it
**Files:** `router.py`, `config.py`, `confidence.py`.

---

## ✅ Done (closed 2026-07-04..07 — details in git history + HANDOFF.md §3)

| # | Was | Outcome |
| --- | --- | --- |
| 9 | I/O adapter (`/input/tasks.json` → `/output/results.json`) | `main.py --input/--output` harness mode, now the Docker CMD (dev `--tasks` path unchanged). results.json ALWAYS written (atomic tmp+rename, every task_id, `""` for unfinished), input order preserved. Exit 0 whenever a valid all-task file landed (partial answers beat voiding the run); exit 1 only on unreadable input / unwritable output. `make docker-run-harness` simulates the mounts. Covered by `test_harness.py` |
| 8 | Calibrate BOTH thresholds from one graded sweep | `scripts/calibrate.py --accuracy grades.json` now also ranks graded LOCAL answers by `local_confidence` and recommends the lowest `LOGPROB_CONFIDENCE_THRESHOLD` whose kept-local answers clear `--min-accuracy` (reports paid escalations + wasted retries; 0 = gate off). When no gate separates right from wrong it prints the mean→min-token-prob switch cue. Unit-checked in `test_harness.py`; verified on a synthetic graded sweep. The mean→min switch itself stays data-dependent (kickoff) |
| — | Category keywords for the 8 kickoff categories | `_SIGNAL_PATTERNS` in confidence.py: hard-remote penalties now DECISIVE at 0.75 (math incl. word problems, code + new code_debug stack, new logic patterns) — one hard signal sends even a short prompt remote (accuracy gate > token savings); easy-local BOOSTS (negative weights: sentiment −0.40, ner −0.40, summarize −0.30) key on instruction words so long-but-easy prompts beat the length ramp. All 8 categories route-asserted in `test_harness.py`; tightest margin code-gen 0.535 vs 0.55 |
| 10 | Honor `ALLOWED_MODELS` at runtime | `resolve_remote_model()` in remote_client.py: allow-list entries used VERBATIM (proxy bills by those IDs), matched on the last path segment so short/full spellings agree. Priority: explicit `REMOTE_MODEL_NAME` if allowed → `REMOTE_MODEL_PREFERENCE` (default deepseek-v4-pro) → list head; unset list = dev fallback; set-but-empty list disables remote per-call (local fallback keeps the run alive). `FIREWORKS_BASE_URL` was already the only HTTP call site. Covered by `test_harness.py` |
| 11 | Concurrency for the 10-min cap | `run_all` thread pool (`REMOTE_CONCURRENCY`, default 8) around the untouched `run_task`; local generation serializes on a `LocalModel` lock, `TokenTracker` locked. Global deadline `RUN_DEADLINE_S` (540 s from process start, model load included): at the deadline finished-but-uncollected answers are harvested, the rest abandoned, results written, and stuck worker threads bypassed via `os._exit` so the write always lands inside the cap. Proven: 8×2 s remote stubs in 2.0 s wall |
| 1 | Prove real local model path | Qwen2.5-1.5B generates real answers, exact tokenizer counts, chat-template + plain branches both work |
| 2 | Prove real Fireworks remote path | Live calls + real `usage` billing; found `llama-v3p3-70b` retired → default now `deepseek-v4-pro`; REMOTE_MAX_TOKENS 4096; clean bad-key error path |
| 3 | Docker build + smoke test | BOTH images build + run mock correctly in-container: CPU 1.66 GB (`torch 2.12.1+cpu`) and ROCm 4.94 GB (`torch 2.9.1+rocm6.4`); logs persist via mount; real API call works in-container. Only the optional model-bake line untested — needs the final model |
| 5 | Threshold calibration tool | `scripts/calibrate.py` — audited line-by-line 2026-07-05, replay math correct |
| — | Draft-and-judge confidence gate | Local model's own mean token logprob (`local_confidence` in the log); below `LOGPROB_CONFIDENCE_THRESHOLD` (default 0.4) → escalate. Live-tested both directions. Limit: catches *uncertainty*, not *confident error* — calibrate at kickoff |

---

## [P1][NOW] #4 ROCm live test on real AMD hardware — the last unproven path
**Why:** the ROCm image (model baked) now EXISTS — `hybrid-router-agent:latest`,
torch 2.9.1+rocm6.4 — but no real generation has ever run on an AMD GPU.
CPU fallback is proven, so the risk is slow-local (10-min-cap pressure),
not broken.

**Status 2026-07-07:** AMD Dev Cloud access GRANTED; an MI300X x1 droplet
(image "ROCm Software 7.2.4" Quick Start, SSH key = dev Mac's
`~/.ssh/id_ed25519_github`) was being created when the session ended. July
GPU capacity is reduced (AMD event) — don't destroy a working droplet
until completely done.

**Tasks (on the droplet)**
- [ ] `git clone https://github.com/sujugithub/hybrid-token-routing-agent`
      (public), create `.env` with the Fireworks key
- [ ] `make build` (native x86 — minutes)
- [ ] GPU visible through Docker:
      `docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video
      --entrypoint python hybrid-router-agent -c "import torch;
      print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
      → expect `True AMD Instinct MI300X`. Host ROCm is 7.2.4 vs our 6.4
      wheels — normally fine; if False, rebuild with
      `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm7.0`
- [ ] `make docker-run-gpu` → real answers with `local_conf` values (not
      `[mock-local]`); note per-task latency vs CPU (~10–20 s/task)
- [ ] `docker login ghcr.io -u sujugithub` (token from the Mac:
      `gh auth token`) + `make push`

**Done when:** a real local generation completes on an AMD GPU via ROCm and
the pushed `:latest` is anonymous-pullable.
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
- [x] ~~Extend `_SIGNAL_PATTERNS` in `confidence.py` with task-set keywords~~
      **done 2026-07-07** — all 8 kickoff categories covered (see ✅ table);
      `FORCE_ROUTE_BY_CATEGORY` stays empty (input has no category labels)

**Done when:** wrong-but-fluent local answers get caught and escalated. **Files:**
`router.py`, `confidence.py`. Blocked on the reveal.

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
