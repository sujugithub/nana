# Task backlog — GitHub-issue-ready

Copy each block below into a GitHub issue (title = heading, body = the text
under it), or batch-create them with the `gh` commands at the bottom.

Tags: **[NOW]** = doable today with the current scaffold · **[KICKOFF]** =
blocked until the real tasks/models are revealed (11 July 2026).

Priority: **P0** = must work before we can trust anything · **P1** =
competitiveness · **P2** = depends on the reveal.

---

## [P0][NOW] #1 Prove the real local model path end-to-end
**Why:** the scaffold has only ever run in mock mode (hardcoded strings). The
`transformers` load → chat-template → `generate` → token-count path in
`local_model.py` has never actually executed. This is the free-token engine —
if it doesn't work, nothing works.

**Tasks**
- [x] `pip install -r requirements.txt` (venv `.venv/`, torch 2.8.0 +
      transformers 4.57.6 on Python 3.9)
- [x] Run a real (non-mock) generation with the placeholder model:
      `python3 main.py --tasks tasks/sample_tasks.json` (no `--mock`)
- [x] Confirm a coherent answer comes back for `trivial-1` ("The capital of
      France is Paris.")
- [x] Confirm `prompt_tokens` / `completion_tokens` are non-zero and match the
      tokenizer (not word counts) — trivial-1 logged 36+8 (chat template
      included), not the 6-word mock count
- [x] Sanity-check the chat-template branch vs the plain branch (Qwen used the
      template; `gpt2` exercised the plain branch — both generate, no crash)
- [x] Note cold-load time + peak RAM: Qwen2.5-1.5B on the M-series dev Mac —
      mps: 6.8s load / 0.9s short generate / ~3.7 GB peak; forced-cpu (fp32
      guard active, no Half crash): 9.5s load / 2.9s / ~6.4 GB peak. First
      ever run also downloads ~2.9 GB of weights.

**DONE 2026-07-04** — real local answers return, token counts are correct, no
dtype/device/token_type_ids crash. **Files:** `local_model.py`,
`requirements.txt`.

---

## [P0][NOW] #2 Prove the real Fireworks remote path end-to-end
**Why:** the billable side. Payload, auth header, retry policy, and `usage`
parsing in `remote_client.py` have never hit the live API.

**Tasks**
- [x] `export FIREWORKS_API_KEY=fw-...` (a working key) — key in local `.env`
      (gitignored; share within the team via private message only)
- [x] Force everything remote for a test (`--threshold 1.01`: conf can equal
      1.0, and decide() routes local on `>=`, so 1.0 doesn't force remote)
- [x] Confirm a real answer returns and `billable_tokens` reflects the API's
      real `usage` — yes, no ESTIMATED warning
- [x] Verify the model name in `config.py` actually resolves on Fireworks —
      **it did NOT: `llama-v3p3-70b-instruct` is retired from serverless
      (live 404, 2026-07-04).** Ran a bake-off across the 6 available chat
      models; default is now `deepseek-v4-pro` (flagship + fewest completion
      tokens). NOTE: all current serverless chat models bill hidden
      reasoning tokens into completion usage — REMOTE_MAX_TOKENS raised to
      4096 because 1024 truncated the hard sample task mid-thought.
- [x] Test failure handling: a bad key raises a clean `RemoteError` (live
      test 2026-07-04: fake key → Fireworks answers 404 "Model not found,
      inaccessible" — it masks models from bad keys — one clean problems=[]
      line, no traceback, run continued, local fallback answered correctly)

**DONE 2026-07-04** — full default-threshold run: trivial+moderate local
(0 billable), complex remote via deepseek-v4-pro (real usage tokens logged).
**Files:** `remote_client.py`, `config.py`.

---

## [P0][NOW] #3 Build and smoke-test the Docker image
**Why:** submission must be containerized, and `docker build` has never run —
only been reasoned about.

**Tasks** (Docker Desktop installed on the dev Mac 2026-07-04)
- [x] Build: `make build-cpu` builds clean (1.66 GB); `make build` (ROCm
      default from #4) builds clean too — 4.94 GB, `torch 2.9.1+rocm6.4`
      inside, mock runs correctly in-container. Warning: building the ROCm
      image on the arm64 dev Mac took ~5.5 h (emulated layer export) —
      rebuild it on x86 hardware (AMD Dev Cloud) at kickoff instead.
- [x] Run mock in-container (CPU image, `-e AGENT_MOCK=1`): routing correct,
      same output as host mock run. amd64 image runs on the arm64 Mac via
      emulation (platform warning is expected and harmless).
- [x] Confirm `logs/usage.jsonl` is written to the host via the volume mount
      — verified, host file grew by 3 lines
- [x] Record the final image size; check the CPU-torch trick kept CUDA libs
      out — CPU image is 1.66 GB, torch reports `2.12.1+cpu`
- [x] (Bonus) real Fireworks call from inside the container via
      `--env-file .env` — answered correctly with real usage tokens, so the
      key-passing path the submission relies on is proven
- [ ] (Optional) test the model-bake `RUN` line once the real model is chosen

**Done when:** image builds, container runs, logs persist to host. **Files:**
`Dockerfile`, `Makefile`.

---

## [P1][NOW] #4 Make ROCm / AMD-GPU the default path
**Why (alignment flag):** the competition is explicitly on AMD GPUs + ROCm, but
`Dockerfile` defaults to CPU torch and treats ROCm as an optional comment. If
the scoring box has an AMD GPU we're leaving it on the table, and the ROCm
path is currently the least-tested.

**Tasks**
- [x] Add a ROCm build variant (build arg or second Dockerfile) using the
      ROCm torch wheel index — done 2026-07-04: `TORCH_INDEX` build arg,
      **ROCm (rocm6.4) is now the DEFAULT**; `make build-cpu` overrides to
      the CPU wheel. Verified rocm6.4 serves torch 2.8/2.9 cp311 x86_64
      wheels. Added `make docker-run-gpu` (--device=/dev/kfd --device=/dev/dri).
- [ ] Verify `LocalModel._pick_device()` selects `cuda` on a ROCm torch build
      — verified in code + by design (torch.cuda.is_available() is True on
      ROCm builds), but needs a live check on AMD hardware
- [ ] Test a real local generation on AMD Developer Cloud — **needs AMD
      hardware; the dev Mac (arm64) cannot run ROCm wheels at all**
- [x] Decide CPU-default vs ROCm-default: ROCm-default (a ROCm torch build
      falls back to CPU cleanly when no GPU is present, so it costs image
      size, never correctness)

**Done when:** the local model runs on an AMD GPU via ROCm and we know which
build to submit. **Files:** `Dockerfile`, `local_model.py`. Remaining live
test depends on AMD Developer Cloud access.

---

## [P1][NOW] #5 Threshold-calibration analysis script
**Why:** the README promises log-driven calibration; each `usage.jsonl` line
already carries `run_id`, `threshold`, `confidence`. We need the tool that
turns a sweep into a decision.

**Tasks**
- [x] Add `scripts/calibrate.py` that reads `logs/usage.jsonl`, groups by
      `run_id`, and tabulates billable tokens vs threshold
- [x] Given an accuracy signal per task (`--accuracy grades.json`, a
      `{task_id: true/false or 0..1}` file, + `--min-accuracy`), recommend
      the LOWEST threshold that clears the bar. Also replays the exact token
      savings of LOWERING each threshold (already-logged remote costs).
- [x] Document usage in the README calibration section

**DONE 2026-07-04** — tested against a mock `--threshold 0.4/0.55/0.7` sweep:
table, recommendation, and lowering-replay all print. Grading answers stays a
manual step (task-set-specific). Real numbers come at kickoff.

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
(empty/hedge/echo/repetition). If outputs are verifiable, a real validator is
the single biggest accuracy upgrade — it turns the escalation cascade from a
guess into a check.

**Tasks**
- [ ] Determine the revealed output format (exact-match / JSON schema / unit
      tests / numeric)
- [ ] Add a validator to `post_check` that escalates local answers failing it
- [ ] Extend `_SIGNAL_PATTERNS` in `confidence.py` with task-set keywords;
      fill `FORCE_ROUTE_BY_CATEGORY` if tasks come labeled

**Done when:** wrong-but-fluent local answers get caught and escalated. **Files:**
`router.py`, `confidence.py`. Blocked on the reveal.

---

## Optional: batch-create these as GitHub issues

After `gh auth login` and creating the repo, from the repo root:

```bash
gh issue create --title "P0 NOW #1 Prove the real local model path end-to-end" \
  --body "See ISSUES.md #1. Run the transformers load/generate/token-count path for real; confirm coherent answer + correct token counts + no dtype/device crash. Files: local_model.py, requirements.txt."

gh issue create --title "P0 NOW #2 Prove the real Fireworks remote path end-to-end" \
  --body "See ISSUES.md #2. Real API call via FIREWORKS_API_KEY; confirm real usage tokens + clean error handling. Files: remote_client.py, config.py."

gh issue create --title "P0 NOW #3 Build and smoke-test the Docker image" \
  --body "See ISSUES.md #3. make build (linux/amd64), run in-container, confirm logs persist via mount, record image size. Files: Dockerfile, Makefile."

gh issue create --title "P1 NOW #4 Make ROCm / AMD-GPU the default path" \
  --body "See ISSUES.md #4. Add ROCm torch build; verify device pick; test on AMD Developer Cloud. Files: Dockerfile, local_model.py."

gh issue create --title "P1 NOW #5 Threshold-calibration analysis script" \
  --body "See ISSUES.md #5. scripts/calibrate.py: group usage.jsonl by run_id, tabulate tokens vs threshold, recommend lowest passing threshold."

gh issue create --title "P2 KICKOFF #6 Decide single-shot router vs multi-step agent" \
  --body "See ISSUES.md #6. Inspect revealed task shape; add a task loop around run_task if multi-step work is needed. Files: main.py, router.py."

gh issue create --title "P2 KICKOFF #7 Task-specific output validator in post_check" \
  --body "See ISSUES.md #7. Add a real output validator that escalates failing local answers; extend confidence signals. Files: router.py, confidence.py."
```

(Custom labels aren't created here — add labels like `NOW` / `KICKOFF` in the
GitHub UI or with `gh label create` first if you want them.)
