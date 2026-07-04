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
- [ ] `pip install -r requirements.txt`
- [ ] Run a real (non-mock) generation with the placeholder model:
      `python3 main.py --tasks tasks/sample_tasks.json` (no `--mock`)
- [ ] Confirm a coherent answer comes back for `trivial-1`
- [ ] Confirm `prompt_tokens` / `completion_tokens` are non-zero and match the
      tokenizer (not word counts)
- [ ] Sanity-check the chat-template branch vs the plain branch
- [ ] Note cold-load time + peak RAM on a CPU box

**Done when:** a real local answer returns, token counts are correct, no
dtype/device/token_type_ids crash. **Files:** `local_model.py`,
`requirements.txt`.

---

## [P0][NOW] #2 Prove the real Fireworks remote path end-to-end
**Why:** the billable side. Payload, auth header, retry policy, and `usage`
parsing in `remote_client.py` have never hit the live API.

**Tasks**
- [ ] `export FIREWORKS_API_KEY=fw-...` (a working key)
- [ ] Force everything remote for a test: `python3 main.py --tasks
      tasks/sample_tasks.json --threshold 1.0`
- [ ] Confirm a real answer returns and `billable_tokens` reflects the API's
      real `usage` (not the 0-estimate fallback / no warning printed)
- [ ] Verify the model name in `config.py` actually resolves on Fireworks
- [ ] Test failure handling: a bad key should raise a clean `RemoteError`, not
      a traceback; confirm the run continues (local fallback) on a forced
      remote failure

**Done when:** a real remote call returns text + real token usage, and errors
are clean. **Files:** `remote_client.py`, `config.py`.

---

## [P0][NOW] #3 Build and smoke-test the Docker image
**Why:** submission must be containerized, and `docker build` has never run —
only been reasoned about.

**Tasks**
- [ ] `make build` (pinned to `linux/amd64`)
- [ ] Run mock in-container: `make docker-run` (with `AGENT_MOCK=1` in `.env`)
- [ ] Confirm `logs/usage.jsonl` is written to the host via the volume mount
- [ ] Record the final image size; check the CPU-torch trick actually kept
      CUDA libs out
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
- [ ] Add a ROCm build variant (build arg or second Dockerfile) using the
      ROCm torch wheel index
- [ ] Verify `LocalModel._pick_device()` selects `cuda` on a ROCm torch build
- [ ] Test a real local generation on AMD Developer Cloud
- [ ] Decide CPU-default vs ROCm-default based on the confirmed scoring env

**Done when:** the local model runs on an AMD GPU via ROCm and we know which
build to submit. **Files:** `Dockerfile`, `local_model.py`. Depends on knowing
the scoring hardware.

---

## [P1][NOW] #5 Threshold-calibration analysis script
**Why:** the README promises log-driven calibration; each `usage.jsonl` line
already carries `run_id`, `threshold`, `confidence`. We need the tool that
turns a sweep into a decision.

**Tasks**
- [ ] Add `scripts/calibrate.py` that reads `logs/usage.jsonl`, groups by
      `run_id`, and tabulates billable tokens vs threshold
- [ ] Given an accuracy signal per task, recommend the LOWEST threshold that
      clears the bar
- [ ] Document usage in the README calibration section

**Done when:** running a `--threshold 0.4/0.55/0.7` sweep + the script prints a
tokens-vs-accuracy table. **Files:** new `scripts/calibrate.py`. Buildable now
against mock/sample logs; real numbers come at kickoff.

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
