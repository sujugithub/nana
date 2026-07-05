# HANDOFF — Hybrid Token-Efficient Routing Agent

> Paste this file (or point the session at it) when continuing work in a new
> Claude session. It contains everything needed to take over with zero prior
> context. **The SCAFFOLD is complete — don't re-scaffold or restructure it.
> UPDATE 2026-07-04: the real paths are now PROVEN — real local generation,
> real Fireworks calls, and the Docker container have all run successfully
> (see §3). Remaining work: AMD-GPU/ROCm live test + everything blocked on
> the kickoff reveal (~7 July).**

## 1. Mission and scoring rules

- **Event:** AMD Developer Competition ACT II, Track 1 — Hybrid Token-Efficient
  Routing Agent.
- The agent completes tasks autonomously, deciding per task whether to run a
  **local** model or call a **remote** model via the Fireworks AI API.
- **Score = token count + output accuracy.** Local model usage counts as
  **ZERO tokens**. Accuracy below a threshold is penalized, so "always local"
  loses.
- Final scoring runs in a standardized **limited-compute** environment
  (assume CPU or AMD GPU with ROCm — never NVIDIA). Local model target:
  1–3B params, quantized.
- The real tasks and allowed models are **revealed at kickoff** — everything
  is placeholder-driven and swappable by design.
- Submission must be **containerized (Docker)**.

## 2. Project location and layout

Project root: the repo checkout (flat Python modules, no package). On the
current dev Mac: `~/Desktop/hack/demo`, with deps installed in `.venv/`
(Python 3.9 — use `.venv/bin/python` for real runs; bare `python3` suffices
for mock/tests).

| File | Role |
| --- | --- |
| `main.py` | Orchestrator + CLI. `run_task()` = decide → execute → post-check → account. `load_tasks()` is the ONLY place to touch if the task format differs. |
| `router.py` | Decision layer: pre-route on confidence, `post_check()` on local outputs, escalation policy. `FORCE_ROUTE_BY_CATEGORY` dict for kickoff-day hard routing. |
| `confidence.py` | Zero-cost heuristics scoring "can the local model handle this?" (length ramp + math/code/reasoning/multi-part regex penalties, weighted). Tune `_SIGNAL_PATTERNS` and `DEFAULT_WEIGHTS`. |
| `local_model.py` | HF transformers wrapper. Lazy imports, chat template, exact token counts, fp32-on-CPU guard, ROCm-aware device pick. |
| `remote_client.py` | Fireworks `/chat/completions` via plain `requests`. Billing-aware retries (read timeouts NOT retried), Retry-After support, usage-field fallback estimate. |
| `token_tracker.py` | Accounting. One JSONL line per task in `logs/usage.jsonl` with confidence/threshold/signals/run_id — the calibration audit trail. |
| `config.py` | Every knob, env-overridable. The one file to touch on kickoff day. |
| `schemas.py` | Shared `Task` / `Completion` dataclasses. |
| `test_harness.py` | Offline end-to-end wiring test. Mock mode, stdlib only, pins its own env. |
| `scripts/calibrate.py` | Threshold-calibration analysis: groups `logs/usage.jsonl` by run, tabulates billable tokens vs threshold, recommends the lowest threshold clearing an accuracy bar. Stdlib only. |
| `tasks/sample_tasks.json` | 3 dummy tasks (trivial→local, moderate, complex→remote). |
| `Dockerfile` / `Makefile` / `.env.example` / `README.md` | Packaging + docs. |

Architecture in one line: `Router.decide` (heuristics) → local if confident
(free) else remote (billable); local outputs pass `Router.post_check` and
escalate to remote if they look wrong; everything is logged by `TokenTracker`.

Key design bet: local is free, so route **optimistically local** and let the
post-check cascade bound the accuracy risk. `CONFIDENCE_THRESHOLD` (default
0.55) is the single dial trading tokens vs accuracy.

Failure policy (already implemented — preserve it): an answer always beats no
answer. Escalation failure → keep flagged local answer; remote-route failure
→ local fallback; any other per-task error → error row, run continues.

## 3. Current state — READ THIS CAREFULLY

### VERIFIED (safe to rely on) — updated 2026-07-04
- All modules compile; `python3 test_harness.py` passes offline, zero deps.
- **Real local model runs** (issue #1 done): Qwen2.5-1.5B generates coherent
  answers with correct tokenizer counts on the dev Mac — mps: 6.8s load /
  ~3.7 GB peak; forced-cpu (fp32 guard, no Half crash): 9.5s / ~6.4 GB.
  Both the chat-template branch (Qwen) and the plain branch (gpt2) work.
- **Real Fireworks calls work** (issue #2 done): real answers + real `usage`
  token counts, in a full default-threshold run (easy→local free,
  hard→remote billed). Error paths proven live: bad key → one clean
  RemoteError line + local fallback, run continues.
  - **The original placeholder remote model is RETIRED**: Fireworks 404s on
    `llama-v3p3-70b-instruct`. Default is now `deepseek-v4-pro` (won a
    bake-off of the 6 available chat models: flagship quality + fewest
    completion tokens). Note: **every current serverless chat model bills
    hidden reasoning tokens into `completion_tokens`** — that's why
    `REMOTE_MAX_TOKENS` is now 4096 (1024 truncated hard answers
    mid-thought: billed but useless).
- **Docker proven** (issue #3 done): CPU-variant image builds (1.66 GB,
  `torch 2.12.1+cpu`, no CUDA libs), mock runs in-container with correct
  routing, `logs/usage.jsonl` persists to the host through the volume
  mount, and a real Fireworks call from inside the container via
  `--env-file .env` works.
- **Calibration tooling exists** (issue #5 done): `scripts/calibrate.py`.
- **Draft-and-judge confidence gate implemented + live-tested 2026-07-05**:
  the local model now reports its own mean token probability
  (`Completion.confidence`, from `compute_transition_scores`), logged as
  `local_confidence` in usage.jsonl; below `LOGPROB_CONFIDENCE_THRESHOLD`
  (default 0.4, 0 disables) the task escalates to remote with a
  `low_confidence:<val>` problem tag. Verified live with
  Qwen2.5-0.5B-Instruct: gate 0.99 → escalated to remote (57 tokens billed,
  answer correct); gate 0.4 → stayed local (0 billed). Mock mode
  unaffected (confidence=None). KNOWN LIMIT: it flags *uncertainty*, not
  *confident error* — the bat-and-ball trap scored 0.90 while wrong. The
  0.4 default is a safety net; CALIBRATE against graded answers at kickoff
  (that's also when min-token-prob may beat the mean).
- Deps installed in `.venv/` (torch 2.8.0, transformers 4.57.6, Python 3.9).
- Earlier multi-agent-review fixes (per-task error handling, fp32-on-CPU
  guard, billing-aware retries, etc.) now regression-checked in REAL runs.

### REMAINING work
1. **ROCm on real AMD hardware** (issue #4, half done): ROCm torch is now
   the Docker DEFAULT (`TORCH_INDEX` build arg, rocm6.4; `make build-cpu`
   for the small CPU image; `make docker-run-gpu` passes the GPU devices).
   The ROCm image build + a real generation on an AMD GPU still need to be
   exercised — needs AMD Developer Cloud access at kickoff. The dev Mac
   (arm64) cannot run ROCm wheels; it only builds the image via emulation.
2. **Everything blocked on the reveal**: task format (`load_tasks`), model
   allow-list, threshold calibration on real tasks, task-specific
   post_check validator, single-shot vs multi-step decision (see below).

### ALIGNMENT with Track 1 rules — checked 2026-07-04
Verified against the live competition rules (lablab.ai / web3voyager): the
design MATCHES every scored requirement — real-time local-vs-remote routing,
Fireworks remote, local = 0 tokens, accuracy-threshold-aware, containerized,
model-agnostic/reconfigurable. One alignment RISK remains (the CPU-vs-ROCm
risk was resolved 2026-07-04 by making ROCm the default):

1. **Single-shot router vs "complete tasks autonomously."** The code is a
   single-shot router: prompt in → ONE model call → answer out. If Track 1's
   revealed tasks need multi-step work (tool calls, decompose-then-route-
   each-step, react loop), this is **not an agent yet** and needs a task loop
   around `run_task`. Confirm the task shape at kickoff and decide whether
   single-shot is enough.

Timeline: kickoff expected **~7 July 2026**; the competition runs to
**11 July 2026**.

## 4. Commands

```bash
make test            # offline wiring test — run after EVERY change (~50 ms, no deps)
make mock            # mock run of the sample task file
make run             # real run: needs .venv deps + FIREWORKS_API_KEY exported
make build           # docker build, linux/amd64, ROCm torch (submission default)
make build-cpu       # small CPU-torch image (no-GPU environments / fast smoke test)
make docker-run      # containerized run, logs/ mounted out
make docker-run-gpu  # same + passes AMD GPU devices into the container
python3 main.py --tasks X.json --threshold 0.7   # calibration sweep
python3 scripts/calibrate.py                     # analyze the sweep, pick threshold
```

Env: copy `.env.example` → `.env` (already done on the dev Mac, with a
working key — **`.env` is gitignored, never commit it; share the key only by
private message**). main.py does NOT auto-load `.env`: use
`set -a; source .env; set +a` in the shell, or `--env-file .env` for Docker.
`AGENT_MOCK=1` = no weights, no network.

## 5. Kickoff-day playbook (priority order)

0. ~~FIRST: prove the real path works~~ **DONE 2026-07-04** (see §3): real
   local + real remote answers verified, Docker built and smoke-tested.
   Remaining hardware step: one ROCm build + real generation on AMD
   Developer Cloud once access opens.
1. Set `LOCAL_MODEL_NAME` / `REMOTE_MODEL_NAME` (env or `config.py`). Prefer
   a pre-quantized local checkpoint (GPTQ/AWQ — bitsandbytes is NVIDIA-only).
2. If the task format differs, adapt **only** `load_tasks()` in `main.py`.
3. **Calibrate the threshold** — highest-leverage hour: sweep
   `--threshold 0.4/0.55/0.7` on revealed samples, then
   `python3 scripts/calibrate.py --accuracy grades.json --min-accuracy <bar>`
   — it tabulates tokens vs threshold per run and recommends the LOWEST
   threshold that clears the bar. (Lowering-threshold savings are replayable
   from the log; raising needs a rerun. Grading answers stays manual.)
4. If outputs are checkable (exact match / JSON schema / tests), add a
   validator to `router.post_check` — biggest accuracy upgrade available.
5. Add task-set keywords to `_SIGNAL_PATTERNS` in `confidence.py`; fill
   `FORCE_ROUTE_BY_CATEGORY` in `router.py` if tasks come labeled.
6. Optional: draft-and-judge scorer (local model drafts, mean token log-prob
   as confidence) — local compute is free, costs only latency. Slot into
   `ConfidenceEstimator.scorers`.
7. Uncomment the model-bake `RUN` line in the Dockerfile so the scoring run
   downloads nothing. ROCm torch is already the build DEFAULT (rocm6.4 via
   the `TORCH_INDEX` build arg); `_pick_device()` treats ROCm as `cuda`.
   CPU-only scoring box → `make build-cpu` instead.

## 6. Known quirks (accepted, don't "fix" blindly)

- A correct answer that *starts* with a hedge (e.g. translating
  "je ne sais pas" → "I don't know") still escalates. Cost: one paid retry,
  never lost accuracy. Special-case only if the real task set makes it common.
- `test_harness.py` pins `CONFIDENCE_THRESHOLD=0.55` internally — its
  assertions are calibrated to that; don't remove the pin.
- Mock-mode token counts are fake (word counts) — fine for wiring tests,
  meaningless for calibration.
