# HANDOFF — Hybrid Token-Efficient Routing Agent

> Paste this file (or point the session at it) when continuing work in a new
> Claude session. It contains everything needed to take over with zero prior
> context. **The SCAFFOLD is complete and passes mock tests — don't
> re-scaffold or restructure it. But the real routing has NEVER run — see
> §3. Your job is to make it actually work, then tune it.**

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

Project root: `~/Desktop/competition` (flat Python modules, no package).

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

**"Passes tests" here means passes MOCK tests only. Mock mode is a stub** —
`local_model.py` / `remote_client.py` return hardcoded strings like
`f"[mock-local] concise answer to: {prompt[:60]}"`. They do NOT load a model
or call an API. So the green test proves the *plumbing* (decide → route →
log) connects — and nothing about the part that actually scores points.

### VERIFIED (safe to rely on)
- All 9 modules compile; `python3 test_harness.py` passes offline, zero deps.
- Mock end-to-end run writes `logs/usage.jsonl` with full calibration fields.
- A multi-agent review (4 lenses + adversarial verify) found 18 issues; the
  real ones are FIXED and regression-checked *in mock*: per-task error
  handling (one remote failure no longer kills the run), fp32-on-CPU guard,
  Dockerfile layer order + `--platform=linux/amd64`, billing-aware retries
  (429 honors Retry-After), `post_check_min_chars=1`, word-bounded hedge
  detection, `"metadata": null` handling.

### UNPROVEN — this is the actual remaining work
1. **No real local model has EVER run.** The `transformers` load + chat
   template + `generate` + token-count path in `local_model.py` has zero real
   execution. First real run may break on dtype/device/template/token count.
2. **No real Fireworks call has EVER run.** Payload, auth header, and `usage`
   parsing in `remote_client.py` are unvalidated against the live API.
3. **Deps were never installed** in the dev env (`pip install -r
   requirements.txt`) — a real run can't even start until they are.
4. **`docker build` was never executed** — only reasoned about. Build once as
   a smoke test.

### ALIGNMENT with Track 1 rules — checked 2026-07-04
Verified against the live competition rules (lablab.ai / web3voyager): the
design MATCHES every scored requirement — real-time local-vs-remote routing,
Fireworks remote, local = 0 tokens, accuracy-threshold-aware, containerized,
model-agnostic/reconfigurable. Two alignment RISKS remain to resolve:

1. **CPU default vs AMD-GPU/ROCm platform.** The rules say build "on AMD GPUs
   in the cloud, using AMD Developer Cloud, ROCm, and the Fireworks AI API."
   But `Dockerfile` installs CPU torch by default and treats ROCm as an
   optional swap-in comment — so the AMD hardware is currently left on the
   table, and the ROCm path is the LEAST-tested one (a ROCm build won't even
   compile on the arm64 dev Mac). If the scoring box has an AMD GPU, make
   ROCm the DEFAULT, not an afterthought. (See playbook step 7.)

2. **Single-shot router vs "complete tasks autonomously."** The code is a
   single-shot router: prompt in → ONE model call → answer out. If Track 1's
   revealed tasks need multi-step work (tool calls, decompose-then-route-
   each-step, react loop), this is **not an agent yet** and needs a task loop
   around `run_task`. Confirm the task shape at kickoff and decide whether
   single-shot is enough.

Deadline: the competition runs to **11 July 2026**.

### Why this got left here
The dev session ran out of usage credits (a heavy multi-agent review burned
~460k tokens) before any real run could happen. Everything above is why the
prior "done" claim was really "scaffold done, real path untested."

## 4. Commands

```bash
make test        # offline wiring test — run after EVERY change (~50 ms, no deps)
make mock        # mock run of the sample task file
make run         # real run: needs pip install -r requirements.txt + FIREWORKS_API_KEY
make build       # docker build, pinned to linux/amd64
make docker-run  # containerized run, logs/ mounted out
python3 main.py --tasks X.json --threshold 0.7   # calibration sweep
```

Env: copy `.env.example` → `.env`. `AGENT_MOCK=1` = no weights, no network.

## 5. Kickoff-day playbook (priority order)

0. **FIRST: prove the real path works** (this is the UNPROVEN work in §3).
   `pip install -r requirements.txt`, set `FIREWORKS_API_KEY`, then a real
   `make run` on a tiny model to shake out load/template/token-count/API
   bugs. Then one `make build`. Do NOT tune anything until a real local
   answer and a real remote answer have each come back once.
1. Set `LOCAL_MODEL_NAME` / `REMOTE_MODEL_NAME` (env or `config.py`). Prefer
   a pre-quantized local checkpoint (GPTQ/AWQ — bitsandbytes is NVIDIA-only).
2. If the task format differs, adapt **only** `load_tasks()` in `main.py`.
3. **Calibrate the threshold** — highest-leverage hour: sweep
   `--threshold 0.4/0.55/0.7` on revealed samples, group `logs/usage.jsonl`
   lines by `run_id`, pick the LOWEST threshold that clears the accuracy bar.
   (Lowering-threshold savings are replayable from the log; raising needs a
   rerun.)
4. If outputs are checkable (exact match / JSON schema / tests), add a
   validator to `router.post_check` — biggest accuracy upgrade available.
5. Add task-set keywords to `_SIGNAL_PATTERNS` in `confidence.py`; fill
   `FORCE_ROUTE_BY_CATEGORY` in `router.py` if tasks come labeled.
6. Optional: draft-and-judge scorer (local model drafts, mean token log-prob
   as confidence) — local compute is free, costs only latency. Slot into
   `ConfidenceEstimator.scorers`.
7. Uncomment the model-bake `RUN` line in the Dockerfile so the scoring run
   downloads nothing. AMD GPU on the scoring box → swap torch index to ROCm
   (comment in Dockerfile); `_pick_device()` already treats ROCm as `cuda`.

## 6. Known quirks (accepted, don't "fix" blindly)

- A correct answer that *starts* with a hedge (e.g. translating
  "je ne sais pas" → "I don't know") still escalates. Cost: one paid retry,
  never lost accuracy. Special-case only if the real task set makes it common.
- `test_harness.py` pins `CONFIDENCE_THRESHOLD=0.55` internally — its
  assertions are calibrated to that; don't remove the pin.
- Mock-mode token counts are fake (word counts) — fine for wiring tests,
  meaningless for calibration.
