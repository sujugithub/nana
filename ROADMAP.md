# Roadmap

Six workstreams, sized so one person can own each without colliding with the
others — the module boundaries in `ARCHITECTURE.md` are what make that
possible. Claim one by putting your name against it and opening a branch.

Priorities: **P0** blocks the evaluation · **P1** is the core contribution ·
**P2** is depth once the foundation holds.

---

## 1. Local backend — make the cheap tier genuinely good
**Owner:** _unclaimed_ · **Touches:** `local_model.py`, `requirements.txt` · **P0**

The local tier is currently a 1.5B model running fp32 on CPU through
transformers, which on Apple Silicon ignores the GPU entirely. That is close to
the slowest possible configuration and it caps the whole project: **a stronger
local tier raises the local share, which is the headline result.**

- [ ] Swap the backend to MLX or llama.cpp with Q4 quantisation (Metal
      accelerated). Keep the `schemas.py` contract so nothing else changes.
- [ ] Move up in model size — a 7–8B model at Q4 is ~4.5 GB and fits
      comfortably in 16 GB. Evaluate a reasoning-distilled checkpoint, which
      should attack the math and logic categories that currently always escalate.
- [ ] Benchmark load time, tokens/sec, and peak memory before and after.
- [ ] Decide whether Docker stays in the local path — Docker on macOS runs in a
      VM and cannot see Metal, so containerised local inference is CPU-only.

## 2. Evaluation harness — the thing the project is graded on
**Owner:** _unclaimed_ · **Touches:** new `eval/` · **P0**

Implements [`EVALUATION.md`](EVALUATION.md). Nothing else can be measured until
this exists.

- [ ] Dataset loaders for the eight categories, sampling 300–500 items each.
- [ ] Programmatic graders (exact match, unit-test execution, span F1).
- [ ] The four baselines: all-local, all-remote, random-at-rate-p, oracle.
- [ ] Pareto plotting, bootstrap confidence intervals, paired tests.
- [ ] LLM judge for summarisation, plus the human-validation sample and the
      verbosity-bias check.

## 3. Router policy — the core contribution
**Owner:** _unclaimed_ · **Touches:** `routing/`, `evaluation/`, `router.py` · **P1**

The learned-router machinery is BUILT (see `docs/LEARNED_ROUTING.md`): a
versioned outcome-dataset format, leakage-safe group splits, two trained +
calibrated candidates, data-driven threshold policies, a full offline
evaluator with all baselines, and `ROUTER_MODE=learned|auto` runtime
integration — proven end-to-end on synthetic toy data. What remains is
feeding it real data and hardening the post-generation side:

- [x] **Learned router** — classifier on (prompt features →
      did-local-get-it-right); RouteLLM-style threshold calibration.
      *(pipeline complete; needs the real dataset from workstream 2)*
- [x] Report AUC for each signal separately (training report covers the
      heuristic reference, both candidates, and all three post-gen
      statistics).
- [x] **Alternative post-gen statistics** — min token prob and low-confidence
      fraction are now computed, logged, gate-selectable
      (`LOCAL_CONF_STAT`), and AUC-compared per training run.
- [ ] **Collect the real dataset** (`scripts/collect_outcomes.py`) and
      retrain/evaluate; only then change any default.
- [ ] **Calibrate the logprob gate** on real graded answers — the machinery
      reports the AUCs; the calibration decision needs real data.
- [ ] **Self-consistency** — sample the local model k times and measure
      disagreement. This catches *confident-wrong*, which logprobs cannot, and
      local compute is cheap.
- [ ] *(carried forward)* Task-specific output validators in `post_check` — if
      a category has checkable output, verifying it beats guessing at it.

## 4. Cost model — replace the zero-token fiction
**Owner:** _unclaimed_ · **Touches:** `token_tracker.py`, `scripts/calibrate.py` · **P1**

The competition counted local tokens as zero. That was a scoring rule, not a
fact. Without a real cost model the router is optimising nothing meaningful.

- [ ] Define the objective: API dollars + amortised local compute, and/or
      latency, and/or energy (macOS exposes real power metrics).
- [ ] Instrument local inference time and energy per query.
- [ ] Extend `scripts/calibrate.py` to recommend **both** thresholds —
      `CONFIDENCE_THRESHOLD` and `LOGPROB_CONFIDENCE_THRESHOLD` — from one
      graded sweep. *(carried forward)*
- [ ] Include discarded local generations from escalations in the accounting.

## 5. Demo and observability
**Owner:** _unclaimed_ · **Touches:** `scripts/banana.py`, new dashboard · **P2**

- [ ] Extend the `banana` CLI as the human-facing entry point.
- [ ] A dashboard over `logs/usage.jsonl`: routing mix, cost over time,
      confidence distributions, escalation reasons.
- [ ] A live demo path suitable for the final presentation.

## 6. Report, reproducibility, CI
**Owner:** _unclaimed_ · **Touches:** `docs/`, CI config · **P2**

- [ ] CI running `make test` on every PR.
- [ ] Pinned dependencies and a documented one-command reproduction of the
      headline results.
- [ ] The written report: related work (cascades, FrugalGPT, RouteLLM,
      conditional computation such as mixture-of-experts), method, results,
      limitations.

---

## Open design question

**Single-shot versus multi-step.** *(carried forward)* The agent currently does
prompt → one model call → answer. If the target workload needs decomposition or
tool use, a task loop around `run_task` is required, routing each step
independently. Decide this deliberately — it changes the architecture — and
record the decision in `ARCHITECTURE.md`.

## Housekeeping

- [ ] Rotate the API key in `.env`. It was in use while the repository and
      container images were public. `.env` was never committed (verified), but
      rotation is cheap insurance.
- [ ] Regenerate `graphify-out/` — its community labels still carry
      competition-era vocabulary.
