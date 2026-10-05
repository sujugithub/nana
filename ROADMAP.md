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
**Owner:** _unclaimed_ · **Touches:** `evaluation/`, `scripts/collect_outcomes.py` · **P0**

Implements [`EVALUATION.md`](EVALUATION.md). The harness now EXISTS
(`evaluation/run.py`) and has been run once on real data — the remaining work
is scale and coverage, not machinery.

- [x] The four baselines: all-cheap, all-strong, random-at-rate-p, oracle
      (plus the heuristic and every learned candidate).
- [x] Pareto plotting, bootstrap confidence intervals, paired tests,
      per-category results, workload mixtures.
- [x] Programmatic graders: exact / numeric / contains / choice
      (`scripts/collect_outcomes.py`).
- [x] Three categories collected at pilot scale: 200 GSM8K, 200 MMLU,
      200 BBH — 600 graded outcomes, 420/90/90 split.
- [ ] **Scale to 300–500 items per category.** n = 90 test rows cannot
      separate the routers; every quality CI currently includes zero. Build
      a new, disjoint expansion with a preregistered split and evaluation
      policy. The old local `real_extension_qwen_pro` work file is an
      interrupted/obsolete collection, not the next evaluation set.
- [ ] The remaining five categories: sentiment, NER, summarisation, code
      generation, code debugging (the last two need unit-test execution).
- [ ] Repeated seeds — report the spread across splits, not one lucky split.
- [ ] LLM judge for summarisation, plus the human-validation sample and the
      verbosity-bias check.

## 3. Router policy — the core contribution
**Owner:** _unclaimed_ · **Touches:** `routing/`, `evaluation/`, `router.py` · **P1**

The learned-router machinery is BUILT (see `docs/LEARNED_ROUTING.md`): a
versioned outcome-dataset format, leakage-safe group splits, two trained +
calibrated candidates, data-driven threshold policies, a full offline
evaluator with all baselines, and `ROUTER_MODE=learned|auto` runtime
integration. It is proven end-to-end on synthetic toy data AND trained once
on a real 600-outcome pilot (`docs/HANDOFF.md` §3). What remains is more
real data and hardening the post-generation side:

- [x] **Learned router** — classifier on (prompt features →
      did-local-get-it-right); RouteLLM-style threshold calibration.
- [x] Report AUC for each signal separately (training report covers the
      heuristic reference, both candidates, and all three post-gen
      statistics).
- [x] **Alternative post-gen statistics** — min token prob and low-confidence
      fraction are now computed, logged, gate-selectable
      (`LOCAL_CONF_STAT`), and AUC-compared per training run.
- [x] **Collect a real dataset at pilot scale** — 600 graded Qwen-2.5-1.5B vs
      DeepSeek-V4-Pro outcomes; trained, evaluated, artifact deployed to the
      demo. Result: leads the heuristic and random on point estimates, but
      **no quality CI clears zero at n = 90**, and unsafe-local is worse than
      the heuristic (18.9% vs 15.6%). Not yet a claim.
- [ ] **Scale that dataset** (workstream 2) and re-run before changing any
      default or writing a results claim.
- [ ] **Retrain under the `max_unsafe` policy** and report the safety/cost
      frontier — the deployed artifact is the `quality_floor` pick, which is
      the riskier of the two.
- [ ] **Collect a `remote_pair` dataset.** No artifact exists for the
      two-Fireworks pair, so that demo mode is heuristic-only by necessity.
- [ ] **Calibrate the logprob gate** on real graded answers. Pilot post-gen
      AUCs: mean 0.730, min 0.673, low_frac 0.630 — `mean` stays the default
      until more data says otherwise.
- [ ] **Self-consistency** — sample the local model k times and measure
      disagreement. This catches *confident-wrong*, which logprobs cannot, and
      local compute is cheap.
- [ ] *(carried forward)* Task-specific output validators in `post_check` — if
      a category has checkable output, verifying it beats guessing at it.

## 4. Cost model — replace the zero-token fiction
**Owner:** _unclaimed_ · **Touches:** `token_tracker.py`, `scripts/calibrate.py` · **P1**

The earlier competition counted local tokens as zero. That was a scoring
rule, not a fact. Without a real cost model the router is optimising nothing
meaningful.

- [ ] Define the objective: API dollars + amortised local compute, and/or
      latency, and/or energy (macOS exposes real power metrics).
- [ ] Instrument local inference time and energy per query.
- [ ] Extend `scripts/calibrate.py` to recommend **both** thresholds —
      `CONFIDENCE_THRESHOLD` and `LOGPROB_CONFIDENCE_THRESHOLD` — from one
      graded sweep. *(carried forward)*
- [ ] Include discarded local generations from escalations in the accounting.

## 5. Demo and observability
**Owner:** _unclaimed_ · **Touches:** `webui/`, `scripts/banana.py`, new dashboard · **P2**

- [x] **Browser demo** (`webui/`, `make ui`): Hybrid / Remote / Fully-local
      modes, per-session model selection, plain-language config, provider and
      billing shown honestly, mock-only unless started with `--real`.
      See [`docs/DEMO_UI.md`](docs/DEMO_UI.md).
- [x] A live demo path suitable for the final presentation.
- [ ] Extend the `banana` CLI as the terminal-side entry point.
- [ ] A dashboard over `logs/usage.jsonl`: routing mix, cost over time,
      confidence distributions, escalation reasons.
- [ ] Finish the persistent chat UI. SQLite-backed conversations, bounded
      multi-turn context, and history/search are implemented; streaming and
      stop-generation remain. RAG is optional and is not required for chat memory.

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
