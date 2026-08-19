# banana — Hybrid Token-Efficient Routing Agent

A routing layer that decides, **per request**, whether a small **local** model
can answer it or whether the question deserves a frontier **remote** model.
Cheap questions are answered locally at near-zero marginal cost; only the
genuinely hard ones pay for a large model.

**The research question:** can a router using cheap difficulty signals plus the
local model's own self-assessed confidence beat single-model baselines on the
accuracy-versus-cost frontier — and beat *random* routing at the same spend?

See [`EVALUATION.md`](EVALUATION.md) for how we intend to answer that,
[`ARCHITECTURE.md`](ARCHITECTURE.md) for how the system works, and
[`ROADMAP.md`](ROADMAP.md) for what we're building next.

> **Project history.** This began as a 4-day AMD Developer Competition
> submission. The competition-era documents are archived in
> [`docs/history/`](docs/history/) for provenance — including the measured
> results — but they describe rules that no longer apply. In particular the
> competition scored local tokens as **zero**; the capstone replaces that with
> a real cost model.

## Design rationale

Routing is worthwhile only if the cheap tier is genuinely cheaper *and* you can
tell, in advance or shortly after, when it isn't good enough. Five ideas drive
the design:

1. **Default to local.** Local inference is far cheaper per token than a
   frontier API, so the router is biased toward local and treats routing as
   *risk detection* — "is there reason to believe the small model will fail?"
2. **Detect risk cheaply.** `confidence.py` scores each query with zero-cost
   heuristics (length, plus math/code/logic/reasoning/multi-part signals and
   sentiment/NER/summarisation boosts). Only queries that look beyond a small
   model go straight to remote.
3. **Bound the accuracy downside.** When local runs, two further checks gate
   the answer: `router.post_check` inspects the output for small-model failure
   modes (empty output, repetition loops, prompt echo, hedging), and a
   **draft-and-judge confidence gate** reads the model's own mean token
   probability (`local_confidence`) — below `LOGPROB_CONFIDENCE_THRESHOLD`
   (default 0.4) the task **escalates to remote**. A discarded local attempt
   costs compute and latency, not API spend.
4. **Make the cost observable.** `token_tracker.py` writes one JSONL line per
   task — confidence, active threshold, per-signal scores, local confidence,
   and a run_id — so calibration is a log replay, not a rerun.
5. **No single failure kills the run.** Escalation failures keep the flagged
   local answer, remote failures fall back to a local attempt, and any other
   per-task error is recorded and skipped: an answer always beats no answer.

```
task ──▶ Router.decide  (confidence.py heuristics — zero cost)
           │
           ├─ score ≥ threshold ──▶ LocalModel  (no API spend)
           │                          │
           │                     Router.post_check(output)
           │                          ├─ looks good ──▶ answer
           │                          └─ looks bad ───▶ escalate ─┐
           │                                                      ▼
           └─ score < threshold ─────────▶ RemoteClient (Fireworks, billable)
                                                                   │
every step ──▶ TokenTracker (logs/usage.jsonl + summary)           ▼
                                                                answer
```

## Module map

| File | Role |
| --- | --- |
| `main.py` | Orchestrator + CLI. `run_task()` is the decide→execute→check→account loop; `build_router()` picks the router for `ROUTER_MODE`. |
| `router.py` | Decision layer: pre-route + post-check + escalation policy. |
| `confidence.py` | Heuristic scorers estimating "can the local model handle this?" |
| `routing/` | **Learned pre-router**: outcome-dataset format, leakage-safe splits, training + calibration + threshold policies, artifact I/O, runtime router. See [`docs/LEARNED_ROUTING.md`](docs/LEARNED_ROUTING.md). |
| `evaluation/` | Offline evaluator: all-local / all-remote / random / heuristic / learned / oracle on the held-out test split, with bootstrap CIs and a Pareto chart. |
| `tests/` | Deterministic offline suite for the learned routing system (`make test`). |
| `local_model.py` | HF transformers wrapper (lazy load, chat template, exact token counts). |
| `remote_client.py` | Remote model client (`/chat/completions`, retries, usage-based counts). |
| `token_tracker.py` | Local-vs-remote accounting, JSONL audit log, run summary. |
| `config.py` | Every knob, env-overridable. The one file to touch when swapping models. |
| `schemas.py` | Shared `Task` / `Completion` dataclasses — the contract between backends. |
| `test_harness.py` | Offline end-to-end wiring test (mock mode, stdlib only). |
| `scripts/banana.py` | Interactive CLI + `--demo` mode with a session token graph. |
| `scripts/calibrate.py` | Threshold calibration analysis over `logs/usage.jsonl`. |
| `scripts/make_toy_dataset.py` | Synthetic outcome dataset for offline end-to-end runs. |
| `scripts/collect_outcomes.py` | Collect REAL both-model outcomes (requires explicit `--run-paid-calls`). |

## Quickstart

```bash
# 0) Wiring test — offline, zero dependencies:
python3 test_harness.py

# 1) Mock run of the sample task file (no model, no network):
python3 main.py --tasks tasks/sample_tasks.json --mock

# 2) Real run:
pip install -r requirements.txt
cp .env.example .env        # then add your API key
python3 main.py --tasks tasks/sample_tasks.json

# 3) Interactive CLI (model loads once, stays warm):
python3 scripts/banana.py            # ask questions at the `banana ›` prompt
python3 scripts/banana.py --demo     # 8-category run + token graph

# 4) Learned router — offline end-to-end loop on synthetic data:
make toy-data && make train && make evaluate
ROUTER_MODE=learned python3 main.py --tasks tasks/sample_tasks.json --mock
```

The pre-router now has two implementations selected by `ROUTER_MODE`:
`heuristic` (the keyword rules above, the default), `learned` (a trained,
calibrated classifier predicting *P(local answer acceptable)* from real
model outcomes), and `auto` (learned if its artifact loads, else heuristic
with a warning). [`docs/LEARNED_ROUTING.md`](docs/LEARNED_ROUTING.md) covers
dataset collection, training, threshold policies, and evaluation.

Never commit `.env` — it is gitignored and holds a live API key.

### Batch mode

The agent also runs headlessly, reading a JSON task list and writing a JSON
answer list. This is how it is evaluated in bulk:

```bash
python3 main.py --input tasks/demo_tasks.json --output results.json
```

Input is `[{task_id, prompt}]`; output is `[{task_id, answer}]`, always valid
JSON, exit 0 on success.

Container images build with `make build` (ROCm torch, for AMD GPU hosts) or
`make build-cpu` (smaller, CPU-only). `make docker-run-harness` runs batch mode
in-container against the `/input` and `/output` mounts.

## Debugging

- Every routing decision prints its confidence **and per-signal breakdown** —
  "why did task 7 go remote?" is answered by the log line itself.
- `AGENT_MOCK=1` (or `--mock`) isolates wiring bugs from model/API bugs.
- `logs/usage.jsonl` is the audit trail: one line per task, replayable.
- `make test` after every change; it runs in ~50 ms with no deps.

## Team

Six-person university capstone. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
the workflow and [`ROADMAP.md`](ROADMAP.md) for the workstreams — each is sized
so one person can own it without colliding with the others.
