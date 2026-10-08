# Nana — Hybrid Token-Efficient Routing Agent

Nana's native water chat and workspace run on port 8642. On the owner's Mac,
connect the T7 and run `.venv/bin/python scripts/start_workspace.py --real`.
See [handoff.md](handoff.md) for the current application state and next steps,
and [workspace setup](docs/WORKSPACE.md) for installation and connections.

A learned routing layer that decides, **per request**, whether the configured
cheap tier is likely to be adequate or the task needs the strong tier. It has
two selectable deployments, so the same FYP can be demonstrated cheaply now
and evaluated with a real local model later:

| `TIER_MODE` | Cheap tier | Strong tier | Purpose |
| --- | --- | --- | --- |
| `local_remote` (default) | Qwen 2.5 1.5B on the machine | gpt-oss-120b on Fireworks | Local cheap tier; local generation may load model weights. |
| `remote_pair` | Nemotron Lightning 3.5 on Fireworks | gpt-oss-120b on Fireworks | Optional two-remote demo; both tiers are billed. |

The router predicts *P(cheap-tier answer acceptable)*. A trained artifact is
valid only for the exact model pair used to collect its outcome data. The
existing learned artifact was trained on Qwen → DeepSeek V4 Pro, not these
current live defaults; use the heuristic until new outcomes are collected.

**The research question:** can an outcome-trained router beat single-model
baselines on the quality-versus-cost frontier — and beat *random* routing at
the same spend? The local deployment additionally tests whether local token
confidence improves the decision after generation.

New to the repository, or picking it up after a break? Start with
[`docs/HANDOFF.md`](docs/HANDOFF.md) — current state, what the evidence
supports, and what to do next.

Canonical public repository: [github.com/sujugithub/nana](https://github.com/sujugithub/nana).

See [`EVALUATION.md`](EVALUATION.md) for how we intend to answer that,
[`ARCHITECTURE.md`](ARCHITECTURE.md) for how the system works, and
[`ROADMAP.md`](ROADMAP.md) for what we're building next.

> **Project history.** This began as a 4-day prototype for an AMD developer
> competition. The competition-era documents are archived in
> [`docs/history/`](docs/history/) for provenance — including the measured
> results — but they describe rules that no longer apply. In particular the
> competition scored local tokens as **zero**; the capstone replaces that with
> a real cost model.

## Design rationale

Routing is worthwhile only if the cheap tier is genuinely cheaper *and* you can
tell, in advance or shortly after, when it isn't good enough. Five ideas drive
the design:

1. **Prefer the cheap tier when evidence supports it.** The router treats
   routing as risk detection: “is there reason to believe this model will
   fail?” In `local_remote` that tier has no API charge; in `remote_pair` it
   is a lower-cost API model, not local or free.
2. **Detect risk cheaply.** `confidence.py` scores each query with zero-cost
   heuristics (length, plus math/code/logic/reasoning/multi-part signals and
   sentiment/NER/summarisation boosts). Only queries that look beyond a small
   model go straight to remote.
3. **Bound the accuracy downside.** When the cheap tier runs,
   `router.post_check` inspects the output for failure modes (empty output,
   repetition loops, prompt echo, hedging). In `local_remote`, a second
   **draft-and-judge confidence gate** reads the model's own mean token
   probability (`local_confidence`) — below `LOGPROB_CONFIDENCE_THRESHOLD`
   (default 0.4) the task **escalates to strong**. Fireworks Flash does not
   expose this local-logit signal, so `remote_pair` uses the surface gate.
4. **Make the cost observable.** `token_tracker.py` writes one JSONL line per
   task — confidence, active threshold, per-signal scores, local confidence,
   and a run_id — so calibration is a log replay, not a rerun.
5. **No single failure kills the run.** Escalation failures keep the flagged
   cheap answer, strong-tier failures fall back to a cheap attempt, and any other
   per-task error is recorded and skipped: an answer always beats no answer.

```
task ──▶ Router.decide  (heuristic or trained artifact)
           │
           ├─ score ≥ threshold ──▶ cheap tier
           │                          │
           │                     Router.post_check(output)
           │                          ├─ looks good ──▶ answer
           │                          └─ looks bad ───▶ escalate ─┐
           │                                                      ▼
           └─ score < threshold ─────────▶ strong tier (Fireworks)
                                                                   │
every step ──▶ TokenTracker (logs/usage.jsonl + summary)           ▼
                                                                answer
```

## Module map

| File | Role |
| --- | --- |
| `main.py` | Orchestrator + CLI. `run_task()` is the decide→execute→check→account loop; `build_router()` picks the router for `ROUTER_MODE`. |
| `router.py` | Decision layer: pre-route + post-check + escalation policy. |
| `confidence.py` | Heuristic fallback estimating “can the cheap tier handle this?” |
| `routing/` | **Learned pre-router**: outcome-dataset format, leakage-safe splits, training + calibration + threshold policies, artifact I/O, runtime router. See [`docs/LEARNED_ROUTING.md`](docs/LEARNED_ROUTING.md). |
| `evaluation/` | Offline evaluator: all-cheap / all-strong / random / heuristic / learned / oracle on the held-out test split, with bootstrap CIs and a Pareto chart. |
| `tests/` | Deterministic offline suite for the learned routing system (`make test`). |
| `local_model.py` | HF transformers wrapper (lazy load, chat template, exact token counts). |
| `remote_client.py` | Fireworks cheap/strong clients (`/chat/completions`, retries, usage counts). |
| `token_tracker.py` | Provider-aware accounting, estimated API cost, JSONL audit log, run summary. |
| `config.py` | Every knob, env-overridable. The one file to touch when swapping models. |
| `schemas.py` | Shared `Task` / `Completion` dataclasses — the contract between backends. |
| `test_harness.py` | Offline end-to-end wiring test (mock mode, stdlib only). |
| `scripts/banana.py` | Interactive CLI + `--demo` mode with a session token graph. |
| `scripts/calibrate.py` | Threshold calibration analysis over `logs/usage.jsonl`. |
| `scripts/make_toy_dataset.py` | Synthetic outcome dataset for offline end-to-end runs. |
| `scripts/collect_outcomes.py` | Collect REAL both-model outcomes (requires explicit `--run-paid-calls`). |
| `webui/` | Browser demo (Hybrid / Remote / Fully local), stdlib-only server. See [`docs/DEMO_UI.md`](docs/DEMO_UI.md). |
| `chat/` | SQLite conversations, bounded multi-turn context, and atomic chat turns. |
| `api.py` | Optional FastAPI endpoints for a separate frontend; local access by default. |

## Quickstart

```bash
# 0) Wiring test — offline, zero dependencies:
python3 test_harness.py

# 1) Mock run of the sample task file (no model, no network):
python3 main.py --tasks tasks/sample_tasks.json --mock

# 2) Real two-remote demo (both tiers bill Fireworks):
pip install -r requirements.txt
cp .env.example .env        # then add your API key
python3 main.py --tier-mode remote_pair --tasks tasks/sample_tasks.json

# Or keep the local model as the cheap tier:
python3 main.py --tier-mode local_remote --tasks tasks/sample_tasks.json

# 3) Interactive CLI (model loads once, stays warm):
python3 scripts/banana.py            # ask questions at the `banana ›` prompt
python3 scripts/banana.py --demo     # 8-category run + token graph

# 4) Learned router — offline end-to-end loop on synthetic data:
make toy-data && make train && make evaluate
ROUTER_MODE=learned python3 main.py --tasks tasks/sample_tasks.json --mock

# 5) Browser demo UI (mock-only by default; --real enables billable calls):
make ui                              # http://127.0.0.1:8642
# Open http://127.0.0.1:8642/chat for persistent chat (mock by default).
# make ui-real permits real calls; uncheck Mock in chat before sending.

# Optional FastAPI server for another frontend (localhost:8643):
make api                             # mock-only by default
# make api-real permits real calls when Mock is explicitly false.
```

The pre-router now has two implementations selected by `ROUTER_MODE`:
`heuristic` (the keyword rules above, the default), `learned` (a trained,
calibrated classifier predicting *P(cheap-tier answer acceptable)* from real
model outcomes), and `auto` (learned if its artifact loads, else heuristic
with a warning). [`docs/LEARNED_ROUTING.md`](docs/LEARNED_ROUTING.md) covers
dataset collection, training, threshold policies, and evaluation.

### Where the results stand

A **real 600-outcome pilot** has been collected and evaluated: Qwen 2.5 1.5B
(local cheap tier) vs DeepSeek V4 Pro (strong tier) over 200 GSM8K, 200 MMLU
and 200 BIG-Bench Hard prompts, split 420/90/90 group-aware.

On the 90-row test split the learned router reached **0.789** routed quality
against 0.756 (heuristic), 0.707 (random at the same 60% remote rate), 0.411
(all-local) and 0.900 (all-remote), at lower cost than the heuristic.

**That is not yet a claim.** No quality confidence interval clears zero at
n = 90 — including the comparison against random — and the learned router's
unsafe-local rate is *worse* than the heuristic's (18.9% vs 15.6%). The
pipeline is real and the pilot is real; the statistics need more data. Full
numbers, caveats and the next steps are in
[`docs/HANDOFF.md`](docs/HANDOFF.md) §3.

The pilot artifact lives at `artifacts/router_qwen_pro_600.joblib` and the
local `.env` points the browser demo to it. It is valid only for the historical
Qwen → DeepSeek V4 Pro pair; the current live defaults show a retraining
warning when **Learned** is selected. `artifacts/router.joblib` is the
**synthetic toy** router — it
validates the pipeline only and is labelled as such everywhere it appears.

For the live demo, prefer **Hybrid → Local + remote → Heuristic**, or
**Remote → gpt-oss-120b**. The older `deepseek-v4-flash` and
`deepseek-v4-pro` deployments both returned Fireworks `404 NOT_FOUND` for this
account. The current model IDs appeared in a read-only account model list on
2026-10-06, and real gpt-oss-120b replies were verified on 2026-10-08. Starting
`make ui-real` enables real backends but spends nothing by itself—only an
actual Fireworks request is billable.

Never commit `.env` — it is gitignored and holds a live API key. Also do not
run `scripts/collect_outcomes.py --run-paid-calls` until the intended
`TIER_MODE` and exact model pair have been verified; collection executes both
tiers for every task. Paid collection also requires an explicit
`--max-api-cost-usd` ceiling and refuses to start when the configured
worst-case cost exceeds it.

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
- `make test` after every change; it is deterministic and makes no API calls.

## Team

Six-person university capstone. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
the workflow and [`ROADMAP.md`](ROADMAP.md) for the workstreams — each is sized
so one person can own it without colliding with the others.
