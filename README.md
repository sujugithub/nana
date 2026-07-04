# Hybrid Token-Efficient Routing Agent

Starter skeleton for **AMD Developer Competition ACT II — Track 1**.

An agent that completes tasks autonomously, deciding per task whether to run a
small **local** model (counts as **zero** tokens under the scoring rules) or
call a **remote** model via the Fireworks AI API (billable, but accurate).

## Scoring model → design

`score = token count + output accuracy`, local = 0 tokens, and accuracy below
a threshold is penalized. Four consequences drive the whole design:

1. **Default local.** Every task the small model can handle is free — so the
   router is biased toward local and treats routing as *risk detection*.
2. **Detect risk for free.** `confidence.py` scores each query with zero-cost
   heuristics (length, math/code/reasoning/multi-part signals). Only queries
   that look beyond a 1–3B model go remote.
3. **Bound the accuracy downside.** When local runs, `router.post_check`
   inspects the output for small-model failure modes (empty output,
   repetition loops, prompt echo, hedging) and **escalates to remote** if it
   looks wrong. A failed local attempt costs nothing but latency.
4. **Make the bill observable.** `token_tracker.py` writes one JSONL line per
   task — including confidence, the active threshold, per-signal scores, and
   a run_id — so calibration analysis happens on the log, not by rerunning.
5. **No single failure kills the run.** Escalation failures keep the flagged
   local answer, remote failures fall back to a local attempt, and any other
   per-task error is recorded and skipped — an answer always beats no answer.

```
task ──▶ Router.decide  (confidence.py heuristics — zero cost)
           │
           ├─ score ≥ threshold ──▶ LocalModel  (0 billable tokens)
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
| `main.py` | Orchestrator + CLI. `run_task()` is the decide→execute→check→account loop. |
| `router.py` | Decision layer: pre-route + post-check + escalation policy. |
| `confidence.py` | Heuristic scorers estimating "can the local model handle this?" |
| `local_model.py` | HF transformers wrapper (lazy load, chat template, exact token counts). |
| `remote_client.py` | Fireworks AI client (`/chat/completions`, retries, usage-based counts). |
| `token_tracker.py` | Local-vs-remote accounting, JSONL audit log, run summary. |
| `config.py` | Every knob, env-overridable. The one file to touch on kickoff day. |
| `schemas.py` | Shared `Task` / `Completion` dataclasses. |
| `test_harness.py` | Offline end-to-end wiring test (mock mode, stdlib only). |

## Quickstart

```bash
# 0) Wiring test — offline, zero dependencies:
python3 test_harness.py

# 1) Mock run of the sample task file:
python3 main.py --tasks tasks/sample_tasks.json --mock

# 2) Real run:
pip install -r requirements.txt
export FIREWORKS_API_KEY=fw-...
python3 main.py --tasks tasks/sample_tasks.json

# 3) Docker (or `make build` / `make docker-run`):
docker build --platform=linux/amd64 -t hybrid-router-agent .   # scoring host is x86_64
docker run --rm --env-file .env -v "$(pwd)/logs:/app/logs" hybrid-router-agent
```

## Kickoff-day checklist

1. **Swap models**: set `LOCAL_MODEL_NAME` / `REMOTE_MODEL_NAME` (env vars or
   `config.py`). Prefer a pre-quantized local checkpoint (GPTQ/AWQ) for the
   limited-compute scoring box.
2. **Adapt task input**: if the task format differs, edit `load_tasks()` in
   `main.py` — nothing else should need to change.
3. **Calibrate the threshold** (the highest-leverage hour of the day): run
   the revealed sample tasks at several thresholds
   (`--threshold 0.4 / 0.55 / 0.7`), then run `python3 scripts/calibrate.py`
   — it groups `logs/usage.jsonl` by run, tabulates billable tokens vs
   threshold, and (given `--accuracy grades.json`, a `{task_id: true/false}`
   file, plus `--min-accuracy`) recommends the LOWEST threshold that clears
   the accuracy bar. It also replays what LOWERING each threshold would have
   saved (remote costs of flipped tasks are already logged; raising needs a
   rerun). Grading answers is task-set-specific and stays manual.
4. **Add task-specific signals**: extend `_SIGNAL_PATTERNS` in
   `confidence.py`; if tasks carry categories, fill in
   `FORCE_ROUTE_BY_CATEGORY` in `router.py`.
5. **Add a real validator** if outputs are checkable (exact match, JSON
   schema, tests): plug it into `router.post_check` — it's the single biggest
   accuracy upgrade available.
6. **Bake the model into the image**: uncomment the `RUN python -c ...` line
   in the Dockerfile so the scoring run downloads nothing.
7. **AMD GPU in the scoring env?** Switch the torch install to the ROCm wheel
   index (comment in Dockerfile). `LocalModel._pick_device()` already treats
   ROCm as `cuda`.

## Debugging live

- Every routing decision prints its confidence **and per-signal breakdown** —
  "why did task 7 go remote?" is answered by the log line itself.
- `AGENT_MOCK=1` (or `--mock`) isolates wiring bugs from model/API bugs.
- `logs/usage.jsonl` is the audit trail: one line per task, replayable.
- `make test` after every change; it runs in ~50 ms with no deps.
