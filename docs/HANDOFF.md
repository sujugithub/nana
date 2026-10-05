# HANDOFF — current state

Point a new session (or a new teammate) at this file. It says what exists,
what the measured evidence actually supports, what is safe to run, and what
to do next. Last updated **2026-10-02**.

For the earlier competition-era handoff see
[`history/HANDOFF.md`](history/HANDOFF.md)
— that describes rules that no longer apply and is provenance only.

---

## 0. One-paragraph status

The project is a **two-tier LLM router**: a cheap tier and a strong tier,
with a pre-router deciding per prompt, plus a post-generation cascade that
escalates bad cheap answers. Both the heuristic (keyword) router and a
**learned, calibrated router trained on real observed outcomes** are
implemented, along with a full offline evaluator, a reproducible training
pipeline, a browser demo UI with persistent chat, and 156 deterministic offline tests. A **real
600-outcome pilot** (Qwen 2.5 1.5B local → DeepSeek V4 Pro) has been
collected and evaluated. The learned router is **promising but not yet
statistically conclusive** — see §3 before making any claim about it.

Canonical public repository: [github.com/sujugithub/nana](https://github.com/sujugithub/nana).

## 1. What runs, and how

```bash
make test                      # 156 offline tests, no network, no cost
make ui                        # browser demo, mock-only, 127.0.0.1:8642
make ui-real                   # permit real backends; calls may bill Fireworks
make api                       # FastAPI chat backend, mock-only, 127.0.0.1:8643
python3 main.py --tasks tasks/sample_tasks.json --mock    # batch CLI, mock
python3 scripts/banana.py                                  # interactive CLI
```

Two orthogonal switches drive everything:

| Switch | Values | Meaning |
| --- | --- | --- |
| `TIER_MODE` | `local_remote` · `remote_pair` | what the **cheap tier** is: local Qwen, or Fireworks Nemotron Lightning |
| `ROUTER_MODE` | `heuristic` · `learned` · `auto` | who makes the **pre-route decision** |

The strong tier is always Fireworks (gpt-oss-120b by default). Routes are
named `cheap`/`strong` — policy tiers, not hosting. **Provider is recorded
per completion** (`local` vs `fireworks`) and billing is derived from that,
never inferred from the route name. In `remote_pair`, *both* tiers bill.

For a live presentation use `local_remote` + `heuristic` until a new learned
artifact is trained for the current pair. Both older DeepSeek V4 model IDs
returned Fireworks `404 NOT_FOUND` for this account. The replacement model
IDs were listed by the account on 2026-10-06 but generation is not yet
verified. `make ui-real` itself makes no model request; spending begins only
when a real Fireworks-backed run is submitted.

## 2. Repository map (what to read first)

| Path | Why you care |
| --- | --- |
| `docs/LEARNED_ROUTING.md` | the ML system end to end: dataset → train → threshold → evaluate |
| `docs/DEMO_UI.md` | the browser demo, its three modes and safety rails |
| `ARCHITECTURE.md` | how the runtime fits together and the invariants that must hold |
| `EVALUATION.md` | the methodology the FYP is graded on |
| `ROADMAP.md` | what is done, what is next, who could own it |
| `routing/` | dataset format, splits, features, training, policies, artifact, runtime router |
| `evaluation/run.py` | the one-shot test-split evaluator (all baselines) |
| `webui/` | demo UI: `service.py` is the logic, `server.py` is a thin stdlib wrapper |
| `main.py` | `build_router()`, `build_backends()`, `run_task()` — the cascade |

## 3. The real pilot — what the evidence supports

**Dataset** (`data/real_qwen_pro_600.json`, hash `30c51583d6c8ab07`):
600 automatically graded outcomes, both tiers run on every prompt.

- 200 GSM8K (math, numeric grader) · 200 MMLU (knowledge, choice grader) ·
  200 BIG-Bench Hard (reasoning, choice grader; 4 subsets × 50)
- Pair: `Qwen/Qwen2.5-1.5B-Instruct` → `accounts/fireworks/models/deepseek-v4-pro`
- Cheap-tier success rate 38.2% at `quality_threshold = 0.6`
- Group-aware split 420 / 90 / 90, seed 21

**Test-split result** (`reports/eval_qwen_pro_600/`, n = 90, spent once):

| System | Quality | Cost | Remote% | Unsafe-local% |
| --- | --- | --- | --- | --- |
| all-local | 0.411 | 0.0062 | 0% | 58.9% |
| random @ matched rate | 0.707 | — | 60% | — |
| heuristic | 0.756 | 0.0817 | 68.9% | 15.6% |
| **learned (selected)** | **0.789** | **0.0626** | **60.0%** | **18.9%** |
| all-remote | 0.900 | 0.1085 | 100% | 0% |
| oracle | 0.956 | 0.0653 | 58.9% | 0% |

**Read this before quoting any of it:**

- The learned router beat random at a matched remote rate by **+0.082 quality
  as a point estimate, but the bootstrap CI does NOT clear the random line**
  (`CI-clears-random: False`). This is directionally encouraging and *not* a
  demonstrated win.
- Versus the heuristic: quality **+0.033, CI [−0.056, +0.122]** — includes
  zero. Cost is genuinely lower (CI excludes zero) at a lower remote rate.
- **Unsafe-local rate is worse than the heuristic** (18.9% vs 15.6%): the
  learned router keeps more work local and pays for it in risk. If the FYP
  cares about safety, retrain with the `max_unsafe` policy.
- Test ROC-AUC **0.645** vs validation **0.820** — a real generalization
  drop; n = 90 is small and the gap may be variance, overfitting, or both.
- Per category (learned): math 1.00 @ 96% remote, knowledge 0.688 @ 38%
  remote, reasoning 0.700 @ 50% remote. It has essentially learned "send
  math to Pro" — which is correct here (local math quality is 0.179) but is
  a shallow policy, and the heuristic reaches the same math conclusion.
- Workload mixtures (learned): hard-heavy 0.939, balanced 0.796, easy-heavy
  0.755 — routing pays most when the workload is genuinely mixed/hard.

**Honest summary for the report:** the pipeline is real and the pilot is
real, but at n = 90 the learned router is *not yet* statistically
distinguishable from the heuristic or from random at matched spend. The
correct next move is more data, not better prose.

### Earlier variants, for context

`reports/eval_qwen_pro_{pilot,remote50,maxunsafe10}/` are from the first
120-row collection (hash `dad5032fb9fe2696`, 19-row test split). They are
kept as a record of the threshold-policy exploration; the test splits are far
too small to conclude anything. The `remote50` run is a useful cautionary
example: forcing a 50% remote target made quality **worse than the
heuristic** (−0.21).

### The toy artifact

`artifacts/router.joblib` is trained on **synthetic** data
(`scripts/make_toy_dataset.py`). Its excellent numbers (test ROC-AUC 0.957)
validate the *pipeline only* and must never be presented as ML evidence. The
demo UI detects it (no pair metadata) and labels every result accordingly.

## 4. Cost safety — the rules that must not be relaxed

- `scripts/collect_outcomes.py` refuses to run without **both**
  `--run-paid-calls` and an explicit `--max-api-cost-usd` ceiling, and
  aborts when the projected worst case exceeds it.
- Collection is resumable: it appends to `<out>.work.jsonl` after every task,
  so an interrupted run never re-pays for finished tasks.
- The demo server is mock-only unless started with `--real`; without that
  flag the browser **cannot** opt into spending, and mock results still show
  the hypothetical cost.
- `remote_pair` collection is **two paid calls per task**; `local_remote` is
  one. Verify `TIER_MODE` and the exact model pair before paying.
- `.env` holds a live key. Never commit it, never print it, never let the UI
  read it. `/api/config` exposes only `api_key_present: bool`.

## 5. Invariants a change must not break

1. **Route name ≠ provider.** Billing comes from `Completion.provider`. A
   cheap Fireworks answer is billable and must never be shown as free/local.
2. **Score direction.** Higher confidence / higher `p_cheap_ok` always means
   "the cheap tier is safer here", for every router and every gate statistic.
3. **Pre-route vs post-generation separation.** Pre-router features come from
   the prompt alone. `tests/test_features.py` fails if `routing/features.py`
   ever references a post-generation field.
4. **Artifacts are pair-specific.** A learned artifact trained for one
   cheap/strong pair is invalid for another — the runtime raises, and the UI
   says "router retraining required".
5. **Mode isolation.** `local_only` never calls Fireworks; `remote_only`
   never calls the local backend — including on failure.
6. **The test split is spent once.** Never tune against
   `reports/eval_*/results.json`.
7. **Artifacts are executables.** joblib is pickle-based; load only files
   this repo produced.

## 6. What to do next (in priority order)

1. **More real data.** The single highest-value action. n = 90 test rows
   cannot separate the routers. Generate a new disjoint expansion and define
   its split/decision policy before collecting outcomes. Do not use the old
   local `real_extension_qwen_pro` files as the next evaluation set: they are
   remnants of an interrupted collection and the final 600-row pilot already
   uses the corrected `real_extension_fast_qwen_pro` task set.
2. **Repeat across seeds.** Train/evaluate at several split seeds and report
   the spread, not a single lucky split.
3. **Retrain under `max_unsafe`** and report the safety/cost frontier — the
   current artifact is the `quality_floor` pick and is the riskier choice.
4. **Broaden the categories.** Three families (math/knowledge/reasoning) let
   the router learn "math → strong". Add the summarisation, sentiment, NER,
   and code families from `EVALUATION.md` so the decision is less trivial.
5. **Collect a `remote_pair` dataset** if the two-Fireworks demo is to be
   evaluated rather than just demonstrated — no artifact exists for that
   pair, so it currently runs heuristic-only.
6. **Post-generation gate on real data.** Pilot post-gen AUCs: mean 0.730,
   min 0.673, low_frac 0.630 — `mean` remains the right default; revisit
   with more data before changing `LOCAL_CONF_STAT`.

## 7. Reproducing the pilot numbers

```bash
python3 -m routing.train --dataset data/real_qwen_pro_600.json \
    --out artifacts/router_qwen_pro_600.joblib \
    --report reports/train_qwen_pro_600 --seed 21
```
```bash
python3 -m evaluation.run --dataset data/real_qwen_pro_600.json \
    --artifact artifacts/router_qwen_pro_600.joblib \
    --train-report reports/train_qwen_pro_600 \
    --report reports/eval_qwen_pro_600
```

Same dataset + same seed ⇒ same splits, same model, same threshold. The
artifact records the dataset hash; the evaluator warns on a mismatch.

`data/real_qwen_pro_600.json` is intentionally committed to the public repo
so these numbers can be reproduced. Trained artifacts, other/raw datasets,
work JSONL files and reports are gitignored local products. A fresh clone
must run the training command above before selecting Learned in the UI.

## 8. Known gaps

- No CI. `make test` is manual.
- Summarisation-style categories need an LLM judge; none is wired up, and
  the judge-validation plan in `EVALUATION.md` is unexecuted.
- Local inference is fp32 CPU transformers — slow, and it caps how much real
  data can be collected per hour (ROADMAP workstream 1).
- Cost rates in `config.py` are documented assumptions, not live pricing.
- The demo server serialises requests (one settings singleton, one lock) —
  fine for a presentation, not for concurrent users.
- The browser offers both a single-turn routing demo and SQLite-backed chat
  with bounded multi-turn context and conversation history/search. Streaming
  and stop-generation remain unimplemented; RAG is not required.
- Docker is optional and is not the recommended local-Mac path: Docker
  Desktop cannot expose Apple Metal acceleration to Qwen. Keep it for
  reproducible server/evaluator deployment if needed.
