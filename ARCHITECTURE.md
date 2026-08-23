# Architecture

How the system is put together, why it is shaped this way, and the sharp edges
worth knowing before you change something.

## The one-line version

`Router.decide` estimates whether the **cheap** tier is adequate → otherwise
the request goes to the **strong** tier → cheap answers pass
`Router.post_check` and, when the cheap backend is local, a logprob confidence
gate → `TokenTracker` records the model, provider, tokens, and estimated cost.

`TIER_MODE=remote_pair` selects Fireworks Flash → Pro.
`TIER_MODE=local_remote` selects local Qwen → Fireworks Pro. The router logic
is shared; only the cheap backend changes.

## The routing pipeline

1. **Pre-route** — one of two interchangeable implementations behind the
   same `Router.decide` surface, selected by `ROUTER_MODE`:
   - **heuristic** (default): `confidence.py` → `router.decide`, described
     below.
   - **learned**: `routing/learned_router.py` loads a trained artifact once
     and predicts a calibrated *P(cheap-tier answer acceptable)* from TF-IDF and
     engineered prompt features; the artifact's own data-selected threshold
     replaces `CONFIDENCE_THRESHOLD`. Trained on observed outcomes of this
     project's model pair — see `docs/LEARNED_ROUTING.md`. `auto` mode uses
     learned when the artifact loads and falls back to heuristic loudly.
   Both emit the same `RoutingDecision`, log `router`/`artifact_version`/
   `p_cheap_ok` (plus legacy `p_local`), and obey the same direction contract:
   higher score = the cheap tier is safer. Features are computable from the
   prompt alone — post-generation signals are structurally excluded
   (`routing/dataset.py` POST_GEN_FIELDS, enforced by tests).

   The heuristic scorer: nine keyword pattern
   groups plus a length ramp produce a 0–1 score. Penalties (math, code,
   code-debug, logic, multi-part, explicit reasoning demands) push toward
   strong; boosts (sentiment, NER, summarisation) push toward cheap, because
   those are long prompts but easy work and need to overcome the length
   penalty. Weighted 40% length / 60% signals, compared against
   `CONFIDENCE_THRESHOLD` (default 0.55). Costs microseconds and no API spend.

2. **Cheap generation.** In `remote_pair`, `CheapRemoteClient` calls Flash.
   In `local_remote`, `local_model.py` applies the Qwen chat template, uses
   deterministic greedy decoding, and counts exact tokenizer tokens.

3. **Local self-assessment** (only `local_remote`). `generate()` keeps the
   per-step logits and computes three statistics of its own answer's token
   probabilities via `compute_transition_scores(..., normalize_logits=True)`:
   the mean (`Completion.confidence`, logged as `local_confidence`), the
   minimum, and the fraction below 0.5 — all from logits the forward pass
   already produced. The gate compares the statistic chosen by
   `LOCAL_CONF_STAT` (default `mean`) against
   `LOGPROB_CONFIDENCE_THRESHOLD` (default 0.4); below it the task
   escalates. All three are logged, and the training report's post-gen AUC
   comparison (`routing/train.py`) is the evidence basis for switching.

4. **Post-check** (`router.post_check`). Pattern-level failure detection:
   empty output, hedging or refusal near the start, prompt echo, degenerate
   repetition (one trigram dominating the output).

5. **Escalation.** A failed cheap answer goes to `StrongRemoteClient`. A
   discarded local attempt costs compute/latency; a discarded Flash attempt
   also costs API tokens. Provider-aware accounting records the difference.

## Design rules that must hold

**Backends never import each other.** `local_model.py`, `remote_client.py`,
and `confidence.py` communicate only through the `Task` and `Completion`
dataclasses in `schemas.py`. This is what makes a backend swappable — replacing
transformers with MLX or llama.cpp should touch `local_model.py` and nothing
else. Breaking this rule is the fastest way to make the project unmaintainable
across six people.

**An answer always beats no answer.** The failure policy in `main.run_task`:
if escalation's strong call fails, keep the flagged cheap answer; if a
strong-routed call fails, fall back to a cheap attempt; any other per-task
error records an error row and the run continues. One bad task must never kill
a batch.

**Config is env-overridable.** Everything tunable lives in `config.py` and can
be set by environment variable, so behaviour can change without editing code.
Prefer adding a knob there over hardcoding a value.

**Route name is not provider.** `cheap`/`strong` are policy tiers. What a
call costs is decided by `Completion.provider` (`local` vs `fireworks`) —
in `remote_pair` the cheap tier is a billable Fireworks call. Never infer
billing, or the words "local" and "free", from the route name.

**Artifacts are pair-specific.** A learned router encodes the cheap/strong
pair it was trained on. Deploying it against another pair is refused
(`LearnedRouter._validate_model_pair`), because a threshold calibrated for
one pair means nothing for another.

**Determinism where it matters.** Greedy decoding locally, `temperature=0`
remotely — so accuracy measurements are reproducible and debugging isn't
chasing sampling noise.

## Known quirks — do not "fix" these blindly

- **Leading hedges escalate even when correct.** An answer that legitimately
  *starts* with a hedge — translating "je ne sais pas" to "I don't know" — is
  flagged and escalated. Cost is one unnecessary remote call, never a lost
  answer. Only special-case it if the real workload makes it common.

- **`test_harness.py` pins its own thresholds.** It sets
  `CONFIDENCE_THRESHOLD=0.55` and `ENABLE_ESCALATION=1` internally so that an
  exported value from a tuning session can't make the suite fail spuriously.
  Its routing assertions are calibrated to those values — don't remove the pin.

- **Mock-mode token counts are fake.** Mock backends return hardcoded strings
  and word-count "tokens". Fine for wiring tests, meaningless for calibration.
  A green mock run proves the plumbing connects and nothing about model quality.

- **The logprob gate detects uncertainty, not wrongness.** A fluent, confident,
  *incorrect* answer passes it — a bat-and-ball trick question scored 0.90
  while being wrong. Treat the 0.4 default as a safety net, not a correctness
  oracle. Calibrating it against graded answers is a `ROADMAP.md` workstream.

- **The mean flatters short answers.** A three-token reply is "confident"
  almost by construction. If that bites, switch the statistic to minimum token
  probability or the fraction below a floor — the plumbing is identical.

- **Local generation is serialised.** In `local_remote`, one model sits behind
  one lock, so local tasks queue. Fireworks calls are thread-pooled.

- **`confidence.py` weights were tuned against a pass/fail accuracy floor.**
  The decisive 0.75 penalties deliberately over-escalate rather than risk a
  wrong answer — a biology question containing the word "function" can be sent
  remote unnecessarily. That trade is intentional; re-tune it against measured
  data rather than intuition.

## Front ends are consumers, never a second implementation

Three surfaces drive the same runtime: batch/CLI (`main.py`), the
interactive terminal (`scripts/banana.py`), and the browser demo
(`webui/`). All three call `build_router()`, `build_backends()` and
`run_task()` — **no routing, cascade or accounting logic may be
reimplemented in a front end.** If a UI needs behaviour the runtime lacks,
add it to the runtime.

The demo UI adds two things of its own, both deliberately outside the
runtime: (a) *per-session* configuration, applied by snapshotting the
`settings` singleton, mutating it under a lock, and restoring it afterwards
— nothing from a browser is ever persisted or written to `.env`; and (b) two
routing-bypass modes, `remote_only` and `local_only`, which construct
exactly one backend so the other is unreachable *by construction* rather
than by a flag check. See `docs/DEMO_UI.md`.

## Where things live

The project root holds the agent modules; `routing/` is the learned
pre-router package (dataset format, splits, training, artifact, runtime
router); `evaluation/` is the offline evaluator; `webui/` is the browser
demo (framework-free `service.py` + stdlib `server.py` + one static page);
`tests/` is the deterministic offline suite for all of it; `scripts/` holds
tooling that is not part of the agent itself (`banana.py`, `calibrate.py`,
`make_toy_dataset.py`, `collect_outcomes.py`); `tasks/` holds task
fixtures; `logs/usage.jsonl` is the append-only audit trail; `data/`,
`artifacts/`, and `reports/` hold learned-routing products. The curated
`data/real_qwen_pro_600.json` evidence set is committed so the pilot is
reproducible; other datasets, trained routers, and reports remain gitignored
working products. See `docs/LEARNED_ROUTING.md`. `docs/HANDOFF.md` is the
current-state takeover note; `docs/history/` is archival provenance and is
not current guidance.

One more sharp edge: **`artifacts/*.joblib` are executables.** joblib is
pickle-based, so loading an artifact runs code embedded in it. Only ever
point `ROUTER_ARTIFACT` at files this repo's own training produced.
