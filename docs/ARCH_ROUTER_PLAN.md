# Plan: Arch-Router, newest-message routing, and learning from users

Status: **proposed**, 14 October 2026. Nothing in this plan is implemented yet.

## Why

A real benchmark on the current pair (local `qwen3.5:4b` → Fireworks
`gpt-oss-120b`, 60 graded GSM8K/MMLU/BBH questions, $0.0075 total spend)
found:

| Policy | Quality | Remote % |
| --- | --- | --- |
| always local | 0.53 | 0% |
| always remote | 0.90 | 100% |
| heuristic router, question text only | 0.67 | 38% |
| random at the same 38% | 0.68 | 38% |
| oracle | 0.90 | 37% |

- The heuristic router performed the same as random once the "Solve this…"
  benchmark instruction (which contains the keyword *solve*) was removed.
- In chat, the router scores the whole rendered conversation. After one
  earlier Python question, "what's a good name for a cat?" scores 0.29 and
  goes remote for the rest of the chat.
- With `LOCAL_BACKEND=ollama` there are no token probabilities, so the
  post-generation confidence gate never fires; the surface post-check caught
  0 of 28 wrong local answers.
- The oracle row is the opportunity: always-remote quality at about a third
  of the remote calls.

This plan has three goals, in order:

1. Try **Arch-Router 1.5B** (Katanemo) as a local, intent-based pre-router
   and keep it only if it beats random at matched spend.
2. Route on the **newest user message**, not the whole transcript.
3. **Log user feedback** (thumbs, regenerate, model switches) as training
   labels.

## About Arch-Router

- 1.5B finetune of Qwen2.5-1.5B. Paper: arXiv 2506.16655.
- Input: a list of routes, each `{"name", "description"}` in plain language,
  plus the conversation. Output: `{"route": "<name>"}`, or `"other"` when no
  route fits. New routes or models need no retraining.
- It classifies **intent** (domain + action). It does **not** predict
  difficulty or answer quality, so route descriptions must encode difficulty
  where it matters, and the evaluation gate below decides whether that is
  enough.
- Community GGUF builds exist (e.g. `tensorblock/katanemo_Arch-Router-1.5B-GGUF`,
  Q4_K_M ≈ 1 GB). Running it in Nana's Ollama is not yet verified.
- **License: Katanemo / DigitalOcean Community License.** Use, modification
  and redistribution are allowed with attribution ("Built with DigitalOcean"
  plus a Notice file). **Commercial use requires a separate licence from
  DigitalOcean.** Fine for the FYP and personal use; if Nana is ever sold,
  switch to the fallback below or obtain a licence.
- **Fallback with the same interface:** Aurelio `semantic-router` (MIT) or an
  embedding classifier trained with the existing `routing/` pipeline.

## Invariants that must still hold (from `docs/HANDOFF.md`)

- Route name ≠ provider; billing comes from `Completion.provider`.
- Pre-route features come from the prompt alone, never post-generation fields.
- `local_only` never calls Fireworks; `remote_only` never calls local, and
  bypasses the pre-router entirely (so it never needs Arch-Router).
- Arch-Router runs locally, so routing never sends text off the Mac.
- An unavailable router never blocks a reply: fall back to the heuristic
  **loudly**, like `ROUTER_MODE=auto` does today.

---

## Sprint 0: spike (1–2 days)

Goal: prove Arch-Router runs well enough on the M5 to be worth evaluating.

- [x] Pull a Q4_K_M GGUF into Nana's bundled Ollama (`127.0.0.1:11435`,
      weights on the T7). Write a Modelfile with the ChatML template if the
      default one does not return clean JSON.
- [x] Use the model card's exact system prompt (`TASK_INSTRUCTION` +
      `FORMAT_PROMPT`); use temperature 0 and a 32-token output cap.
- [x] Measure: routing latency p50/p95 warm and cold, memory, and how often
      the output fails to parse as JSON.
- [x] Memory check with `qwen3.8:27b` loaded (24 GB Mac): router + 4B fits
      easily; router + 27B may not. Decide `keep_alive` for the router.
- [ ] Record the licence decision (FYP/personal use) in `README.md` with the
      attribution and Notice file. **Moved to Sprint 2**: attribution is
      needed once the model ships in Nana, not for a local spike.

**Exit:** p95 routing latency ≤ 300 ms warm, ≥ 99% parseable outputs. If it
fails, stop and evaluate the fallback instead.

### Sprint 0 results (14 October 2026)

Script: `scripts/evals/arch_router_spike.py` (local only, no cost).
Model: official `hf.co/katanemo/Arch-Router-1.5B.gguf:Q4_K_M` (0.99 GB on
disk, 1.1 GB loaded, fully on GPU). Ollama 0.40.0 already applies the correct
ChatML template; no Modelfile is needed.

| Measure | Result | Target |
| --- | --- | --- |
| Parseable outputs | 27/27 | ≥ 99% ✅ |
| Local/remote decision correct (27 labelled probes) | 23/27 | — |
| Same 20 single-turn probes: wrong decisions | 4 (heuristic: 9) | — |
| Multi-turn cases routed on latest intent | 4/4 | — |
| Private-data prompts kept local | 3/3 | — |
| Warm latency, new prompts | p50 339 ms, p95 417 ms | p95 ≤ 300 ms ❌ |
| Cold start | ~1.2 s | — |

Findings that change the build:

1. **Output is Python-dict style**, `{'route': 'coding'}`, not JSON. The
   parser must accept both (`json.loads`, then `ast.literal_eval`).
2. **Prefill the answer** with `{'route': '`. The model then writes only the
   route name: 9 → ~3 output tokens, generation ~200 → ~60 ms.
3. **The remaining cost is re-reading the fixed route list** (~260 ms of
   ~490 prompt tokens per call). Sprint 2 options, each to be checked
   against accuracy on the dev split: shorter route descriptions; a
   cache-friendly prompt order (fixed routes first, conversation last); or
   start local generation in parallel and cancel it if the route is remote.
4. **Blocking issue: `OLLAMA_MAX_LOADED_MODELS=1`** in
   `scripts/start_workspace.py:35`. The router and the chat model evict
   each other: +1.2 s per hybrid turn with `qwen3.5:4b`, and a ~90 s reload
   of `qwen3.8:27b`. Sprint 2 must set it to 2. 4B + router ≈ 5 GB is
   fine; 27B + router ≈ 20 GB on the 24 GB Mac needs a memory check before
   allowing hybrid with 27B.
5. Errors worth targeting with route descriptions in Sprint 1: "Explain why
   the sky is blue" → `reasoning`; "Calculate my mood today" → `maths`;
   "python joke" → `coding`; a Korean proof → `private_data` (wrong, though
   privacy-safe).

**Decision: GO to Sprint 1, with conditions.** Routing quality and parsing
pass; latency misses the 300 ms target by ~120 ms at p95. For context, the
remote model's median reply was 1.4 s in the 60-question run. Fix finding 4
before any integration, and re-measure latency after finding 3.

## Sprint 1: evaluation bench and go/no-go gate (3–4 days)

Goal: a realistic benchmark, and a decision based on data.

- [ ] Build `tasks/route_bench_v1.json`, about 200 prompts written the way a
      user types (no "Solve this…" instruction templates):
  - the 60 graded GSM8K/MMLU/BBH questions with templates stripped and the
    answer format given naturally;
  - chit-chat and greetings; writing and rewriting; code (explain, write,
    debug); multilingual (FR/KO/ES maths and code); fresh-knowledge
    ("latest F1 race"); high-stakes (medical, legal); private data (emails,
    phone numbers, API keys, "my bank statement…");
  - multi-turn cases: a hard technical turn followed by a trivial one, and
    the reverse.
- [ ] Grade open-ended answers with a pairwise LLM judge (`gpt-oss-120b`:
      is the local answer as good as the remote one?). Validate the judge on
      30 hand-labelled items before trusting it.
- [ ] Collect both tiers on every prompt with `scripts/collect_outcomes.py`
      (cost at measured rates: about $0.03 for 200 prompts plus judging).
- [ ] Define the first route set in `config/routes.json`:

  | Route | Description (sketch) | Tier |
  | --- | --- | --- |
  | `chit_chat` | greetings, small talk, simple opinions | local |
  | `simple_writing` | short emails, rewrites, summaries of given text | local |
  | `simple_facts` | well-known facts, definitions, translations | local |
  | `private_data` | messages containing or asking about personal data, credentials, finances, health records | **local, enforced** |
  | `maths` | calculations, word problems, proofs | remote |
  | `coding` | writing, debugging or reviewing code beyond a one-liner | remote |
  | `reasoning` | logic puzzles, multi-step planning, comparisons with constraints | remote |
  | `high_stakes` | medical, legal or financial advice | remote |
  | `fresh_knowledge` | recent events, prices, sports results | remote + web |
  | `other` | anything else | heuristic decides |

- [ ] Add `evaluation/route_bench.py`, generalising the scratchpad analysis
      from the 60-question run. Rows: always-local, always-remote, heuristic,
      Arch-Router, random at Arch-Router's remote rate (200 draws), oracle.
      Columns: quality, remote %, wrong-kept-local %, API cost, latency
      including routing overhead, **private prompts sent remote** (must be 0).
- [ ] Bootstrap confidence intervals, like `evaluation/run.py` already does.

**Go/no-go gate** (spend the test split once, per HANDOFF invariant 6):

- Arch-Router quality is above random-at-matched-rate and the bootstrap CI
  clears random;
- quality ≥ heuristic;
- private prompts sent remote = 0;
- multi-turn cases route on the latest intent.

If it fails: tune route descriptions **on the dev split only** and re-run
once, or move to the fallback. The 2026 study of four open-source routers
(arXiv 2608.14641) found most gains came from *how often* each tier was
picked, not *which* prompts. This gate exists to catch exactly that.

## Sprint 2: newest-message routing and integration (3–4 days)

Can start in parallel with Sprint 1; it also fixes the heuristic.

**Newest-message routing**

- [ ] `chat/service.py`: pass the newest user message and the structured
      `messages` to the router through `Task.metadata`
      (`latest_user_message`, `messages`) instead of relying only on the
      rendered transcript.
- [ ] `router.py` (heuristic): score `latest_user_message` when present.
      The full prompt is still used for generation, and for batch/CLI tasks.
- [ ] Arch-Router receives the last **3 turns** (truncated to ~2,000
      characters) so follow-ups like "now do it in Rust" keep their intent;
      the model card's "latest intent" rule handles topic changes.
- [ ] **Sticky routing:** keep the previous turn's tier when the new message
      routes to `other` or is a short follow-up, so the model doesn't
      flip-flop mid-thread.
- [ ] Regression tests: the cat-name case routes local after a coding turn;
      a hard follow-up after chit-chat routes remote.

**Arch-Router integration**

- [ ] New `routing/arch_router.py`: `ArchRouter.decide(task) ->
      RoutingDecision`, the same surface as `Router` and `LearnedRouter`.
      `signals` carries `{"route": name}`; `confidence` stays in the
      existing direction (1.0 = local route, 0.0 = remote route) so logs and
      the UI keep working.
- [ ] `main.build_router`: add `ROUTER_MODE=arch`. If Ollama or the router
      model is unavailable, or the output is not valid JSON, fall back to the
      heuristic with a stderr warning and `router_kind="heuristic-fallback"`.
- [ ] **Deterministic privacy guard** before any router: regexes for email,
      phone, card numbers and API keys, plus "the turn includes a local
      attachment" → force local. Never rely on a 1.5B model alone for privacy.
- [ ] `webui/service.py` and `chat.html`: show the chosen route name, e.g.
      "coding → gpt-oss-120b", in the message footer next to model and latency.
- [ ] Tests: router-down fallback, invalid JSON, privacy guard, and mode
      isolation (`remote_only` and `local_only` never call Arch-Router).

## Sprint 3: learn from users (4–5 days)

Goal: every real conversation produces labelled routing data, stored only
on the Mac.

**Storage** (`chat/database.py`, additive migrations):

- [ ] `routing_events`: `message_id`, `router_kind`, `route_name`,
      `tier`, `model`, `routing_latency_ms`, `privacy_forced`, `created_at`.
- [ ] `feedback`: `message_id`, `kind` (`thumbs_up` | `thumbs_down` |
      `regenerate` | `switch_model` | `copy`), `from_model`, `to_model`,
      `created_at`.

**UI** (`chat.html`, keep the water canvas and the shared-scope rules):

- [ ] Under each assistant message: thumbs up/down, **Regenerate**, and
      **Try stronger model** (local → remote) or **Try local** (remote → local).
- [ ] The retry replaces the answer in place and keeps the original as
      history, so the pair (local answer, remote answer, user preference)
      is captured.

**Labels** (`scripts/export_feedback.py` → `routing/dataset.py` format):

| Signal | Label for that prompt |
| --- | --- |
| thumbs up on a local answer | local OK |
| thumbs down on local, or "try stronger" then thumbs up | needs remote |
| regenerate on the same tier | weak negative for that tier |
| remote answer, user switches to local and likes it | local OK |
| no feedback | unlabelled (not negative) |

- [ ] Opt-in: the export is off by default. The export can drop message
      text and keep only embeddings for sharing with teammates.
- [ ] Weekly loop: review the labels → adjust route descriptions or train an
      embedding classifier on them → re-run `route_bench` → ship only if the
      gate still passes.

## Sprint 4: later, after the gate passes

- Route among all four models, not two tiers: `qwen3.5:4b`, `qwen3.8:27b`,
  Nemotron Lightning, `gpt-oss-120b`. Apply hard constraints first (vision,
  tools, context length), then a user setting for privacy vs quality vs speed.
- Restore a post-generation check for Ollama: either the runtime's logprobs
  (verify support) or agreement between two local samples.
- Run `route_bench` in CI on every routing change.

## Definition of done

- `ROUTER_MODE=arch` is the default only if the Sprint 1 gate passed, with
  the numbers recorded in `reports/route_bench_v1/`.
- Chat routing uses the newest message, with regression tests.
- Feedback is captured in SQLite and exports to a training dataset.
- `make test` passes; the licence attribution is in place.

## Risks

| Risk | Mitigation |
| --- | --- |
| Intent ≠ difficulty: "coding" includes trivial one-liners | Encode difficulty in route descriptions; the gate measures it |
| Router adds latency to every message | Keep it warm; skip routing in local-only/remote-only modes; p95 budget 300 ms |
| Memory pressure with `qwen3.8:27b` | Q4 router (~1 GB); unload when 27B is active if needed |
| Commercial licence | Fallback router with the same interface |
| Small benchmark → noisy conclusions | Bootstrap CIs; grow the bench from user feedback |
| Feedback bias (people rate only bad answers) | Treat no-feedback as unlabelled; use switch/regenerate signals too |
