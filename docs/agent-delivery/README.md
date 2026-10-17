# Nana harness delivery — 9–17 October 2026

This is an incremental upgrade of Nana's existing engine. The water-canvas UI, existing chats, installed models, hybrid routing, and the user's uncommitted dropdown changes are preserved. See [the board](board.md) for completed versus unfinished scope and [the engine decision](engine-decision.md) for the isolated framework comparison.

## Working increments

- **Contracts and execution:** strict schemas before any tool call; explicit success/error envelopes; native Ollama and Fireworks tool catalogs/calls; strict JSON fallback for hybrid/other backends; native ordinary final text remains valid. Malformed fallback decisions stop after three errors. Identical local writes in one run reuse a receipt instead of making duplicate records.
- **Budgets and progress:** persisted model attempts, token/cost totals and active seconds; eight attempts, 20,000 tokens, $0.10 estimated cost and 180 active seconds by default. Remote dispatch reserves input/output allowance first. Cancellation stops follow-up actions, streamed generation and controllable shell process groups. Network reads may wait until a chunk or their bounded timeout; cancelling never implies an external effect was undone. Interrupted remote usage that is unknown cannot silently resume.
- **Context and memory:** complete native assistant/tool pairs and matching call IDs; whole-exchange compaction; receipts retain item IDs/source URLs/file references. Large tool output stays in per-user files with a hash and bounded preview. Relevant memory is ranked rather than dumping the newest 30 entries. Memories expose scope, category, confidence, provenance and revisions. Saved-chat search returns attributed complete exchanges in the current account. Skills load only after user approval; editing content revokes that approval.
- **Effects:** review is bound to a persisted action ID; publication to the reviewed draft version. Intent is saved before execution. Uncertain actions never auto-replay. The action inspector lets a user record what they verified; these records are labeled user attestations, not provider receipts. Email/calendar/label/delete paths use durable operation records. Host shell/MCP remains owner-only; a working directory is not a sandbox.
- **Schedules:** occurrence history is separate from the run. Dispatch is not completion. Claim, run creation and linkage are atomic. No overlapping occurrences; intervals preserve their anchor and coalesce missed times. Daily recurrence follows the saved IANA timezone; DST gaps move forward and repeated wall times use the first occurrence. Failed/interrupted repeating jobs pause visibly. Queued scheduled reads can recover; in-flight generations remain interrupted. A process lock prevents two schedulers owning the same Mac workspace.
- **Mac operation:** readiness checks for T7, dedicated runtime and installed tags; launcher uses one parallel local inference and one loaded model. Optional launchd file generation is provided, without installing or enabling it. New agent tasks default to local unless a saved preference explicitly says otherwise; an explicit model choice remains respected.

## Acceptance evidence

- Original baseline: **189 tests** passed; starting file hashes are in [preservation-check.json](preservation-check.json). The user's starting edits were retained. Later server/Chat changes add installed-model filtering and opt-in Web access; water physics and the original dropdown behavior are preserved.
- Current suite: **261 tests** passed again on 17 October; **30 original focused tests** also passed in a fresh minimal environment during the initial delivery. `python3 -S test_harness.py` passed without site packages.
- **Generated files:** saved code replies support Copy and Download, and explicit Word output requests produce local DOCX downloads. Fourteen export regressions cover content, ownership and persistence. Actual local-model downloads and rendered Word output were checked on 11 October. [Acceptance evidence and limits](chat-generated-files.json).
- **Chat design:** compact + menu with grouped icon actions, nested connected/advanced tools and a model picker beside Send. Local/cloud choices save to the chat while preserving its draft. [Design and UI acceptance](chat-design-update.md).
- **Chat uploads/tools:** the + menu supports local PDF/image/Office/text extraction, capability checks, owner-scoped attachment history, all existing agent tools, exact review and reload restoration. Actual 4B typed/scanned PDF, vision, follow-up, note and reviewed-command fixtures passed. [Scope, evidence and remaining limits](chat-attachments.md).
- **Web Chat:** opt-in public URL reading and hosted Exa search for local 4B, with source links, bounded excerpts, query-history isolation and no tools during final answering. Actual search and URL fixtures passed; mocked UI flow was inspected. [Connection decision, evidence and limits](web-connection.md).
- Chat streaming incident: a selected model that was not installed returned Ollama HTTP 404, previously reported as a generic connection failure. The picker now lists installed models, existing unavailable selections receive a clear error, and an actual 4B streaming reply succeeded. The live app was restarted and serves `/chat` with the corrected choices.
- A clean minimal workspace environment served `/chat`, `/workspace`, `/welcome` with HTTP 200.
- UI: skill save/approval, persisted state after restart, and approval revocation after editing were exercised on disposable storage at port 8643. A mocked agent completed with visible state/usage, and a daily Australia/Sydney reminder was saved in disposable storage. [Skill evidence](skills-ui.png), [run evidence](agent-preview-ui.png), [schedule evidence](schedule-ui.png).
- **Actual 4B workflow:** public source read → cited note → email draft with the fixture recipient → calendar draft with timezone-aware start/end → final answer. Five model requests, 9,091 provider-reported tokens, approximately 21.1 active seconds, $0 API cost, zero external publications. All storage was disposable. [Full fixture evidence](live-workflow-demo.json).
- A smaller actual 4B note workflow completed in two requests and saved exactly one note. [Evidence](live-agent-smoke.json). Its earlier failure is retained in `live-agent-smoke-before-fix.json`; that failure drove the native-final-text and repeated-write fixes.
- Google tokens were present, but both read-only Gmail and Calendar checks returned HTTP 403. No mailbox contents were copied into artifacts. Public HTTP reading and isolated browser reading succeeded. MCP handshake/schema validation is covered by disposable fixtures, not a configured live server.

## Local measurements and limits

[Raw baseline trials](local-profiles.json), [summary](profile-summary.json). Runtime: Ollama 0.40.0; Python 3.14.8. Ten distinct single-tool tasks, three repeats per model, temperature 0, seed 7, 128 output tokens. Tools were selected/validated without executing effects.

| Profile | Correct selections | Median latency after first request | Median generated tokens/s |
|---|---:|---:|---:|
| Qwen 3.5 4B, 4,096 context | 27/30 | 1.036 s | 30.51 |
| Qwen 3.8 27B, 2,048 context | 29/30 | 4.775 s | 9.41 |

The 4B failures were note-filtering cases; the tool description/schema was subsequently clarified. A [separate refinement trial](local-4b-refined.json) passed **30/30** with a 0.981-second median request latency. Do not combine pre/post results or call this a broad agent reliability score.

The 27B first request timed out. Requests were issued sequentially, but the old trial driver continued after that timeout: server-side completion of that first request was unobserved, so strict non-overlap cannot be certified for that portion. The driver now stops a profile on transport timeout. Subsequent starts also explicitly limit Ollama parallelism and model residency to one.

System swap grew from approximately 2.78 GB to 12.78 GB during the baseline sequence. Ollama's reported allocation peaked around 26.96 GB, exceeding this Mac's physical RAM; it is an allocation report, not a measured resident footprint. Other apps and runtime behavior are confounders. The original app/runtime became unavailable later; the cause is unproven. Both were restored from the existing launcher/SSD installation. **Keep 4B as the default; do not increase 27B context based on these trials.** No model weights were downloaded and no macOS memory settings were changed.

## Run and verify

From the repository root:

```sh
.venv/bin/python -m unittest discover -s tests
python3 -S test_harness.py
.venv/bin/python scripts/check_readiness.py
.venv/bin/python scripts/evals/workflow_demo.py --output /tmp/nana-workflow.json
OLLAMA_BASE_URL=http://127.0.0.1:11435 .venv/bin/python scripts/evals/workflow_demo.py --live-local --output /tmp/nana-live-workflow.json
```

The workflow demo always uses temporary workspace storage. The live option reads only a public fixture URL and calls the installed local 4B model. It cannot send mail or publish events.

Minimal Ollama workspace install (full training/transformers requirements remain separate):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-workspace.txt
.venv/bin/python scripts/start_workspace.py
```

The launcher above starts preview mode. `--real` enables live local and potentially billable remote requests; actual remote use still requires its configured credentials. Do not launch a second scheduler over the same workspace. Browser tooling is optional: the verified installed version was Playwright 1.63.0; it is not needed for the minimal smoke checks.

To **prepare**, not install, an optional service:

```sh
.venv/bin/python scripts/prepare_launchd.py --output /tmp/local.nana.workspace.plist
plutil -lint /tmp/local.nana.workspace.plist
```

The generated service uses preview mode with RunAtLoad/KeepAlive off. The logs directory must exist before any future manual service start. Persistent installation/auto-start remains the user's choice. Laptop sleep is not continuous execution; on wake, missed due times coalesce rather than burst-replaying all occurrences.

## Storage and rollback

New SQLite tables (`actions`, `occurrences`, `message_attachments`, `chat_tool_turns`) are additive; no conversation or record tables are replaced. Versioned records retain revisions. Take a private SQLite backup with the application stopped before a future deployment; do not copy private databases into Git or these evidence artifacts.

The initial 9 October delivery was left uncommitted. On 17 October the owner requested committing the accumulated native changes and pushing them to GitHub `main`. The saved delivery-only patch covers the initial increment, not every later Chat change; use Git history to review the published scope. Do not run a blanket reset. The user's protected baseline copies and original patch are outside the repo in `/Users/suju/Documents/fyi/research/nana-baseline-2026-10-09`. Older code ignores the new tables, but cannot interpret all new receipt/status semantics: retain the data for inspection rather than dropping it. Remove optional generated files individually only after checking the handoff inventory. No launchd service was installed.

## Unfinished acceptance

- Resolve Google's HTTP 403 and supply a general MCP server for live integration acceptance. Exa search is selected and verified; credentials alone do not prove Google access.
- Live Fireworks native tools and current-pair routing calibration need an explicit paid budget. Mock adapter tests are not paid-provider verification or savings evidence.
- Agent contexts have model profiles and labeled estimates. Ordinary chat keeps its established complete-turn character budget and 4,096-token essay allowance; provider-specific long-chat tuning remains open to preserve behavior.
- Memory corrections/edit/delete work on explicit records; automatic semantic conflict resolution is not implemented. Project scope is a retrieval key within the account, not a separate authorization boundary.
- Arbitrary MCP JSON Schema vocabulary is not supported: unsupported keywords fail closed. Validate any concrete configured server before enabling its tool.
- No exactly-once external-effect guarantee, background execution through laptop sleep, broad end-to-end model reliability claim, or 27B larger-context support is claimed.
