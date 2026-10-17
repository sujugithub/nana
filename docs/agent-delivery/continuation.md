# Continue Nana harness delivery

Updated 9 October 2026, Australia/Sydney. Resume in `/Users/suju/Documents/fyi/nana`. Read `board.md` and this file; do not repeat the baseline or start a framework migration.

## Current state

- Branch `main`, starting HEAD `aa19c55a8e6a2c0d16cf4320501b1b6e618c9f08`. No commits, staging, pushes, publication or persistent service installation.
- Nana is running with the latest code on `http://127.0.0.1:8642`; dedicated Ollama is on 11435. Both installed models remain on T7. A separate Ollama.app remains on 11434.
- The live launcher was restarted from `.venv/bin/python scripts/start_workspace.py --real` after connecting opt-in Web Chat; tool session at completion: 65863. This is a manually started process, not launchd. Check readiness before assuming it survived an app/session interruption.
- Search provider `exa` is selected for the owner workspace, under the user's instruction to choose a practical provider used by established harnesses. No API key or paid account was configured. Agent/Research uses the shared search adapter. Ordinary Chat browses only when Web is turned on in live Local 4B mode.
- Disposable UI server at 8643 has been stopped; its data remains in `/tmp/nana-ui-acceptance`. No test fixture was written into the user's notes or chats.
- Candidate framework envs remain in `/tmp/nana-engine-spikes/{deep,pydantic}`. Minimal-install env is `/tmp/nana-package-smoke`. These can be rebuilt or individually removed; they are not Nana production dependencies.
- `tests/test_chat.py` and `handoff-next-session.md` remain byte-for-byte unchanged from the protected snapshot. `webui/server.py` and `webui/static/chat.html` retain the original user edits, with installed-model filtering and the new opt-in Web flow added. The water simulation code matches the protected copy. The chat database hash matched before/after the latest controlled restart.

## Verified

- `234` full-suite tests passed in `8.733s`; expected existing ResourceWarning/deprecation messages do not indicate test failures.
- Actual local 4B Web Chat passed both search (two model requests, three Exa source excerpts, 6.370s) and direct public URL reading (1.378s). All chats were disposable. Provider Off/Exa persistence, Web toggle, 27B disable behavior and clickable citations were inspected on a mocked UI server at 8643; that server/tab are now stopped. See `web-connection.md` and its evidence links.
- Chat streaming incident: the latest two empty local chats saved an unavailable `Qwen/Qwen2.5-0.5B-Instruct` model. Dedicated Ollama has only `qwen3.5:4b` and `qwen3.8:27b`; a disposable request to the unavailable ID returned HTTP 404. HTTP errors now retain their status/meaning, and the picker lists only installed tags when online. An actual 4B stream returned content; `/chat` and `/api/chat/config` on the restarted app returned 200. Existing chat preferences were not silently changed; select 4B in those chats before retrying.
- `30` new focused tests pass in a clean environment containing only the pinned requests/certifi workspace dependencies.
- `python3 -S test_harness.py` passes; no paid calls.
- Clean minimal environment and restored live app return HTTP 200 for `/chat`, `/workspace`, `/welcome`.
- Actual 4B public-source research produced one cited note, one correctly addressed email draft, one event draft with valid supplied start/end. Five requests, 9,091 provider tokens, ~21.1 active seconds, no API charge, no send/publish. Evidence in `live-workflow-demo.json` contains only public fixture data.
- 4B baseline tool-selection pack: 27/30. Clarified record filtering schema/description: separate refined pack 30/30, median request 0.981s. Do not claim broad reliability from this small deterministic pack.
- 27B baseline: 29/30 with timeout and heavy swap growth. Do not increase context or repeat large-model benchmarking by default. See README limitations, including unobserved completion after the original driver timeout. The driver now stops on timeout; launcher now enforces single local parallelism/residency.
- UI proof: `skills-ui.png`, `agent-preview-ui.png`, `schedule-ui.png`. Skills survived restart; edits revoked approval; final mocked agent reports preview clearly; daily Australia/Sydney schedule saved in disposable storage.

## Selected architecture and changed boundaries

Retain the existing engine. Native local/remote tool transport and strict fallback meet at `workspace/contracts.py`. `workspace/context.py` owns relevant retrieval, complete exchanges, offload references and chat search. `workspace/effects.py` owns durable intent/results, exact identity, uncertain outcomes and explicit user attestations. `workspace/runs.py` retains orchestration, usage, review and cancellation. Scheduler records durable occurrences; a Mac process lock gives one scheduler owner per workspace.

SQLite changes are additive tables only. Never reset or replace user data to rerun a demo. Agent runs default to local unless a saved preference explicitly overrides it; scheduling UI captures the chosen local model. Ordinary chat's existing 4096-token output allowance remains unchanged.

## Remaining work — do not relabel as done

1. **H07 remainder:** provider-specific long-chat budgeting while preserving full essays. Agent profiles are implemented. Ordinary chat still uses its original complete-turn character budget. Web Chat requests 8K context, with bounded excerpts and estimated character budgeting; audited requests contain 8K, but Ollama's `/api/ps` reports 4K. Two short public fixtures passed; full-window retention is not established. Investigate this discrepancy before claiming long-chat optimization. 27B larger contexts are unproven.
2. **H09 remainder:** explicit fact keys/conflict correction and deletion semantics. Record editing/deletion, metadata, scoping and relevant recall work; automatic semantic conflict handling does not.
3. **H10/H12 live integrations:** Google configured/connected flags are true, but both read-only APIs returned HTTP 403. Need account/API/scope configuration fixed. Exa search is explicitly selected and verified. General MCP has no configured server. Use read-only checks and schema fixtures; do not send fixture drafts.
4. **H17:** current-pair remote/native live verification and routing calibration require an explicit paid budget. Existing mock accounting/router tests do not establish new savings.
5. **MCP:** supported schema vocabulary is deliberately bounded; unsupported keywords fail closed. Extend against the actual server's schema, with tests.
6. Optional persistent launchd installation remains a user decision. The generated plist has not been installed or loaded.

## Commands

```sh
.venv/bin/python scripts/check_readiness.py
.venv/bin/python -m unittest discover -s tests
python3 -S test_harness.py
.venv/bin/python scripts/evals/workflow_demo.py --output /tmp/nana-workflow.json
```

Use the live workflow only if needed by a new change; it was already verified. Avoid repeating passed tests or inference without a reason. Keep one implementation story active; next internal story is H07 remainder. External configuration blockers should not prevent independent local progress.

## Rollback and preservation

- Baseline code copies/manifest/starting patch: `/Users/suju/Documents/fyi/research/nana-baseline-2026-10-09`.
- Delivery-only tracked + new-file patch: `/Users/suju/Documents/fyi/research/nana-delivery-2026-10-09/delivery-only.patch`. It excludes the unchanged protected test/handoff files. Its `webui/server.py` and `webui/static/chat.html` slices start from the protected user-edited copies, including only our subsequent changes. It contains only this delivery's explicit source/evidence inventory, with no `.env`, credentials, chats, user workspace databases, uploads, weights, binaries or private logs.
- Check a reverse patch before applying it, and inspect later edits. Do not blindly reset/stash/clean. Additive tables should remain for inspection; do not drop them to downgrade code.
- A list of the exact tracked/new files included in the patch is saved alongside it in `inventory.json`.

No usage-limit or billing setting was changed. The implementation and evidence are on disk so a session interruption does not require redoing the work.

## Latest increment: Chat + uploads and tools

Read `chat-attachments.md` for exact supported formats, limits and acceptance. Current full suite: 247 tests, 8.862 seconds. Typed/scanned PDF, local vision/follow-up, attachment-derived note, and exact reviewed command restored after reload passed against disposable storage. Current live launcher PID 31049, runtime PID 31051; exec session 80469. The live app's original conversation/message rows remain unchanged; additive Chat tables retain attachment references and idempotent tool-turn finalization. Reload Chat to use the new + menu. Source/evidence are included in the refreshed delivery-only patch.

H07 remains the next internal harness slice. Audio/video and remote vision are not connected. All existing model-executable workspace tools are available in Chat; remaining workspace features open through + shortcuts. Google 403, absent general MCP and paid calibration blocks still apply. No fixture drafts were sent or published.

## Latest design revision

Read `chat-design-update.md`. The big attachment checkbox dialog is gone; + now opens compact icon actions and nested tool/workspace menus. A direct model picker beside Send persists both mode/model and keeps the draft. UI verification used disposable storage; no cloud inference or 27B inference was issued. Port 8642 serves the new static UI on refresh, without a runtime restart. Older `chat-plus-menu-ui.png` depicts the superseded first design; current screenshots are `chat-compact-menu-ui.png` and `chat-composer-model-ui.png`.
