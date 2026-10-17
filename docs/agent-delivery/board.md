# Nana delivery board

Updated 9 October 2026. Baseline: main `aa19c55`. This bounded increment is verified. No implementation story is currently active; remaining story slices are explicit below. No commits, paid calls, weight downloads, or persistent service installation.

| Story | Status | Acceptance evidence / remaining work |
|---|---|---|
| H01 protect baseline | Done | 189 original tests and mocked harness passed; protected originals, SHA-256 manifest and patch in `/Users/suju/Documents/fyi/research/nana-baseline-2026-10-09` |
| H02 evaluation pack | Done | Identical three-engine fixture; 30 new focused behavioral tests; 10 local tool-selection tasks × 3 repeats × 2 models, all outcomes retained |
| H03 engine decision | Done | `engine-decision.md`: isolated pinned candidates work; retain Nana boundary and implement selected patterns without framework migration |
| H04 structured tools | Done for local/remote adapters | Schema validation, normalized results, native catalogs/calls, bounded JSON fallback; actual 4B agent saved one note and finished; remote adapter needs paid live verification |
| H05 cumulative budgets | Done for this increment | Persistent request/token/cost/time usage, bounded validation retries, reserved remote allowance; interrupted unknown remote billing blocks resume |
| H06 progress/cancel | Done for this increment | Existing cancel tests pass; running shell cancellation tested; UI progress and usage; blocking third-party reads end at their timeout |
| H07 context | Partial | Native roles/call IDs, completion reserve, whole exchanges, overflow references; improve compacted receipts and chat profile budgeting |
| H08 retrieval/skills | Done | Relevant bounded memories, lazy approved skills, approval revoked on edit; user-scoped attributed complete-turn chat search tested |
| H09 controlled memory | Partial | User/scope isolation, provenance, revisions, categories/confidence and edit/delete correction workflow; automatic conflict/key resolution remains open |
| H10 integrations | Search connected; Google/MCP blocked | Exa keyless search verified and selected; explicit Web Chat works for local 4B with source links. Google Gmail/Calendar still HTTP 403; general MCP server absent. See `web-connection.md` |
| H11 effect policy | Done for this increment | Exact action-ID review, version-bound draft publication, write de-duplication, durable uncertain outcomes/user reconciliation; label/delete routes share operation records |
| H12 research → draft | Local live demo passed / Google blocked | Actual 4B read public source → cited note → correctly addressed email draft + timezone-aware event draft; five requests, ~21.1s, $0. Google 403 blocks provider acceptance |
| H13 occurrences | Done | Dispatch != completion, failure propagation, two-dispatcher claim, no overlap, anchored interval coalescing, Sydney DST gap/fold tests |
| H14 restart | Done for this increment | Atomic scheduled run/occurrence creation; queued recovery; in-flight work interrupted; no blind effects replay; Mac scheduler process lock and explicit reconciliation |
| H15 Mac service | Prepared | Readiness passes; absent-SSD path fails clearly; optional plist validates with plutil; not installed. Launcher limits local parallelism/residency to one |
| H16 local profiles | Done for bounded profiles | 4B 4096 baseline 27/30; refined catalog 30/30, median 0.981s; 27B 2048: 29/30, warm median 4.775s; one timeout, large swap growth. Keep 4B default; larger 27B contexts unsupported by evidence |
| H17 routing/cost | Partly implemented / calibration blocked | Per-step provider/route/tokens/timing/cost retained; no current-pair paid data without explicit budget; no savings claims |
| H18 package/rollback | Done for this increment | 247 tests pass, 30 original focused tests in clean minimal env, stdlib harness passes; all pages 200; user baseline changes and chats preserved; UI screenshots, rollback patch and continuation recorded |

## Chat streaming incident resolved

The two latest empty local chats selected `Qwen/Qwen2.5-0.5B-Instruct`, which is absent from the dedicated Ollama runtime. A disposable request to that model returned HTTP 404. The streaming client caught HTTP errors as generic connection failures. Chat now offers only installed Ollama tags while the runtime is available, chooses an installed default if a saved preference is unavailable, and reports an uninstalled model directly for existing chats. The 4B live streaming smoke succeeded; the restarted app serves `/chat` and lists only the installed 4B and 27B models. The three focused regressions and full 222-test suite pass. No conversation setting or content was rewritten.

## Observed blockers and limits

- Google read-only endpoints: HTTP 403 on both Gmail and Calendar. Provider account/API/scope configuration needs resolution; token presence is not proof of usable access.
- Search: Exa is explicitly selected and verified after the user authorised provider choice. Hosted queries leave the Mac; Web Chat is opt-in and excludes earlier chat history from query preparation. General MCP: zero configured servers.
- Web Chat requested 8K context, while Ollama reported 4K after the two successful short fixtures. Long-window retention remains unproven; keep H07 open and the original essay allowance intact.
- No approved paid calibration budget. Mock accounting is not a current-pair routing benchmark.
- The 27B trials coincided with swap growth from 2.78 GB to 12.78 GB. Ollama's reported allocated VRAM grew beyond physical RAM; this is a runtime-reported allocation, not a trustworthy physical resident-memory measurement. The dedicated app/runtime later became unavailable; cause unproven. Both were restored using the existing launcher and SSD weights. No further 27B context scaling is justified in this session.
- Native final-text handling initially failed an actual agent smoke and caused repeated note requests. Fixed the adapter distinction and de-duplicated identical local writes. Keep the failed evidence as well as the successful result.

Evidence lives beside this board. Tests use temporary databases and no live messages, credentials or uploads. UI acceptance uses port 8643 and disposable data; the user's original service uses 8642. Do not describe partial stories as release-complete.

## Next work, in dependency order

1. H07 remainder: preserve the existing essay allowance while measuring a larger 4B chat context and long-conversation retention. Agent profiles are already bounded; do not silently lower chat output limits.
2. H09 remainder: add explicit fact keys and reviewable conflict handling, with deletion/tombstone semantics tested before automatic recall changes.
3. H10/H12: resolve Google 403 and obtain general MCP server configuration, then run read-only integration checks. Exa search is already verified. Do not send fixture drafts.
4. H17: obtain an explicit paid calibration ceiling before any current-pair Fireworks collection. Until then, keep savings unclaimed.
5. Revisit general MCP schema vocabulary against a concrete configured server. No auto-start installation without a separate user choice.

No code was committed or staged. See `continuation.md` for exact resume and rollback instructions.

## User-requested Chat attachments/tools increment

Done within the documented limits. + now uploads supported documents/images, checks the selected model, exposes all existing agent tools and links every workspace page. Chat uses durable workspace runs with exact reviews, cancellation, reload restoration and idempotent atomic finalization. Thirteen new regressions and the full 247-test suite pass. Actual local 4B read typed/scanned PDFs, identified a blue square, retained an image for a follow-up, saved an attachment-derived note, and ran one harmless command only after its restored exact review was approved. See `chat-attachments.md` and the public fixture evidence.

Remaining limits: audio/video and configured remote vision are absent; larger documents use bounded chunks and existing run budgets. Google/MCP and H07/H09/H17 blockers above remain open. Current app: launcher PID 31049, dedicated runtime PID 31051, port 8642/11435. No paid request or model download was made.

## User design feedback: compact menus and direct model switching

Done: replaced the large checkbox dialog with a compact anchored icon menu, grouped tools and workspace navigation. Composer model picker offers installed local/configured cloud models and Auto. Existing-chat settings persistence, draft preservation, upload and selected-model preview dispatch, keyboard selection and submenu dismissal were verified in disposable storage. Thirteen attachment regressions and JS syntax pass; backend scope remains unchanged. See `chat-design-update.md`. No restart, paid generation or user-data edits were needed.

## Generated code and document downloads — 11 October 2026

Done: code fences have Copy/Download actions; explicit Word requests package
the completed text into an actual DOCX. Ordinary answers stay in Chat.
Downloads are scoped to account/conversation/assistant message, survive
reopen and require no model call. Generated HTML remains inert in the UI.
The renderer/styles are separate static files, and the native and FastAPI
servers share export logic. No existing chat data/schema or canvas edits.

Acceptance: fourteen export regressions; all 261 tests and stdlib harness pass;
real local 4B HTML and DOCX generations on disposable port 8643. Browser copy
and paste, downloads and chat reopen succeeded. HTML bytes match the saved
code; both the sample and actual downloaded DOCX rendered cleanly. Screenshots:
`chat-code-download-ui.png`, `chat-docx-download-ui.png`. Details and limits:
`chat-generated-files.json` and root `handoff.md`.

No paid calls, remote inference, model downloads or autostart installation.
The earlier 9 October process IDs are historical snapshots. Current readiness
and chat preservation were checked after restarting the existing 8642 app.
All previously listed unfinished stories remain open.
