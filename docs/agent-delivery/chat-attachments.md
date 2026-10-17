# Chat attachments and workspace tools — 9 October 2026

Loaded on Nana's existing localhost app at port 8642. The composer **+** opens uploads, the entire existing agent-tool registry (plus Read attachments), and shortcuts to every workspace page. Settings remains available in the sidebar. The water-canvas block and protected starting edits are unchanged.

## Behavior

- Four attachments per turn, up to 10 MiB each. Local extraction supports UTF-8 text/code/CSV, text PDFs, PNG/JPEG/WebP/GIF, DOCX, XLSX and PPTX. Spreadsheets use cached cell values; formulas and macros do not execute.
- PDFs have a 40-page and 80,000-character extraction ceiling. Pages without extracted text are rendered locally for vision input, with at most three image pages per turn. Normal Chat admits up to 6,000 document characters. Larger attachments enable Read attachments and expose labeled excerpts plus an owner-scoped chunk reader. The existing cumulative agent budget still applies; this is not unlimited-document analysis.
- Original files, extracted text and normalized images stay in the current user's workspace. Filenames get independent internal IDs, so duplicate display names do not overwrite earlier files. Chat persists attachment references alongside an atomic successful turn. Text evidence and, for compatible local models, images are available in bounded follow-up history.
- Local image input is gated by the selected model's actual Ollama `/api/show` capabilities. Both installed models advertise vision and tools. The 4B path has live acceptance; 27B metadata alone is not a live multimodal performance benchmark.
- Changing to an unsupported model is checked again at send time, before inference. Errors explain the limit and suggest a compatible local model, remote model/service, conversion or splitting, as applicable. There is no automatic escalation, remote upload or model download. Current configured Fireworks choices have no verified image transport; audio/video and remote vision remain unavailable.
- Remote/Hybrid document turns explicitly selected by the user share extracted content with Fireworks and may incur charges. Tool turns require an explicit Local or Remote model; Hybrid tool turns fail clearly instead of guessing a deployment.
- Chat tool turns use the existing durable workspace engine, schemas, owner policy, budgets, receipts and exact-action review. Review, stop, progress, reload/reopen restoration and atomic final-turn persistence are available in Chat. Finalization is idempotent. Failed/cancelled/interrupted runs retain their completed steps in Workspace and do not create a successful chat answer.
- Email and event tools create drafts. Publication remains in the existing reviewed workspace flow. Google 403 and absent general MCP configuration remain concrete integration blockers. Comparisons, scheduling, model management, presets, connections and settings are accessible workspace pages; they are not newly introduced model-executable operations.

Attachment instructions are treated as untrusted source data. Enabling a tool grants the existing tool's scope; it does not weaken shell/MCP ownership or browser review requirements.

## Acceptance

All live fixtures used disposable storage at port 8643 and the already installed local 4B model. No user document, chat, credential, mailbox content or model weight was copied into evidence. No paid request or external publication was made.

| Check | Evidence |
|---|---|
| Typed PDF → exact code 17 | `live-chat-attachments.json`, `chat-pdf-ui.png` |
| Image → Blue square | `live-chat-attachments.json` |
| Scanned PDF → exact code 17; image follow-up → 17 | `live-chat-attachments.json` |
| Attachment → saved note with exact code; final Chat pair | `live-chat-attachments.json`, `chat-tools-ui.json` |
| Command blocked pending exact review; restored on reload/reopen; executed once after approval | `chat-tool-review-ui.png`, `chat-tools-ui.json` |
| Upload through actual + file chooser; attachment chip, reply and clear | `chat-pdf-ui.png` |
| All registry tools and all workspace shortcuts inspected | `chat-plus-menu-ui.png` |
| Isolation, bounds, unsupported inputs/models, Office extraction, follow-ups, atomic failures, review mismatch and finalization de-duplication | 13 tests in `tests/test_chat_attachments.py` |
| Full suite and stdlib routing harness | 247 tests pass in 8.862 seconds; `python3 -S test_harness.py` passes |
| Existing chat rows and water block preserved; deployment pages and capability endpoint HTTP 200 | `chat-attachments-deployment.json` |

H07 remains open: short attachment fixtures passed, but the earlier 8K requested / 4K runtime-reported context discrepancy and full-window retention are not resolved. Keep the 4B default; no additional 27B inference or context scaling was performed.

The attachment reader adds `pypdf 6.19.0`, `Pillow 12.3.0`, and `pypdfium2 5.14.0` to the existing minimal workspace environment. SQLite changes are additive `message_attachments` and `chat_tool_turns` tables; existing conversations/messages are not replaced.

Provider format references: [Ollama vision](https://github.com/ollama/ollama/blob/main/docs/capabilities/vision.mdx), [Fireworks document inlining removal and processing alternatives](https://fireworks.ai/docs/firesearch/inline-multimodal). Nana's configured remote image support is deliberately not inferred from a provider's general multimodal offerings.
