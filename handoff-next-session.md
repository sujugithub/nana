# Nana: next session handoff

Updated: 9 October 2026, Australia/Sydney.

Read this file first in a fresh session. The latest installation and dropdown
fix are complete. Continue from the user's next request; do not automatically
start the longer product roadmap or repeat downloads.

## Workspace and repository

- Checkout: `/Users/suju/Documents/fyi/nana`.
- Repository: `https://github.com/sujugithub/nana`.
- Current branch: `main`, HEAD `aa19c55` (Aryan's launch page merge).
- Nana 2.0 native changes were merged through PR #3. Aryan's `/welcome` page
  was merged through PR #4. No new commit or push was made in this session.
- Uncommitted changes from the model-dropdown fix:
  `webui/server.py`, `webui/static/chat.html`, `tests/test_chat.py`.
- This handoff is a new, uncommitted file. Preserve the existing changes.
- `handoff.md` has the earlier native-app overview and product roadmap; its
  statement that main is unchanged and its single-model inventory are outdated.

## Running app

- Chat: `http://127.0.0.1:8642/chat`.
- Workspace: `http://127.0.0.1:8642/workspace`.
- Local models: `http://127.0.0.1:8642/workspace#models`.
- Aryan's launch page: `http://127.0.0.1:8642/welcome`.
- The app and dedicated Ollama runtime were running at the last check.
  Verify their availability again in a new session.
- The user's browser last showed a `file://.../webui/static/chat.html` page.
  That standalone HTML page cannot use the HTTP chat API properly. Use the
  running app URL above; this has already been explained to the user.
- Start from the checkout, with the T7 connected:

  ```sh
  .venv/bin/python scripts/start_workspace.py --real
  ```

- The launcher starts the bundled Ollama on `127.0.0.1:11435`, stores weights
  on the T7, and sets `OLLAMA_BASE_URL` for its server process. A separate
  Ollama.app also runs on port 11434; do not confuse these two runtimes.
- A standalone Python invocation of `ollama_client` needs
  `OLLAMA_BASE_URL=http://127.0.0.1:11435` to target Nana's installed models.
- `--real` enables live calls. Local mode uses Ollama; remote mode can bill
  the configured Fireworks account. Preview mode produces mock replies.

## Hardware and models

- Owner: M5 MacBook Air, 24 GB unified memory, 512 GB internal storage.
- Samsung T7 1 TB is mounted at `/Volumes/T7`.
- Model storage: `/Volumes/T7/NanaModels`.
- Bundled runtime: `data/runtime/ollama/ollama`, version 0.40.0.
- Installed and verified via `/api/tags`:
  - `qwen3.5:4b`: 3,973,305,013 bytes; existing fast model.
  - `qwen3.8:27b`: 18,174,721,847 bytes; newly installed at the user's request.
- Qwen3.8 uses the MLX runner and NVFP4 weights. Ollama reports 27.8B
  parameters, completion/vision/tools/thinking capabilities.
- Qwen3.8 was tested locally with a 2,048-token context and thinking disabled.
  It correctly identified as Nana running locally. The cold request took
  58.99 seconds, including 28.73 seconds loading; 25 output tokens took
  3.52 seconds to generate. Loaded model memory was about 18.66 GB.
- A second test through Nana's own model client produced a correct two-sentence
  explanation of SSD storage versus RAM in 7.1 seconds (45 output tokens).
  These are short smoke tests, not a quality benchmark or long-context test.
- Default local selection remains `qwen3.5:4b`. The user can select
  `qwen3.8:27b` per chat or through Workspace → Local models.
- The SSD stores weights; inference still uses the Mac's memory. The 27B model
  leaves limited memory headroom. Do not silently download replacements or
  change macOS memory limits.

## Latest code fix and checks

The new model originally appeared in Workspace → Local models but not in the
chat Settings dropdown, which used a fixed preset list.

- `webui/server.py`: `/api/chat/config` now discovers installed Ollama model
  names using `/api/tags` with a two-second timeout when the configured backend
  is Ollama. It merges/deduplicates those names with the selected model and
  presets. If Ollama is offline, settings retain their fallback list.
- `webui/static/chat.html`: opening either Settings button refreshes the config
  and model choices, preserving the current selection.
- `tests/test_chat.py`: new HTTP regression check covers discovery, duplicate
  removal, preservation of the default, and offline fallback.
- `.venv/bin/python -m unittest tests.test_chat.TestDemoChatHTTP`: 2 tests passed.
- `git diff --check`: passed.
- A headless Playwright check of the running `/chat` page opened Settings in
  local mode and confirmed `qwen3.8:27b` appears. The selected model remained
  `qwen3.5:4b`. The server was restarted to load the Python change.
- The complete test suite was not rerun for this small fix. The earlier clean
  native export passed 176 tests before Aryan's later launch-page changes.

## Conversation context and identity

- Chats persist in SQLite on the owner's Mac. `chat/service.py` sends structured
  role messages, keeping recent complete turns within a 32,000-character budget.
- This character budget is distinct from the model's token/context capacity.
  Do not claim every saved message or the full advertised 256K model context
  fits this 24 GB Mac. A new chat has separate conversation history.
- `generation_context.py` supplies request-scoped chat roles and output limits.
  Normal chat allows up to 4,096 output tokens; batch limits are independent.
- `ollama_client.py` supplies accurate runtime facts: assistant name Nana,
  underlying selected model, local loopback deployment. Identity details should
  appear only when asked, not as repeated introductions.
- Prior issues with false identity claims, essay refusals, lost role structure,
  and repeated introductions were addressed in the merged native changes.

## Hugging Face recommendations already given

Only Qwen3.8 was downloaded. Do not assume the following are installed:

- Huihui Qwen3.5 9B: a community variant modified to reduce refusals, suggested
  as the nearest Ajax-style option:
  `https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated`.
- Hermes 4 14B: steerable chat, writing and tool use; use a 4-bit build:
  `https://huggingface.co/NousResearch/Hermes-4-14B`.
- Gemma 4 12B MLX 4-bit, approximately 6.74 GB weights:
  `https://huggingface.co/mlx-community/gemma-4-12B-it-4bit`.
- Ajax's official page was checked and said Coming soon. It describes a
  modified Qwen3.5 9B trained for Odysseus:
  `https://data.pewdiepie.com/`. Recheck before making a new release claim.

## Boundaries and remaining product work

- The user explicitly does not want the separately copied PewDiePie/Odysseus
  application pushed into Nana. Preserve the ignored local copy, but do not
  add `vendor/odysseus/`, `odysseus_integration/`, its launcher, tests, or related
  third-party files to Git. Do not push the historical local backup branch.
- Do not print or commit `.env`, credentials, OAuth tokens, passwords, chat
  databases, model weights, local runtimes, uploads, logs or virtual environments.
  Provider credentials already exist privately; no keys belong in this handoff.
- Preserve the water-canvas physics GUI previously worked on by Claude.
- Google Gmail/Calendar authorization still needs end-to-end verification.
  Earlier errors included blocked test users and stale OAuth state.
- Search, external MCP/browser workflows, image generation, agent/schedule
  behavior, account isolation and installation packaging need realistic checks.
- The learned research router was not calibrated for the owner's current
  Qwen/Fireworks pair. Do not claim proven routing savings or accuracy.
- The user prefers direct action and concise answers, and asked for this fresh
  handoff because the long conversation became slow. Avoid rereading the full
  history, unnecessary permission questions, or restarting completed work.
