# Demo UI

A small browser front-end for demonstrating the routing agent. It reuses the
existing Python runtime (`main.build_router`, `main.build_backends`,
`main.run_task`, the Fireworks clients, `LocalModel`) — no routing logic is
duplicated in the UI layer.

```bash
make ui        # mock-only demo at http://127.0.0.1:8642
make ui-real   # allow REAL calls — remote requests BILL Fireworks
```

The existing routing demo uses stdlib `http.server`. Its **Open chat** link
opens `/chat`, a second static page backed by SQLite conversations. Chat
history is persisted under `data/nana-chat.db` by default, or at
`NANA_CHAT_DB` if set. Each new request includes the newest complete history
that fits a 32,000-character budget. A successful user/assistant turn is
saved in one database transaction; a failed generation saves neither message.

Chat mode and model settings are enforced by the same execution functions as
the routing demo. **Fully local** never calls Fireworks; **Remote only** never
calls the local backend; **Hybrid** follows the configured tier pair. Chat
starts in preview mode on a mock-only server. `make ui-real` starts chat in
live mode unless **Settings → Preview mode** was previously enabled. The
browser remembers preview and chat-mode choices across refreshes. Real
requests remain blocked by a mock-only server regardless of saved preferences.

An optional FastAPI entry point provides the same chat endpoints with
`make api` on `127.0.0.1:8643`; `make api-real` permits real requests when
the client also sends `"mock": false`. Requests from outside localhost need
`NANA_API_TOKEN` as a Bearer token. The API reads `.env` at startup and does
not expose the Fireworks key in responses.

## Chat interface

Open `/chat` for the Nana chat workspace: a translucent history sidebar,
centered message composer, and a blue/lavender cloud background with a water
surface: moving the mouse leaves a rippling wake and clicking makes a splash.
The ripples come from a small wave-equation simulation on a canvas that
sleeps once the water is calm. The background respects the system's
reduced-motion preference.
On phones, chat history opens as a drawer using the sidebar button.

Type a message to start a saved conversation automatically; **New chat**
returns to the welcome screen. Enter sends, Shift+Enter adds a line, and the
suggestion buttons fill the composer without sending. The header selects
Hybrid, Remote, or Local mode. **Settings** (also the composer plus button)
contains model selection, preview mode, rename, and delete. Preview replies
are labelled; mock-only servers always enforce preview mode. File uploads are not implemented.

## The three modes

| Mode | What runs | Billing |
| --- | --- | --- |
| **Hybrid** | The pre-router (heuristic or learned) picks a cheap or strong tier per prompt; the runtime cascade (post-check, confidence gate, escalation, safe fallbacks) is unchanged. | Depends on the pair (below) |
| **Remote** | Exactly one chosen Fireworks model. Routing bypassed. A failure is reported as an error — the local backend is **never** invoked. | Every request is a billable Fireworks call |
| **Fully local** | Exactly one local model. Routing bypassed. Fireworks is **never** contacted, not even as a fallback. | No API calls |

Chat providers receive separate user/assistant messages, with the newest complete
turns kept within the context limit. Chat replies allow up to 4,096 output tokens
so essays can finish; simple questions still receive short answers. Batch and
routing-demo token limits remain configured independently.

Hybrid defaults to **Local + remote** and supports both pair types from the runtime:

- **Cheap remote + strong remote** (`remote_pair`): Nemotron Lightning 3.5 →
  gpt-oss-120b. **Both tiers are billable** — the UI never describes a
  cheap Fireworks answer as local or free.
- **Local + remote** (`local_remote`): Qwen 2.5 1.5B locally → gpt-oss-120b.
  The cheap tier has no API billing (compute/latency cost only); the
  strong tier bills.

## What the UI shows

Before running, the **Active configuration** panel restates the selection in
plain language, including billing implications and any warnings. After each
run: final model, provider (local vs Fireworks), cheap/strong route, whether
escalation happened, routing confidence vs threshold, estimated API cost,
per-tier token counts, and latency — plus a banner:

- `⚠ Billable Fireworks request (≈ $…)` for real remote work,
- `✓ No API request` for fully-local runs,
- `◌ mock run — a real run like this WOULD be a billable Fireworks request`
  in mock mode (the hypothetical cost is still shown).

## Model selection rules

- Local, cheap-remote, and strong-remote models are selectable from curated
  lists (or free-text `custom…`); defaults match `config.py`.
- When `ALLOWED_MODELS` is configured, every Fireworks selection the mode
  would actually use is validated against it. A disallowed model is a
  visible error — **the demo never silently substitutes another model**.
- The Fireworks API key is never sent to the browser; `/api/config` exposes
  only a boolean `api_key_present`.

## Learned router and model pairs

A learned artifact is tied to the exact cheap/strong pair it was trained on:

- Pair metadata **matches** the selection → green note, learned routing runs.
- Pair metadata **differs** → red **“router retraining required”**; learned
  runs are blocked. The heuristic router remains selectable for the
  untrained pair, and is labelled as such.
- Artifact has **no pair metadata** (the toy artifact trained on synthetic
  data) → amber note; learned routing is allowed in **mock demos only**, and
  every result carries a “TOY artifact — not real ML evidence” warning. In
  real mode it is rejected.

The current local `.env` selects the historical 600-outcome
`router_qwen_pro_600.joblib` artifact, trained on Qwen → DeepSeek V4 Pro.
The current live defaults use gpt-oss-120b instead, so **Learned** correctly
shows a pair-mismatch warning and real learned runs are blocked until a new
artifact is trained. Use **Heuristic** for the live pair.

### Current live-demo recommendation

Use **Hybrid → Local + remote → Heuristic** or **Remote → gpt-oss-120b**.
Both old DeepSeek V4 model IDs returned `404 NOT_FOUND` for this account. The
replacement IDs appeared in the account's read-only inference model list on
2026-10-06, and real gpt-oss-120b replies were verified on 2026-10-08. The existing learned
artifact belongs to the historical Qwen → DeepSeek V4 Pro research pair and
cannot be used for the new live pair without retraining. Remote-only
intentionally has no local fallback.

The routing demo remains single-turn. The linked chat page stores multi-turn
conversations and reuses the same mode-specific execution. Fireworks and Ollama
chat support token streaming and stop-generation; unfinished turns are discarded.

## Safety rails

- The server binds `127.0.0.1` only.
- Without `--real`, every request is forced into mock mode **server-side**;
  the browser cannot opt into spending money.
- UI settings are applied per-request to a snapshot of the process settings
  and restored afterwards. The server reads startup defaults from `.env` but
  never writes it or persists browser changes.
- Remote-only failures never fall back to local; fully-local failures never
  fall back to remote; both are enforced by tests
  (`tests/test_webui.py`).
- Demo runs do not write to `logs/usage.jsonl` (that file is calibration
  data).
- Docker is optional. Direct Python execution is the recommended Mac demo
  path because Docker Desktop cannot expose Apple Metal acceleration to the
  local model.

## Layout

```
webui/service.py       framework-free core: config validation, plain-language
                       describe(), execute() for all three modes
webui/server.py        stdlib http.server wrapper + CLI (--port, --real)
webui/static/index.html  the entire front-end (no build step, no CDN)
webui/static/chat.html   persistent chat page
webui/static/welcome.html  launch page at /welcome: what Nana is, how routing
                         works, and a live mock-only view of the router's decision
chat/                    SQLite storage and chat execution
api.py                   optional FastAPI endpoints
tests/test_webui.py    30 offline tests: mode isolation, billing honesty,
                       pair/artifact handling, HTTP round-trips
tests/test_chat.py      persistent chat, mode isolation, failure handling,
                       FastAPI and stdlib HTTP tests
```
