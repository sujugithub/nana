# Nana 2.0 handoff

Updated: 8 October 2026. This branch contains the native Nana application.

## Current state

- Nana chat and workspace run together on localhost port 8642. The existing
  water-canvas GUI is retained.
- Persistent conversations include separate user and assistant messages,
  with newest complete turns kept within a 32,000-character context budget.
- Local, remote and hybrid modes share the existing routing/runtime pipeline.
  Local-only mode does not call the remote provider.
- This Mac uses Ollama on port 11435 with Qwen3.5 4B stored on the T7 at
  `/Volumes/T7/NanaModels`. The model and bundled runtime are local installations,
  excluded from Git. Inference runs on the Mac, not on the SSD itself.
- The local assistant identifies as Nana when asked. Model and hosting details
  stay in the background during ordinary conversation. Chat output allows
  up to 4,096 tokens; batch/demo limits remain independent.
- Fireworks supplies the remote tier. Keys stay in the ignored local `.env`.
- Fireworks and Ollama chat stream tokens and support cancellation. Unfinished
  turns are not saved; partial remote usage may still be billed.
- The native workspace includes notes, tasks, documents/revisions, memories,
  presets, file uploads/gallery, model comparison, agent runs, reviewed tools,
  MCP connections, Google connection setup, reminders, accounts and TOTP.
  Some integrations still require configuration and live verification.

## Start on the owner's Mac

Connect the T7, then run from this checkout:

```sh
.venv/bin/python scripts/start_workspace.py --real
```

Open `http://127.0.0.1:8642/chat` or `http://127.0.0.1:8642/workspace`.
The launcher requires the local bundled Ollama executable under
`data/runtime/ollama/ollama`; it is not included in the repository.
Without `--real`, model replies are mocked.

## Fresh checkout

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Copy `.env.example` to the ignored `.env` and privately configure credentials.
For a separately installed Ollama, start that runtime and set
`LOCAL_BACKEND=ollama`, `LOCAL_MODEL_NAME=qwen3.5:4b` and `OLLAMA_BASE_URL`
to its actual address. Then run `.venv/bin/python -m webui.server --real`.
Download the model explicitly; do not assume the owner's T7 exists on another
machine. Browser tools additionally require
`.venv/bin/python -m playwright install chromium`.

## Validation and limits

- Real local chat, identity responses, and a 1,079-word essay were verified.
  The essay test included an earlier false refusal in its conversation history.
- Real Fireworks replies have been verified. This does not establish routing
  quality, model accuracy or cost savings.
- The clean Nana-only export passed all 176 native tests on 8 October 2026.
- Run the native test suite with
  `.venv/bin/python -m unittest discover -s tests`.
- The historical learned router was trained on Qwen2.5 1.5B and DeepSeek V4 Pro.
  It is not validated for Qwen3.5 4B and gpt-oss-120b. Use heuristic routing
  until new outcomes, calibration and evaluation are available.
- Small local models can produce incorrect facts; runtime instructions improve
  identity behavior but are not a model-quality guarantee.

## Remaining work

1. Verify Gmail/Google Calendar authorization end to end. The previous setup
   encountered blocked-test-user and stale authorization-state errors. Keep
   credentials private and start a fresh authorization from Connections.
2. Configure and live-test search, image generation, external MCP servers and
   browser workflows. Source code and fixture tests alone do not establish that
   these external integrations are working.
3. Validate agent workflows, cancellation, schedules and account isolation
   with realistic tasks. Reminders and scheduled agents require Nana to run.
4. Collect outcomes for the current model pair and compare routing against
   local-only, remote-only and same-cost random baselines. Paid collection
   requires an explicit cost ceiling and authorization.
5. Complete richer document editing and AI image edits as needed. Current native
   document tools and rotate/crop are not full-featured editors.
6. Simplify installation/startup and align the research docs with the live app.

## Files to start with

- `chat/service.py`: bounded history, chat configuration, atomic turn writes.
- `generation_context.py`: request-scoped chat roles, style and output limit.
- `ollama_client.py`: local runtime facts and streaming Ollama transport.
- `webui/service.py`: mode isolation and temporary execution settings.
- `webui/static/chat.html`: chat GUI and water canvas; preserve the physics work.
- `webui/static/workspace.html`, `workspace/`: native workspace UI/services.
- `docs/WORKSPACE.md`: native setup and Google connection instructions.
- `docs/HANDOFF.md`: historical research results and evaluation context.

## Repository boundaries

The separately copied third-party application, its adapter, launcher and tests
are excluded from this branch and its new commit history. Local copies were
preserved outside this branch's tracked files. Do not re-add them accidentally.
Do not commit `.env`, tokens, passwords, chat databases, model weights, local
runtime executables, uploads, logs or virtual environments. Main is unchanged.
