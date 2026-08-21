# Demo UI

A small browser front-end for demonstrating the routing agent. It reuses the
existing Python runtime (`main.build_router`, `main.build_backends`,
`main.run_task`, the Fireworks clients, `LocalModel`) — no routing logic is
duplicated in the UI layer.

```bash
make ui        # mock-only demo at http://127.0.0.1:8642
make ui-real   # allow REAL calls — remote requests BILL Fireworks
```

No new dependencies: the server is stdlib `http.server`, the page is one
static HTML file, and everything works offline in mock mode.

## The three modes

| Mode | What runs | Billing |
| --- | --- | --- |
| **Hybrid** | The pre-router (heuristic or learned) picks a cheap or strong tier per prompt; the runtime cascade (post-check, confidence gate, escalation, safe fallbacks) is unchanged. | Depends on the pair (below) |
| **Remote** | Exactly one chosen Fireworks model. Routing bypassed. A failure is reported as an error — the local backend is **never** invoked. | Every request is a billable Fireworks call |
| **Fully local** | Exactly one local model. Routing bypassed. Fireworks is **never** contacted, not even as a fallback. | No API calls |

Hybrid supports both pair types from the runtime:

- **Cheap remote + strong remote** (`remote_pair`): e.g. DeepSeek V4 Flash →
  DeepSeek V4 Pro. **Both tiers are billable** — the UI never describes a
  cheap Fireworks answer as local or free.
- **Local + remote** (`local_remote`): e.g. Qwen 2.5 1.5B locally → DeepSeek
  V4 Pro. The cheap tier has no API billing (compute/latency cost only); the
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

The current local `.env` selects the real 600-outcome
`router_qwen_pro_600.joblib` artifact. The default Hybrid pair is therefore
local Qwen 2.5 1.5B → remote DeepSeek V4 Pro, which shows a green exact-pair
note when **Learned** is selected. The two-Fireworks pair remains available
with the heuristic until a separate artifact is collected for that pair.

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

## Layout

```
webui/service.py       framework-free core: config validation, plain-language
                       describe(), execute() for all three modes
webui/server.py        stdlib http.server wrapper + CLI (--port, --real)
webui/static/index.html  the entire front-end (no build step, no CDN)
tests/test_webui.py    29 offline tests: mode isolation, billing honesty,
                       pair/artifact handling, HTTP round-trips
```
