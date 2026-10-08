# Nana native workspace plan

Updated 8 October 2026. See [handoff.md](../handoff.md) for the current state.

## Implemented

- Water chat GUI, persistent conversations, bounded role-separated context.
- Local/remote/hybrid execution, provider accounting, streaming and stop.
- Ollama serving with configurable model storage and a T7 launcher.
- Notes, tasks, documents, revisions, memory, presets, uploads and gallery.
- Routed agent runs, two-model comparison, run records and reviewed tools.
- MCP stdio/Streamable HTTP clients and isolated browser tooling.
- Google OAuth and Gmail/Calendar operations with private credentials.
- Durable reminders/schedules, accounts, per-user storage and TOTP.

## Finish and verify

1. Complete the owner's Google sign-in and verify Gmail/Calendar workflows.
2. Configure search, image generation and external MCP; run real workflow checks.
3. Verify browser actions, agents, scheduling/recovery and account isolation.
4. Measure Qwen3.5 4B speed/memory/quality and retrain the router for the live pair.
5. Expand rich document editing and AI image editing where required.
6. Simplify fresh installation and keep app/research documentation consistent.

Fixture tests and implemented code are distinct from verified live integrations.
Keep credentials/runtime data out of Git and preserve the water physics work.
