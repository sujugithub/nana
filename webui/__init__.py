"""Demo web UI for the routing agent.

    python3 -m webui.server            # mock-only demo at http://127.0.0.1:8642
    python3 -m webui.server --real     # allow real model calls (billable!)

Layout:
    service.py  framework-free core: session config, validation, plain-language
                config description, and execution of the three top-level modes
                (hybrid / remote_only / local_only). All logic lives here so
                tests never need a browser or a socket.
    server.py   thin stdlib http.server wrapper + CLI. Binds 127.0.0.1 only.
    static/     single-file UI (no build step, no CDN, works offline).

The UI never sees the Fireworks API key, never writes .env, and applies its
settings per-request against a snapshot of the process settings — nothing is
persisted from the browser.
"""
