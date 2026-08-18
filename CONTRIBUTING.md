# Contributing

Team workflow for the hybrid routing agent. Read
[ARCHITECTURE.md](ARCHITECTURE.md) first — it explains how the system fits
together and lists the known quirks that look like bugs but aren't.

## Get running in 60 seconds

```bash
python3 test_harness.py          # offline wiring test — zero deps, ~50 ms
python3 main.py --tasks tasks/sample_tasks.json --mock   # mock end-to-end run
```

Both run on stdlib alone (mock mode). You do NOT need torch / a model / an API
key to start hacking on routing logic.

For the real path (a real model + API key):
```bash
pip install -r requirements.txt
export FIREWORKS_API_KEY=fw-...   # never commit this
python3 main.py --tasks tasks/sample_tasks.json
```

## The one rule

**Run `python3 test_harness.py` (or `make test`) before every commit.** It's
the fast offline check that the decide → route → log wiring still works. If it
goes red, fix it before pushing.

## Ground rules

- **Never commit secrets.** `.env` is gitignored; put your key there or export
  it. Only `.env.example` (placeholder) belongs in git.
- **Config knobs are env-overridable** — see `config.py`. Prefer adding a knob
  there over hardcoding. Don't change the default `CONFIDENCE_THRESHOLD`
  without a calibration reason (the harness asserts against 0.55).
- **Keep backends swappable.** `local_model.py` / `remote_client.py` /
  `confidence.py` must not import each other — they talk through the
  `Task` / `Completion` dataclasses in `schemas.py`. This is what lets us swap
  a backend (transformers → MLX, one provider → another) without restructuring.
- **Match the existing style** — the modules are heavily commented on purpose,
  so six people can read each other's code without archaeology. Keep that.

## Branch + PR flow

```bash
git checkout -b your-name/issue-3-docker-smoke-test
# ...work...
python3 test_harness.py          # must pass
git commit -m "…"
git push -u origin HEAD
gh pr create                     # or open a PR in the web UI
```

One branch per issue. Small PRs. Link the issue in the PR description.

## Claiming work

See [ROADMAP.md](ROADMAP.md) for the six workstreams. Put your name against
one before you start so two people don't build the same thing — the workstreams
are deliberately scoped to separate files so they can run in parallel.

Workstreams 1 and 2 (local backend, evaluation harness) are **P0**: they block
everything else, so claim those first.

Read [ARCHITECTURE.md](ARCHITECTURE.md) before your first PR — especially the
"known quirks" section, which lists the things that look like bugs but aren't.
