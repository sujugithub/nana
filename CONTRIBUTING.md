# Contributing

Team workflow for the hybrid routing agent. Read [HANDOFF.md](HANDOFF.md)
first — it has the full architecture and the verified-vs-unproven status.

## Get running in 60 seconds

```bash
python3 test_harness.py          # offline wiring test — zero deps, ~50 ms
python3 main.py --tasks tasks/sample_tasks.json --mock   # mock end-to-end run
```

Both run on stdlib alone (mock mode). You do NOT need torch / a model / an API
key to start hacking on routing logic.

For the real path (needed for issues #1–#3):
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
  the real model in on kickoff day without restructuring.
- **Match the existing style** — the modules are heavily commented on purpose
  (this gets judged and debugged live). Keep that.

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

See [ISSUES.md](ISSUES.md) for the task list. Assign yourself the GitHub issue
(or drop a comment) before starting so two people don't do #1 twice. The
issues are ordered by priority and marked **NOW** (doable today) vs
**KICKOFF** (blocked until the real tasks/models are revealed).
