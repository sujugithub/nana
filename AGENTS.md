# Agent rules for Nana

Durable, project-wide rules for any coding agent (Claude Code, Codex, or
others) working in this repository. Current task progress lives in
`docs/AGENT_HANDOFF.md`; read it after this file.

## Git

- The project owner performs every Git write operation. Agents may run only
  read-only Git commands (`status`, `diff`, `log`, `show`, `branch
  --show-current`, `remote -v`, `ls-files`, `check-ignore`).
- Never stage, commit, push, pull, fetch, switch, merge, rebase, stash, reset,
  restore, clean, tag, or change Git configuration or remotes — directly or
  through a script.
- At a checkpoint, give the owner exact selective staging commands. Never
  recommend `git add .` or force-adding data, reports, artifacts, logs, or
  model directories.

## Approval boundaries

Explicit owner approval is required before:

- any paid API call, Fireworks request, or remote model judging;
- sending prompts or datasets to an external service;
- downloading models or external datasets;
- collecting fresh outcomes from real models;
- long or resource-intensive jobs;
- running the reserved final held-out evaluation;
- replacing the deployed router artifact;
- deleting or overwriting datasets, reports, or artifacts.

A spending request must state the exact command, providers/models, data sent,
expected call count, runtime, cost assumptions, budget ceiling, output path,
resume behaviour, and stopping conditions. Do not label an estimate "worst
case" unless it covers input, output, task lengths, retries, and every
additional paid attempt, with provider pricing verified.

## Evidence and honesty

- Never invent model results, grades, costs, latency, or test outcomes.
- Synthetic or mock data is for software tests and pipeline validation only
  and must never be presented as evaluation evidence.
- Report infrastructure failures separately from answer correctness, but
  include them in reliability reporting. State the denominator of every
  quality metric.
- Report negative results plainly. Do not weaken tests to make them pass.
- Do not claim latency improvements without live measurement on the target
  machine.
- Mark unverified deployment facts (model identity, quantisation, context
  length) as pending until evidence is provided.

## Router artifacts

- A learned-router artifact is valid only for the model pair and generation
  configuration recorded with it. Never reuse thresholds across pairs.
- Joblib artifacts execute code on load; load only artifacts produced by this
  project's own pipeline.
- Synthetic/toy artifacts are allowed only in explicit mock or test use.

## Secrets and private data

- Never print, copy, or commit `.env` values, API keys, OAuth tokens,
  passwords, chat databases, uploads, logs, model weights, local runtimes,
  `data/*` (except the curated `data/real_qwen_pro_600.json`), `artifacts/`,
  `reports/`, or virtual environments.
- Do not read private conversations or uploads unless specifically needed and
  authorised.
- Handoff and documentation files contain facts only: no secrets, private
  prompts, private model outputs, or absolute personal paths.

## Code changes

- Preserve existing features and unrelated work. No unrelated refactors.
- Keep changes small, reviewable, and covered by tests. Reuse the existing
  architecture unless a demonstrated blocker requires a change.
- Preserve the chat water-canvas GUI.
- Do not re-add the excluded Odysseus files listed in `.gitignore`.

## Environment notes

- Run the offline checks before every checkpoint:
  `python test_harness.py` and `python -m unittest discover -s tests`
  (use the project `.venv` interpreter).
- On Windows, set `PYTHONUTF8=1`: several data-pipeline `open()` calls rely on
  the platform default encoding. Some workspace tests are POSIX-only; see
  `docs/AGENT_HANDOFF.md` for the current known-failure baseline.
- Git for Windows may check files out with CRLF line endings. Compare file
  fingerprints using committed content (`git cat-file -p HEAD:<path> |
  sha256sum`) or parsed-content hashes, not raw working-tree bytes.
