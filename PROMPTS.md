# Claude Code prompts — paste these to continue the work

Two prompts. Open Claude Code **inside this repo** (`cd` into the project
first), then paste the matching one.

- **Prompt 1 — NOW (before kickoff):** harden the unproven scaffold. Use this
  today. The competition hasn't started, so the real tasks/models are unknown —
  this only makes the plumbing real and reliable.
- **Prompt 2 — KICKOFF DAY:** plug in the revealed tasks + models and tune.
  Use this the moment the benchmark drops.

Each prompt tells Claude to read the repo's own docs first, so it starts with
full context and doesn't re-scaffold what already exists.

---

## PROMPT 1 — pre-kickoff hardening (use now)

```
You're joining an existing competition project (AMD Developer Competition ACT II,
Track 1: Hybrid Token-Efficient Routing Agent). The scaffold is already built
and committed — DO NOT re-scaffold or restructure it.

First, read these three files to get full context:
- HANDOFF.md  (architecture + what's VERIFIED vs UNPROVEN — read this carefully)
- CONTRIBUTING.md  (workflow + the "run test_harness before every commit" rule)
- ISSUES.md  (the task backlog)

Critical fact: the project passes MOCK tests only. Mock mode returns hardcoded
strings — no model, no API. So nothing on the real path has ever actually run.

The competition has NOT started yet, so the real tasks and allowed models are
unknown. Do NOT hardcode any task-specific logic. Your job is only to make the
plumbing real and reliable, using the placeholder model already in config.py.

Work through the [NOW] tasks in ISSUES.md, in priority order:
  #1 Prove the real local model path (transformers load → generate → token count)
  #2 Prove the real Fireworks remote path (needs a FIREWORKS_API_KEY)
  #3 Build and smoke-test the Docker image
  #4 Make ROCm/AMD-GPU the default path (alignment flag — see HANDOFF.md §3)
  #5 Add a threshold-calibration analysis script

Rules:
- One branch per issue; run `python3 test_harness.py` before every commit; it
  must stay green.
- Keep the backends decoupled (they talk through schemas.py — don't make them
  import each other).
- Never commit secrets. .env is gitignored; only .env.example belongs in git.
- If you can't complete a task (e.g. no API key, no AMD GPU handy), say so
  clearly and leave notes in the issue rather than faking it.

When done, report: what's now PROVEN to work for real, what's still blocked and
why, and update the VERIFIED/UNPROVEN sections of HANDOFF.md to match reality.
```

---

## PROMPT 2 — kickoff day (use when the real tasks + models are revealed)

```
The competition (AMD Developer Competition ACT II, Track 1) has started and the
real tasks and allowed models are now known. The routing agent scaffold is
already built, committed, and (per the pre-kickoff pass) proven on the real
model + API paths — DO NOT restructure it.

First read HANDOFF.md (esp. §5 "Kickoff-day playbook" and the §3 alignment
flags) and ISSUES.md (#6 and #7). Then execute the playbook:

1. Set the real models: LOCAL_MODEL_NAME / REMOTE_MODEL_NAME (env or config.py).
   Prefer a pre-quantized local checkpoint for the limited-compute scoring box.
2. Adapt load_tasks() in main.py ONLY if the revealed task format differs from
   {task_id, prompt, metadata}.
3. Decide single-shot vs multi-step (ISSUES.md #6): if the real tasks need
   decomposition or tool use, add a task loop around run_task; otherwise keep
   single-shot. Document the decision.
4. Add a task-specific output validator to router.post_check (ISSUES.md #7) if
   outputs are checkable (exact-match / JSON / numeric) — this is the single
   biggest accuracy upgrade. Extend _SIGNAL_PATTERNS in confidence.py and fill
   FORCE_ROUTE_BY_CATEGORY if tasks are labeled.
5. CALIBRATE THE THRESHOLD (highest-leverage step): run the sample tasks at
   several thresholds (--threshold 0.4 / 0.55 / 0.7), analyze logs/usage.jsonl
   by run_id, and pick the LOWEST threshold that clears the accuracy bar.
6. Final check before submitting: a real end-to-end run on the revealed tasks
   and a clean `make build` (linux/amd64, or the ROCm variant if the scoring
   box has an AMD GPU).

Optimize for the score = billable tokens + accuracy, where local tokens are
FREE. Report the final local/remote split, billable token total, and the
chosen threshold.
```

---

Tip: whoever runs Prompt 1 should commit and push their results so the team
starts kickoff day from a proven scaffold, not an unproven one.
