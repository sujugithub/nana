# Archive — competition-era documents

This project began as a 4-day submission to the **AMD Developer Competition
ACT II (Track 1)**, July 2026. It is now a university capstone project. The
documents in this folder are kept as **provenance**, not as current guidance.

They describe a competition that has ended: a scoring harness, an LLM-judge
accuracy gate, a leaderboard ranked by token count, a 10-minute runtime cap, a
10 GB image limit, and a submission deadline of 11 July 2026. None of those
constraints apply any more.

**Do not follow the instructions in these files.** Current documentation lives
at the repository root: `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`,
`EVALUATION.md`, and `CONTRIBUTING.md`.

## Why they are worth keeping

Every measurement in them was real and taken on the date given, and several are
worth citing in the final report:

| File | What it preserves |
| --- | --- |
| `HANDOFF.md` | Verified results as of 2026-07-07: 12/12 correct on the full test battery, −58% remote tokens after the concise-answer prompt, the reasoning behind the `deepseek-v4-pro` choice (the original placeholder model had been retired from Fireworks serverless), the fp32-on-CPU and offline-loading fixes, and validation on an AMD Instinct MI300X. |
| `ISSUES.md` | The task backlog as it stood at the end of the competition, including which items were proven and which were left open. Open technical items were carried forward into `ROADMAP.md`. |
| `video-script.md` | The pitch-video narration, with the measured claims it was allowed to make. |

## One premise that no longer holds

The competition scored **local model tokens as zero**. That was a scoring rule,
not a fact: local inference costs real compute, energy, and time. The
architecture was designed around that rule, and the capstone replaces it with a
real cost model. Read anything in this folder about "free" local tokens with
that in mind — see `EVALUATION.md` for the replacement.
