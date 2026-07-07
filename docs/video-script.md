# 🍌 Demo video script (~2:45) — narration + screen directions

Presentation material only — not part of the scored submission. Record the
takes per the scene directions, assemble in iMovie, read the quoted lines as
voiceover. Every number below is real and measured (2026-07-07 runs, see
HANDOFF §3); don't improvise new claims on camera.

Recording setup: Terminal at 18–20pt, dark theme, Cmd+Shift+5 to record the
window. Run `banana --demo` once BEFORE recording to warm caches. Never show
`.env` or the API key on screen.

---

## Scene 1 — The hook (0:00–0:20)
**Screen:** RouteFlow/banana title card, then cut to an empty terminal.

> Every AI agent in this competition pays for tokens. Ours mostly doesn't.
> This is banana — a hybrid routing agent that answers everything it can
> with a small local model running on AMD hardware — which costs ZERO
> tokens — and pays for a frontier model only when a task is genuinely
> hard. The whole competition is one question: can you tell the
> difference? Watch.

## Scene 2 — Live session (0:20–1:20)
**Screen:** run `banana`. Type these, pausing a beat after each answer:

1. `What is the capital of France?`
2. `Classify the sentiment as positive, negative, or neutral: the screen is great but the battery is awful and support never replied`
3. `If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Deduce step by step.`
4. `ily`
5. `exit` (session graph = closing frame)

> One command. The local model loads once — about fifteen seconds — and
> stays warm.
>
> Easy factual question. Green means local — zero tokens, instant, correct.
>
> Sentiment classification — a category small models nail. Still free.
>
> Now a logic puzzle. The router reads the prompt, knows this is beyond a
> small model, and routes it to DeepSeek — the ONLY time we pay. Right
> answer, six hundred tokens instead of guessing wrong for free.
>
> And here's my favorite part. Slang the local model can't handle. It
> drafts an answer, reads its OWN confidence — twenty-seven percent —
> rejects itself, and escalates. **The agent knows when it isn't good
> enough.** That's not a keyword filter; that's the model judging its own
> token probabilities.
>
> End of session: the split, on screen. Everything green was free.

## Scene 3 — Proof at scale (1:20–2:00)
**Screen:** `banana --demo`, let it run to the bar graph.

> All eight scored categories — factual, sentiment, entities,
> summarization, math, logic, and both code tasks. Half route local: zero
> tokens, all correct. Half route remote: the hard half, also correct.
> Bottom line — an all-remote agent would have paid roughly double. And
> accuracy? Twelve out of twelve on our full test battery.

## Scene 4 — Built to survive (2:00–2:25)
**Screen:** `docker logout ghcr.io` then
`docker pull --platform linux/amd64 ghcr.io/sujugithub/hybrid-token-routing-agent:latest`

> This isn't a demo hack — it's a hardened container. We tried to kill it:
> fake API keys, thirty-second deadlines, malformed input. It answers
> anyway — valid results, every time, inside the ten-minute cap. Public
> image, seven gigabytes, model baked in, downloads nothing at scoring
> time — and validated end-to-end on an AMD Instinct MI300X, where local
> answers take a tenth of a second.

## Scene 5 — Close (2:25–2:45)
**Screen:** end card.

> Score equals accuracy first, then fewest tokens. So we made wrong
> answers rare — and paid-for answers rarer. banana, by team banana. Free
> when it can be. Right when it counts.

**End card text:**
`50% of tasks free · 12/12 correct · −58% remote tokens · AMD MI300X validated`
+ repo URL.

---

Delivery: read ~10% slower than feels natural. The bolded Scene-2 line is
the one judges remember — hit it. Overrunning? Cut Scene 3's narration,
never Scene 2. Target < 3:00.
