# Nana: presentation script (under 10 minutes)

Present from `http://127.0.0.1:8642/launch` and scroll between sections.
The numbered slides below also match the backup PowerPoint deck.

| # | Slide | Speaker | Time |
|---|---|---|---|
| 1–3 | Title · The problem · What Nana is | Spandan | 1:15 |
| 4–6 | Live demo · Architecture · Tools | Sujay | 3:00 |
| 7 | Local first and chat history | Bhavin | 1:00 |
| 8 | Three routers | Aryan | 1:00 |
| 9 | Routing results and what's next | Mim | 1:00 |
| 10 | Testing and stewardship | Kezia | 1:00 |
| 11 | Lessons | Sujay | 0:45 |
| 12 | Thank you and questions | Spandan | 0:15 |
| | **Total** | | **9:15** (45 s spare) |

Speaking pace is about 130 words a minute, so each part below fits its time
when read at a normal speed.

---

## Slides 1–3 · Spandan (1:15)

**[Slide 1: "Nana" and "Harness" at the edges]**

> Hi everyone, we're the Nana team. Our project is Nana, an AI assistant that
> runs on your own laptop.

**[Slide 2: "Why pay for every request?"]**

> Why pay cloud API costs for every request? Rewriting a sentence or
> summarising a short note still uses paid tokens when you send it to a
> cloud API. A local model can handle some everyday tasks without those
> charges, but difficult requests can need a stronger model. Nana explores
> that trade-off: what can stay local, and when is cloud access worth the cost?

**[Slide 3: What Nana is]**

> Nana is our approach. It's a harness: the software around the model. You
> can run requests locally, choose a cloud model, or use hybrid routing to
> select a tier. Nana records the model, tokens and cost, so you can see
> what you spent. It also gives the model tools and asks for approval before
> selected actions. Sujay will show you.

---

## Slides 4–6 · Sujay (3:00)

**[Slide 4: the letters fill with the race and the zoom lands on the live game]**

> This is Reef Rally, a 3D kart racing game that Aryan, Kezia and I created
> using the Nana harness, with help from Claude Code. It's running live inside
> our page right now.

*(Click Start race, drive for about 20 seconds, then click outside the game
and scroll down to continue.)*

> Everything, the island, the karts, the music, is generated in code.

**[Slide 5: Architecture]**

> Here's how Nana handles a message. First, a router decides: can the small
> model on the laptop answer this, or does it need the cloud? If it goes local,
> Nana checks the answer for obvious failures, like an empty reply or a
> refusal, and sends it to the cloud model if it looks bad. Every request
> records which model answered, how many tokens it used and what it cost.

**[Slide 6: Tools]**

> In the workspace, Nana works as an agent. It picks a step, calls a tool,
> reads the result and decides again. It has sixteen tools, like searching the
> web, reading files and checking your calendar. Three of them, running
> terminal commands, controlling a browser and connecting to other apps, always
> stop and ask you first. Each run also has hard limits: eight model calls,
> ten cents and three minutes, so it can't run away.

---

## Slide 7 · Bhavin (1:00)

**[Slide 7: Local first]**

> I built the chat backend: the part that remembers your conversations. Every
> chat is saved in a small database on your own Mac, not in the cloud. You can
> search old conversations, rename or delete them, and each chat remembers
> which mode and model you picked. When you send a message, the backend feeds
> the earlier messages back to the model, so Nana remembers what you talked
> about. It all sits behind a small web API that the chat page uses. And in
> local mode, Nana never calls a cloud model at all, not even as a backup.

---

## Slide 8 · Aryan (1:00)

**[Slide 8: Three routers]**

> I built our "How it works" page, which explains routing step by step. The
> router is the heart of Nana. We have three. The heuristic router uses
> keyword and length rules. It's instant and needs no extra model. The
> learned router is a classifier trained on graded answers from both models.
> Arch-Router is a small open model from Katanemo that reads what a message is
> asking for. It's still experimental. The first two are our own. I also
> designed a mock-up of a glass-style chat page for Nana.

---

## Slide 9 · Mim (1:00)

**[Slide 9: the benchmark chart, Quality tab, then the "Sent to cloud" tab]**

> To check whether routing actually helps, we gave sixty graded questions to
> both models. Always using the local model gets 53% right. Always using the
> cloud gets 90%. Our keyword router got 67%, about the same as choosing at
> random.
>
> *(Switch to "Sent to cloud".)*
>
> But look at the perfect router: it gets the full 90% while sending only 37%
> of questions to the cloud. That gap is the opportunity. My part of the
> project is closing it: I'm training a router on our current models so it
> learns which questions the small model can handle.

---

## Slide 10 · Kezia (1:00)

**[Slide 10: Testing and stewardship]**

> I worked on keeping the project clean and reliable. I cleaned up our GitHub
> repository by removing outdated documents from an earlier version, so the documentation matches the system we have now. Nana has 276
> automated tests. They check the things that matter most: that local mode
> never calls the cloud, that costs are reported honestly, and that routing
> behaves as designed.
>
> This also connects to stewardship of resources. Nana uses the computer you
> already own first, and only spends cloud computing and money when a question
> really needs it. Your data stays on your device unless you choose otherwise.

---

## Slide 11 · Sujay (0:45)

**[Slide 11: Lessons]**

> Three lessons. First, measure before you claim. Our first router sounded
> smart but performed like a coin flip, and only testing showed that. Second,
> safety has to be built in: approvals, spending limits and preview mode by
> default. Third, local AI is practical today. A 4-billion-parameter model on a
> laptop handles everyday chat, and the cloud fills the gaps. Next, we're
> training a better router and adding user feedback, so Nana learns which
> model to use.

---

## Slide 12 · Spandan (0:15)

**[Slide 12: Thank you]**

> Thank you for listening. We're happy to take any questions.

*(For questions about code or design, hand over to Sujay. Routing questions go
to Aryan or Mim, the chat backend to Bhavin.)*

---

## Before presenting

- Start the server: `.venv/bin/python scripts/start_workspace.py --real`
- Open `http://127.0.0.1:8642/launch` in Chrome and use the browser’s full-screen option.
- Run through the demo once: the game should already be loaded when slide 4 arrives.
- Mim's line says the router training is in progress. Update it if the work is
  finished, or soften it if it hasn't started by presentation day.
