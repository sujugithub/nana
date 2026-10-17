"""Sprint 0 spike for docs/ARCH_ROUTER_PLAN.md: is Arch-Router 1.5B usable in Nana?

Measures cold and warm routing latency, JSON parse rate, route accuracy on a
small labelled probe set, and memory, against Nana's bundled Ollama. Local
only: no Fireworks calls, no cost. This is a feasibility check, not the
Sprint 1 quality gate.

    OLLAMA_BASE_URL=http://127.0.0.1:11435 .venv/bin/python scripts/evals/arch_router_spike.py
"""
import ast
import json
import os
import re
import statistics
import sys
import time
import urllib.request

BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11435").rstrip("/")
MODEL = os.environ.get("ARCH_ROUTER_MODEL", "hf.co/katanemo/Arch-Router-1.5B.gguf:Q4_K_M")

# Prompt text copied from the official model card (katanemo/Arch-Router-1.5B).
TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
"""
FORMAT_PROMPT = """
Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags.  Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent.
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
{"route": "route_name"}
"""

# First route set from the plan (Sprint 1 table). "other" is built in.
ROUTES = [
    {"name": "chit_chat", "description": "Greetings, small talk, casual conversation and simple personal opinions"},
    {"name": "simple_writing", "description": "Short emails, messages, rewrites, rephrasing or summaries of text the user provides"},
    {"name": "simple_facts", "description": "Well-known facts, definitions, word meanings and translations"},
    {"name": "private_data", "description": "Messages that contain or ask about personal data: passwords, API keys, bank or card details, addresses, phone numbers, health records or private documents"},
    {"name": "maths", "description": "Calculations, arithmetic, word problems, number theory, algebra, calculus or proofs, in any language"},
    {"name": "coding", "description": "Writing, debugging, reviewing or explaining code, software errors and programming tools"},
    {"name": "reasoning", "description": "Logic puzzles, riddles, multi-step planning and comparisons with several constraints"},
    {"name": "high_stakes", "description": "Medical symptoms, legal documents or contracts, and financial or investment advice"},
    {"name": "fresh_knowledge", "description": "Recent events, news, current prices, sports results or anything that changes over time"},
]
TIER = {"chit_chat": "local", "simple_writing": "local", "simple_facts": "local", "private_data": "local",
        "maths": "remote", "coding": "remote", "reasoning": "remote", "high_stakes": "remote",
        "fresh_knowledge": "remote", "other": "fallback"}

PREFILL = "{'route': '"
U, A = "user", "assistant"
# (conversation, expected route). Single turns reuse the earlier router probes.
PROBES = [
    ([(U, "hi")], "chit_chat"),
    ([(U, "What's the capital of Australia?")], "simple_facts"),
    ([(U, "Write a haiku about rain.")], "simple_writing"),
    ([(U, "Translate 'where is the train station' into Japanese.")], "simple_facts"),
    ([(U, "Write me a short python joke")], "chit_chat"),
    ([(U, "What is 2+2?")], "maths"),
    ([(U, "Calculate my mood today lol, I feel tired")], "chit_chat"),
    ([(U, "Summarise this: " + "The meeting covered budget items and the hiring plan. " * 20)], "simple_writing"),
    ([(U, "Explain why the sky is blue")], "simple_facts"),
    ([(U, "Is 2,147,483,647 prime? Answer yes or no.")], "maths"),
    ([(U, "What's the 40th Fibonacci number?")], "maths"),
    ([(U, "Who won the most recent F1 race?")], "fresh_knowledge"),
    ([(U, "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?")], "maths"),
    ([(U, "Calcule la dérivée de x³·sin(x).")], "maths"),
    ([(U, "x의 제곱근 2가 무리수임을 증명하라")], "maths"),
    ([(U, "Write a lock-free concurrent queue in Rust with memory ordering explained.")], "coding"),
    ([(U, "My Next.js app hydrates wrong after upgrading to React 19, why?")], "coding"),
    ([(U, "Draft a legally sound NDA clause for a Delaware LLC covering trade secrets.")], "high_stakes"),
    ([(U, "I have chest pain spreading to my left arm, what should I do?")], "high_stakes"),
    ([(U, "Three boxes are labelled wrongly: apples, oranges, mixed. You may pick one fruit from one box. How do you relabel them all?")], "reasoning"),
    ([(U, "My API key is sk-live-7d9f2a and it stopped working, can you check what's wrong with it?")], "private_data"),
    ([(U, "Here's my bank statement: 03/10 Woolworths -$84.20, 04/10 Rent -$650. Where am I overspending?")], "private_data"),
    ([(U, "Can you remember my home address is 12 Smith St, Parramatta?")], "private_data"),
    # Multi-turn: the latest intent should win.
    ([(U, "Can you help me debug this python function that crashes?"), (A, "Sure, paste the traceback."),
      (U, "thanks! what's a good name for a cat?")], "chit_chat"),
    ([(U, "hey how are you"), (A, "Good, thanks! How can I help?"),
      (U, "Prove that there are infinitely many primes.")], "maths"),
    ([(U, "Write a function that reverses a linked list in C."), (A, "Here is one: ..."),
      (U, "now do it in Rust")], "coding"),
    ([(U, "What's the weather like on Mars?"), (A, "Cold and dusty, around -60°C on average."),
      (U, "ok cool, thanks!")], "other"),
]


def call(path, payload, timeout=120):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def route(conversation):
    content = TASK_INSTRUCTION.format(
        routes=json.dumps(ROUTES),
        conversation=json.dumps([{"role": r, "content": c} for r, c in conversation]),
    ) + FORMAT_PROMPT
    started = time.monotonic()
    # Prefilling the answer's opening means the model only writes the route
    # name: about 3 tokens instead of 9, which halves warm latency.
    data = call("/api/chat", {"model": MODEL, "stream": False, "keep_alive": "10m",
                              "messages": [{"role": "user", "content": content},
                                           {"role": "assistant", "content": PREFILL}],
                              "options": {"temperature": 0, "num_predict": 16, "num_ctx": 2048}})
    elapsed = time.monotonic() - started
    text = PREFILL + data["message"]["content"].strip()
    # The model often answers in Python dict style ({'route': 'x'}), not JSON.
    match = re.search(r"\{.*\}", text, re.S)
    name = None
    for parse in (json.loads, ast.literal_eval):
        try:
            name = parse(match.group(0))["route"]
            break
        except (AttributeError, ValueError, SyntaxError, KeyError, TypeError):
            continue
    return name, text, elapsed, data.get("prompt_eval_count", 0)


def main():
    # Unload first so the first measured call is a true cold start.
    call("/api/generate", {"model": MODEL, "keep_alive": 0})
    _, _, cold, _ = route(PROBES[0][0])
    with urllib.request.urlopen(BASE + "/api/ps") as response:
        loaded = json.load(response)
    memory = {m["name"]: round(m.get("size", 0) / 1e9, 2) for m in loaded.get("models", [])}

    results, warm = [], []
    for conversation, expected in PROBES:
        name, raw, elapsed, prompt_tokens = route(conversation)
        warm.append(elapsed)
        results.append({"expected": expected, "got": name, "raw": raw, "ms": round(elapsed * 1000),
                        "prompt_tokens": prompt_tokens, "last": conversation[-1][1][:70]})

    parsed = [r for r in results if r["got"] in TIER]
    exact = sum(r["got"] == r["expected"] for r in results)
    tier_ok = sum(TIER.get(r["got"]) == TIER[r["expected"]] for r in results)
    private = [r for r in results if r["expected"] == "private_data"]
    warm_ms = sorted(round(x * 1000) for x in warm)
    p95 = warm_ms[min(len(warm_ms) - 1, round(0.95 * (len(warm_ms) - 1)))]
    for r in results:
        mark = "OK " if r["got"] == r["expected"] else ("~  " if TIER.get(r["got"]) == TIER[r["expected"]] else "BAD")
        print(f"{mark} {r['ms']:4}ms want={r['expected']:15} got={str(r['got']):15} {r['last']!r}")
    summary = {
        "model": MODEL,
        "cold_start_ms": round(cold * 1000),
        "warm_p50_ms": round(statistics.median(warm_ms)),
        "warm_p95_ms": p95,
        "parse_rate": f"{len(parsed)}/{len(results)}",
        "route_exact": f"{exact}/{len(results)}",
        "tier_correct": f"{tier_ok}/{len(results)}",
        "private_kept_local": f"{sum(TIER.get(r['got']) == 'local' for r in private)}/{len(private)}",
        "loaded_memory_gb": memory,
        "max_prompt_tokens": max(r["prompt_tokens"] for r in results),
    }
    print(json.dumps(summary, indent=2))
    return 0 if len(parsed) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
