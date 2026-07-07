#!/usr/bin/env python3
"""banana — team banana's interactive CLI for the hybrid routing agent.

Demo + local-use tooling ONLY (the pitch video, live Q&A). The SCORED
submission path (main.py entrypoint, Dockerfile, /input → /output contract)
is deliberately untouched: this script IMPORTS the exact modules the scored
run executes — run_task, Router, LocalModel, RemoteClient — so what a banana
session shows is literally the submission's routing, not a reimplementation.

Modes:
    banana                  interactive session (model loads once, stays warm)
    banana "a question"     answer one question and exit
    banana --demo           run tasks/demo_tasks.json + ANSI summary graph

Stdlib only, on purpose (the video machine needs nothing extra installed).
TokenTracker(log_path="") — demos never pollute logs/usage.jsonl, which is
the threshold-calibration audit trail.
"""
import argparse
import json
import os
import sys
import textwrap
import time
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load_dotenv():
    """Minimal KEY=VALUE .env loader so `banana` works from any shell
    without `source .env`. Runs BEFORE importing config (it reads env at
    import time). Existing env vars win — a tuning session's exports are
    never clobbered."""
    try:
        with open(os.path.join(REPO, ".env")) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    except FileNotFoundError:
        pass


_load_dotenv()
# Silence library noise (deprecations, hub progress bars, LibreSSL warning)
# before any heavy import — this is a stage-facing tool.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

from config import ROUTE_LOCAL, ROUTE_REMOTE, settings  # noqa: E402
from local_model import LocalModel  # noqa: E402
from main import run_task  # noqa: E402
from remote_client import RemoteClient  # noqa: E402
from router import Router  # noqa: E402
from schemas import Task  # noqa: E402
from token_tracker import TokenTracker  # noqa: E402

# ── colors (graceful fallback: pipes, NO_COLOR, dumb terminals) ──────────
USE_COLOR = (
    hasattr(sys.stdout, "isatty")
    and sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM", "") != "dumb"
)


def _c(code, s):
    return "\033[{}m{}\033[0m".format(code, s) if USE_COLOR else s


def green(s):
    return _c("32", s)


def yellow(s):
    return _c("33", s)


def red(s):
    return _c("31", s)


def dim(s):
    return _c("2", s)


def bold(s):
    return _c("1", s)


class Session:
    """One warm agent: model loaded once, totals accumulated across asks."""

    def __init__(self):
        self.router = Router()
        self.local = LocalModel()
        self.remote = RemoteClient()
        self.tracker = TokenTracker(log_path="")  # never touch usage.jsonl
        self.asked = 0
        self.local_n = 0
        self.billable = 0
        self.free_tokens = 0

    def warm(self):
        if settings.mock_mode or self.local.loaded:
            return
        print(dim("loading local model ({}) …".format(settings.local_model_name)))
        started = time.time()
        self.local.load()
        print(dim("ready in {:.1f}s — model stays warm for this session".format(
            time.time() - started)))

    def ask(self, prompt, task_id=None):
        task = Task(task_id or "q{}".format(self.asked + 1), prompt)
        result = run_task(task, self.router, self.local, self.remote, self.tracker)
        self.asked += 1
        if result["route"] == ROUTE_LOCAL:
            self.local_n += 1
        self.billable += result["billable_tokens"]
        rec = self.tracker.records[-1]
        self.free_tokens += rec.local_prompt_tokens + rec.local_completion_tokens
        return result

    def footer(self):
        return dim("session: {} asked · {} local (free) · {} billable tokens".format(
            self.asked, self.local_n, self.billable))


def route_tag(result, pad=0):
    """Colored route label; padding happens on the PLAIN string so ANSI
    codes never break column alignment."""
    if result["route"] == ROUTE_LOCAL:
        plain = "LOCAL · free"
        paint = green
    elif result["route"] == ROUTE_REMOTE:
        plain = "REMOTE · {} tok".format(result["billable_tokens"])
        if result["escalated"]:
            plain += " (esc)"
        paint = yellow
    else:
        plain = "ERROR"
        paint = red
    return paint(plain.ljust(pad) if pad else plain)


def print_answer(result):
    body = result["answer"].strip() or "(no answer)"
    for para in body.splitlines():
        print(textwrap.fill(para, width=76, initial_indent="  ",
                            subsequent_indent="  ") if para.strip() else "")


def interactive():
    print(bold("🍌 banana") + dim(" — token-efficient routing agent  (team banana)"))
    print(dim("local model answers for FREE; hard questions go remote and bill tokens"))
    print(dim("type a question · exit/quit/:q to leave"))
    print()
    try:
        import readline  # noqa: F401  (line editing / history for the demo)
    except ImportError:
        pass
    session = Session()
    session.warm()
    while True:
        try:
            line = input("banana › ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("exit", "quit", ":q"):
            break
        result = session.ask(line)
        print("  " + route_tag(result) + dim("   confidence {:.2f}".format(
            result["confidence"])))
        print_answer(result)
        print("  " + session.footer())
        print()
    print("bye 🍌")


def one_shot(question):
    session = Session()
    session.warm()
    result = session.ask(question)
    print("  " + route_tag(result))
    print_answer(result)
    return 0


def _bar(value, max_value, width=24):
    if max_value <= 0:
        return ""
    n = int(round(width * value / max_value))
    if value > 0:
        n = max(n, 1)  # nonzero values always get a visible bar
    return "█" * n


def demo():
    print(bold("🍌 banana — demo") + dim("  (8 tasks, one per Track-1 category)"))
    print()
    session = Session()
    session.warm()
    print()
    with open(os.path.join(REPO, "tasks", "demo_tasks.json")) as fh:
        tasks = json.load(fh)

    for item in tasks:
        result = session.ask(item["prompt"], task_id=item["task_id"])
        preview = result["answer"].replace("\n", " ").strip()[:52]
        print("  {} {} {}".format(
            item["task_id"].ljust(12), route_tag(result, pad=22), dim(preview)))

    total = session.asked
    remote_n = total - session.local_n
    billed = session.billable
    free = session.free_tokens

    # All-remote comparison: mean billable of the REMOTE-answered tasks in
    # THIS run, applied to every task. Clearly an estimate — the local
    # tasks' remote cost was never measured (that's the point of the agent).
    est = None
    if remote_n:
        est = int(round(billed / remote_n * total))

    print()
    max_count = max(session.local_n, remote_n, 1)
    print("  routing   {} {} {}".format(
        "local ".ljust(7), green(_bar(session.local_n, max_count)), session.local_n))
    print("            {} {} {}".format(
        "remote".ljust(7), yellow(_bar(remote_n, max_count)), remote_n))
    max_tok = max(billed, free, 1)
    print("  tokens    {} {} {:,}".format(
        "billed".ljust(7), yellow(_bar(billed, max_tok)), billed))
    print("            {} {} {:,}  {}".format(
        "free  ".ljust(7), green(_bar(free, max_tok)), free,
        dim("(local — costs nothing)")))
    print()
    print("  " + bold("{} of {} tasks answered free".format(session.local_n, total)))
    if est and est > billed:
        saved = 100.0 * (1 - billed / est)
        print("  all-remote agent: ~{:,} tokens {} → banana billed {:,}, "
              "saved ~{:.0f}%".format(est, dim("(estimate)"), billed, saved))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="banana",
        description="team banana's CLI for the hybrid routing agent "
                    "(demo tooling — the scored path lives in main.py)",
    )
    parser.add_argument("--demo", action="store_true",
                        help="run tasks/demo_tasks.json and print the summary graph")
    parser.add_argument("question", nargs="*",
                        help="ask one question and exit (no args = interactive)")
    args = parser.parse_args(argv)

    if args.demo:
        return demo()
    if args.question:
        return one_shot(" ".join(args.question))
    interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
