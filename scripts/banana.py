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

Interactive extras (demo conveniences, NOT scored behavior — the harness
sends independent tasks, so main.py has no history):
    follow-ups              the whole conversation rides along as context
                            (oldest turns trimmed past BANANA_CTX_CHARS)
    /save [name]            write the last answer's code block to banana_out/
    /clear                  forget the conversation context
    :stats                  session bar graph without leaving

Stdlib only, on purpose (the video machine needs nothing extra installed).
TokenTracker(log_path="") — demos never pollute logs/usage.jsonl, which is
the threshold-calibration audit trail.
"""
import argparse
import contextlib
import io
import json
import os
import re
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


# Figlet-style wordmark — printed line by line so each gets its own color.
_WORDMARK = [
    r"  _                                    ",
    r" | |__   __ _ _ __   __ _ _ __   __ _  ",
    r" | '_ \ / _` | '_ \ / _` | '_ \ / _` | ",
    r" | |_) | (_| | | | | (_| | | | | (_| | ",
    r" |_.__/ \__,_|_| |_|\__,_|_| |_|\__,_| ",
]


def print_banner():
    print()
    for i, line in enumerate(_WORDMARK):
        suffix = "  🍌" if i == 1 else ""
        print(yellow(bold(line)) + suffix)
    print(dim(" token-efficient routing agent · team banana"))
    print()
    local_name = settings.local_model_name.rsplit("/", 1)[-1]
    with contextlib.redirect_stderr(io.StringIO()):  # mute the dev-fallback NOTE
        remote_name = (RemoteClient().model_name or "none").rsplit("/", 1)[-1]
    print("   {} {}".format(dim("models "),
          green(local_name + " (local · free)") + dim("  →  ")
          + yellow(remote_name + " (remote · billed)")))
    print("   {} {}".format(dim("router "),
          dim("confidence ≥ {:.2f} stays local · self-check gate {:.2f}".format(
              settings.confidence_threshold,
              settings.logprob_confidence_threshold))))
    print()


class Session:
    """One warm agent: model loaded once, totals accumulated across asks."""

    # Context is an INTERACTIVE-ONLY convenience: the scored contract is
    # independent {task_id, prompt} pairs, so main.py has no history and
    # never will. The WHOLE chat is carried, newest-first, within a char
    # budget (the local model's window is finite, and history drags the
    # router's length score down → long sessions drift remote/billed —
    # /clear resets). Override the budget with BANANA_CTX_CHARS.
    CTX_CHARS = int(os.environ.get("BANANA_CTX_CHARS", "8000"))

    def __init__(self):
        self.router = Router()
        self.local = LocalModel()
        with contextlib.redirect_stderr(io.StringIO()):  # mute model-pick notes
            self.remote = RemoteClient()
        self.tracker = TokenTracker(log_path="")  # never touch usage.jsonl
        self.asked = 0
        self.local_n = 0
        self.billable = 0
        self.free_tokens = 0
        self.history = []      # [(question, answer)] — the whole session
        self.last_answer = ""

    def warm(self):
        if settings.mock_mode or self.local.loaded:
            return
        print(dim("loading local model ({}) …".format(settings.local_model_name)))
        started = time.time()
        self.local.load()
        print(dim("ready in {:.1f}s — model stays warm for this session".format(
            time.time() - started)))

    def _contextual(self, question):
        """Prepend the conversation so follow-ups ("how old is he?") resolve.
        Whole chat, newest-first selection within CTX_CHARS, then restored to
        chronological order — when a long session exceeds the budget it's the
        OLDEST turns that fall off. The router decides on the full contextual
        prompt — context-heavy follow-ups routing remote is the safe
        direction."""
        if not self.history:
            return question
        picked = []
        used = 0
        for q, a in reversed(self.history):
            turn = "user: " + q + "\nassistant: " + a
            if used + len(turn) > self.CTX_CHARS and picked:
                break
            picked.append(turn)
            used += len(turn)
        lines = ["Previous conversation (context only):"]
        lines.extend(reversed(picked))
        lines.append("")
        lines.append("Answer only this new question: " + question)
        return "\n".join(lines)

    def ask(self, prompt, task_id=None, use_context=False):
        sent = self._contextual(prompt) if use_context else prompt
        task = Task(task_id or "q{}".format(self.asked + 1), sent)
        result = run_task(task, self.router, self.local, self.remote, self.tracker)
        self.asked += 1
        if result["route"] == ROUTE_LOCAL:
            self.local_n += 1
        self.billable += result["billable_tokens"]
        rec = self.tracker.records[-1]
        self.free_tokens += rec.local_prompt_tokens + rec.local_completion_tokens
        if use_context:
            self.history.append((prompt, result["answer"]))
        self.last_answer = result["answer"]
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


_EXT = {"python": ".py", "py": ".py", "javascript": ".js", "js": ".js",
        "typescript": ".ts", "html": ".html", "css": ".css", "bash": ".sh",
        "sh": ".sh", "json": ".json", "sql": ".sql", "c": ".c", "cpp": ".cpp",
        "java": ".java", "go": ".go", "rust": ".rs"}


def save_code(answer, name=None):
    """Write the LAST fenced code block of `answer` to banana_out/<name>.
    Returns the path, or None if there is no code block. This is the whole
    'it can code' feature: the model writes code as text, banana puts it on
    disk — no agentic tool-calling is claimed or implemented."""
    blocks = re.findall(r"```(\w*)\n(.*?)```", answer, re.S)
    if not blocks:
        return None
    lang, code = blocks[-1]
    out_dir = os.path.join(os.getcwd(), "banana_out")
    os.makedirs(out_dir, exist_ok=True)
    if not name:
        name = "snippet" + _EXT.get(lang.lower(), ".txt")
    elif "." not in name:
        name += _EXT.get(lang.lower(), ".txt")
    path = os.path.join(out_dir, os.path.basename(name))
    with open(path, "w") as fh:
        fh.write(code.rstrip() + "\n")
    return path


def session_graph(session):
    """ANSI bar graph of the session so far — shared by --demo, the :stats
    command, and the end-of-session summary."""
    remote_n = session.asked - session.local_n
    billed = session.billable
    free = session.free_tokens
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
    print("  " + bold("{} of {} tasks answered free".format(
        session.local_n, session.asked)))
    # All-remote comparison: mean billable of the REMOTE-answered tasks in
    # THIS session, applied to every task. Clearly an estimate — the local
    # tasks' remote cost was never measured (that's the point of the agent).
    if remote_n and session.asked > remote_n:
        est = int(round(billed / remote_n * session.asked))
        if est > billed:
            saved = 100.0 * (1 - billed / est)
            print("  all-remote agent: ~{:,} tokens {} → banana billed {:,}, "
                  "saved ~{:.0f}%".format(est, dim("(estimate)"), billed, saved))


def interactive():
    print_banner()
    print(dim(" type a question — follow-ups remember the whole conversation"))
    print(dim(" /save [name] writes the last code block to banana_out/ · "
              "/clear forgets context · :stats graph · exit"))
    print()
    try:
        import readline  # noqa: F401  (line editing / history for the demo)
    except ImportError:
        pass
    session = Session()
    session.warm()
    print()
    while True:
        try:
            line = input("banana › ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low in ("exit", "quit", ":q"):
            break
        if low in (":stats", ":s"):
            if session.asked:
                print()
                session_graph(session)
            else:
                print(dim("  nothing asked yet"))
            print()
            continue
        if low == "/clear":
            session.history = []
            print(dim("  context cleared — next question starts fresh"))
            print()
            continue
        if low == "/save" or low.startswith("/save "):
            name = line[6:].strip() or None
            path = save_code(session.last_answer, name)
            if path:
                print(green("  saved → {}".format(os.path.relpath(path))))
            else:
                print(dim("  no code block in the last answer"))
            print()
            continue
        result = session.ask(line, use_context=True)
        print("  " + route_tag(result) + dim("   confidence {:.2f}".format(
            result["confidence"])))
        print_answer(result)
        print("  " + session.footer())
        print()
    if session.asked:
        print()
        session_graph(session)
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
    print_banner()
    print(dim(" demo: 8 tasks, one per Track-1 category"))
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

    print()
    session_graph(session)
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
