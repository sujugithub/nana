"""Bounded sequential smoke pack for installed Ollama models; never pulls weights.

python scripts/evals/local_profiles.py --output /tmp/nana-local-profiles.json
Each model gets <= 300 seconds, requests timeout after 45 seconds, 128 output
tokens and three repeats of ten distinct fixtures. Tools are never executed.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace.contracts import catalog, decision
from workspace.tools import TOOLS

FIXTURES = [
    ("list-notes", "List my notes now.", "list_items", lambda a: a.get("kind") == "note"),
    ("read-note", "Read the workspace item whose id is fixture123.", "read_item", lambda a: a.get("id") == "fixture123"),
    ("save-note", "Save a note titled Mac with content Nana runs on macOS.", "save_item", lambda a: a.get("kind") == "note" and a.get("title") == "Mac"),
    ("email-draft", "Create an email draft titled Hello with content Review tomorrow. Do not send it.", "save_item", lambda a: a.get("kind") == "draft"),
    ("calendar-draft", "Create a local calendar event draft titled Review with content Discuss the plan. Do not publish it.", "save_item", lambda a: a.get("kind") == "event"),
    ("search", "Search the web for macOS documentation.", "search", lambda a: "mac" in a.get("query", "").lower()),
    ("read-source", "Read https://example.com/ now.", "read_url", lambda a: a.get("url", "").rstrip("/") == "https://example.com"),
    ("load-skill", "Read the approved skill whose id is research123.", "read_skill", lambda a: a.get("id") == "research123"),
    ("list-files", "List workspace files now.", "list_files", lambda a: a == {}),
    ("read-file", "Read the workspace file notes.txt.", "read_file", lambda a: a.get("path") == "notes.txt"),
]


def request(path, data=None, timeout=45):
    req = urllib.request.Request("http://127.0.0.1:11435" + path, data=json.dumps(data).encode() if data else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def memory():
    return {"vm_stat": subprocess.run(["vm_stat"], capture_output=True, text=True).stdout,
            "swap": subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", action="append", choices=["qwen3.5:4b", "qwen3.8:27b"])
    args = parser.parse_args()
    tags = {m["name"] for m in request("/api/tags")["models"]}
    report = {"mock": False, "runtime": request("/api/version"), "before": memory(), "results": []}
    output = Path(args.output)
    for model, context in [("qwen3.5:4b", 4096), ("qwen3.8:27b", 2048)]:
        if args.model and model not in args.model:
            continue
        if model not in tags:
            report["results"].append({"model": model, "error": "not installed"})
            continue
        started = time.monotonic()
        try:
            for repeat in range(3):
                for name, prompt, tool, accepts in FIXTURES:
                    if time.monotonic()-started > 300:
                        raise TimeoutError("Profile's 300-second trial budget exhausted")
                    payload = {"model": model, "messages": [{"role": "system", "content": "Use the requested tool exactly once with the supplied values. Do not claim it has executed."}, {"role": "user", "content": prompt}],
                               "tools": catalog([tool], TOOLS), "stream": False, "think": False, "keep_alive": "5m",
                               "options": {"temperature": 0, "num_ctx": context, "num_predict": 128, "seed": 7}}
                    at = time.monotonic()
                    row = {"model": model, "context": context, "task": name, "repeat": repeat+1, "success": False}
                    try:
                        response = request("/api/chat", payload)
                        message = response.get("message", {})
                        parsed = decision({"answer": message.get("content", ""), "tool_calls": message.get("tool_calls", [])}, [tool])
                        row["decision"] = parsed
                        row["success"] = parsed.get("tool") == tool and accepts(parsed.get("arguments", {}))
                        row.update({key: response.get(key) for key in ["prompt_eval_count", "eval_count", "load_duration", "eval_duration", "total_duration"]})
                        row["loaded"] = [{k: m.get(k) for k in ["name", "size_vram", "context_length"]} for m in request("/api/ps")["models"]]
                    except Exception as exc:
                        row["error"] = str(exc)[:300]
                    row["latency_s"] = round(time.monotonic()-at, 3)
                    report["results"].append(row)
                    output.write_text(json.dumps(report, indent=2))
                    print(json.dumps(row), flush=True)
                    if "error" in row and ("timed out" in row["error"].lower() or "unavailable" in row["error"].lower()):
                        # A client timeout does not prove inference stopped.
                        # Do not queue another generation behind an unknown one.
                        raise TimeoutError("Transport timeout: stop this profile and unload before any further inference")
        except TimeoutError as exc:
            report.setdefault("limits", []).append({"model": model, "error": str(exc)})
        finally:
            request("/api/generate", {"model": model, "keep_alive": 0})
            report[model + "_memory"] = memory()
            output.write_text(json.dumps(report, indent=2))
    report["after"] = memory()
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
