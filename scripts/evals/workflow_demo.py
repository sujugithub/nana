"""Research → cited note → reviewable email/calendar drafts in disposable storage.

Defaults to deterministic fixtures. --live-local uses installed Qwen 4B and
reads example.com; it cannot send mail or publish a calendar event.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace import store, runs, tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-local", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prompt = ("Read https://example.com/ and save a note titled Example source with a short factual summary and its source URL. "
              "Then create an email draft titled Example briefing, addressed to nobody@example.com, summarizing the note. "
              "Then create a local calendar event draft titled Review example, starting 2026-10-10T10:00:00+11:00 and ending 2026-10-10T10:30:00+11:00. "
              "Finally report the items saved. Do not send or publish anything. Use one tool per turn.")
    replies = iter([
        {"tool": "read_url", "arguments": {"url": "https://example.com/"}},
        {"tool": "save_item", "arguments": {"kind": "note", "title": "Example source", "content": "Example Domain is for documentation. Source: https://example.com/"}},
        {"tool": "save_item", "arguments": {"kind": "draft", "title": "Example briefing", "content": "Review this example.", "meta": {"to": "nobody@example.com"}}},
        {"tool": "save_item", "arguments": {"kind": "event", "title": "Review example", "content": "Discuss the source", "meta": {"start": "2026-10-10T10:00:00+11:00", "end": "2026-10-10T10:30:00+11:00"}}},
        {"answer": "Saved a cited note and two drafts. Source: https://example.com/"},
    ])
    def fixture(*unused):
        return {"answer": json.dumps(next(replies)), "final_model": "fixture", "provider": "local", "route": "local", "tokens": {}, "estimated_cost_usd": 0, "mock": True}
    with tempfile.TemporaryDirectory() as root, patch.object(store, "ROOT", Path(root)):
        store.preferences({"local_backend": "ollama", "local_model": "qwen3.5:4b"})
        run_id = store.run_create("research", {"mode": "local", "model": "qwen3.5:4b", "prompt": prompt, "tools": ["read_url", "save_item"]})
        if args.live_local:
            runs.worker("local", run_id, True)
        else:
            with patch.object(runs, "generate", side_effect=fixture), patch.object(tools, "read_url", return_value={"url": "https://example.com/", "content": "Example Domain is for documentation."}):
                runs.worker("local", run_id, False)
        run = store.run_get(run_id)
        counts = {kind: len(store.items(kind)) for kind in ["note", "draft", "event"]}
        validations = {"one_of_each": all(count == 1 for count in counts.values()),
                       "cited_note": any("https://example.com/" in item["content"] for item in store.items("note")),
                       "recipient": any(item["meta"].get("to") == "nobody@example.com" for item in store.items("draft")),
                       "event_times": any(item["meta"].get("start") == "2026-10-10T10:00:00+11:00" and item["meta"].get("end") == "2026-10-10T10:30:00+11:00" for item in store.items("event"))}
        success = run["status"] == "completed" and all(validations.values())
        evidence = {"live_local": args.live_local, "success": success, "counts": counts,
                    "validations": validations, "run": run, "items": store.items(), "external_publications": 0}
        Path(args.output).write_text(json.dumps(evidence, indent=2))
        print(json.dumps({"live_local": args.live_local, "success": success, "counts": counts, "status": run["status"], "result": run["result"]}))
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
