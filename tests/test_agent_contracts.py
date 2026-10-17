"""Deterministic acceptance pack; no live credentials, storage or model calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from workspace import store, runs, tools, contracts, context, effects


def response(value, **extra):
    return {"answer": value if isinstance(value, str) else json.dumps(value), "final_model": "fixture",
            "provider": "local", "route": "local", "tokens": {"prompt": 10, "completion": 5},
            "estimated_cost_usd": 0, "mock": True, **extra}


class AgentContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = patch.object(store, "ROOT", Path(self.temp.name))
        self.root.start()
        self.token = store.USER.set("local")

    def tearDown(self):
        store.USER.reset(self.token)
        self.root.stop()
        self.temp.cleanup()

    def run_fixture(self, replies, selected=(), kind="agent"):
        key = store.run_create(kind, {"prompt": "Look up Nana's platform", "tools": list(selected)})
        with patch.object(runs, "generate", side_effect=[response(r) for r in replies]):
            runs.worker("local", key, False)
        return store.run_get(key)

    def test_malformed_is_never_success_and_retries_are_bounded(self):
        for _ in range(3):
            result = self.run_fixture(["I did the task", "[]", '{"answer":2}'])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(runs.usage(result["steps"])["requests"], 3)
            self.assertFalse(store.items("note"))

    def test_validation_can_recover_without_running_bad_tool(self):
        result = self.run_fixture([{"tool": "save_item", "arguments": {"kind": "note", "title": 3}}, {"answer": "Please provide a title"}], ["save_item"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(store.items("note")), 0)
        self.assertEqual(runs.usage(result["steps"])["validation_errors"], 1)

    def test_native_call_and_finish_have_same_contract(self):
        self.assertEqual(contracts.decision({"protocol": "native", "answer": "Saved the note"}, []), {"answer": "Saved the note"})
        with self.assertRaises(ValueError):
            contracts.decision({"answer": "Saved the note"}, [])
        for args in ({"id": "abc"}, '{"id":"abc"}'):
            value = {"tool_calls": [{"id": "1", "function": {"name": "read_item", "arguments": args}}]}
            self.assertEqual(contracts.decision(value, ["read_item"])["arguments"], {"id": "abc"})
        with self.assertRaises(ValueError):
            contracts.decision({"tool_calls": [{"function": {"name": "finish", "arguments": {"answer": ""}}}]}, [])
        with self.assertRaises(ValueError):
            contracts.arguments("shell", {"command": "echo unsafe"})

    def test_tool_failure_is_not_a_successful_result(self):
        result = self.run_fixture([{"tool": "read_item", "arguments": {"id": "missing"}}, {"answer": "The item was not found"}], ["read_item"])
        self.assertFalse(result["steps"][1]["ok"])
        self.assertTrue(result["steps"][1]["error"])
        self.assertFalse(contracts.result_envelope({"exit_code": 1})["ok"])

    def test_budget_survives_resume(self):
        key = store.run_create("agent", {"prompt": "x", "tools": []})
        store.run_save(key, steps=[{"type": "model", "cost": .10}])
        with patch.object(runs, "generate") as generate:
            runs.worker("local", key, False)
            generate.assert_not_called()
        self.assertEqual(store.run_get(key)["status"], "failed")

    def test_offload_is_scoped_complete_and_context_has_whole_exchanges(self):
        raw = {"url": "https://example.com", "content": "large" * 5000}
        result = context.offload("fixture", raw)
        self.assertEqual(json.loads(tools.file_path(result["path"]).read_text()), raw)
        self.assertEqual(result["url"], raw["url"])
        steps = [{"type": "model", "answer": "x"*300}, {"type": "tool", "output": "old"}, {"type": "model", "answer": "new"}, {"type": "tool", "output": "new"}]
        history = context.history(steps, 150)
        self.assertEqual([s["type"] for s in history], ["compaction", "model", "tool"])
        store.USER.set("other")
        self.assertFalse((store.home()/"files"/result["path"]).exists())

    def test_memory_relevance_provenance_versions_and_deletion(self):
        a = store.save({"kind": "memory", "title": "Mac preference", "content": "Use concise Mac instructions"})
        store.save({"kind": "memory", "title": "Cooking", "content": "vegetables"})
        store.save({"kind": "memory", "title": "Private Mac", "content": "Mac", "meta": {"scope": "chat:2"}})
        self.assertEqual([x["id"] for x in context.relevant("memory", "Mac instructions")], [a["id"]])
        b = store.save({"content": "Detailed Mac instructions"}, a["id"])
        self.assertEqual(b["meta"]["version"], 2)
        self.assertEqual(len(store.revisions(a["id"])), 1)
        store.delete(a["id"])
        self.assertEqual(context.relevant("memory", "Mac instructions"), [])

    def test_search_chats_returns_complete_attributed_user_scoped_turns(self):
        from chat import database, repository
        with patch.object(database, "DB_PATH", Path(self.temp.name)/"chat.db"):
            database.initialise_database()
            chat = repository.create_conversation("Mac facts")
            repository.add_message(chat.id, "user", "Which Mac model?")
            repository.add_message(chat.id, "assistant", "Fixture Mac")
            result = tools.call("search_chats", {"query": "Mac"})
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["conversation_id"], chat.id)
            self.assertEqual([m["role"] for m in result[0]["messages"]], ["user", "assistant"])
            store.USER.set("other")
            self.assertEqual(tools.call("search_chats", {"query": "Mac"}), [])

    def test_skill_requires_explicit_user_approval(self):
        item = store.save({"kind": "skill", "title": "Research", "content": "Read sources"})
        with self.assertRaises(ValueError):
            tools.call("read_skill", {"id": item["id"]})
        store.save({"meta": {"approved": True}}, item["id"])
        self.assertEqual(tools.call("read_skill", {"id": item["id"]})["content"], "Read sources")
        store.save({"content": "Changed instructions"}, item["id"])
        with self.assertRaises(ValueError):
            tools.call("read_skill", {"id": item["id"]})

    def test_identical_local_write_has_one_effect(self):
        decision = {"tool": "save_item", "arguments": {"kind": "note", "title": "Fixture", "content": "Once"}}
        self.assertEqual(effects.identity("run", 1, decision), effects.identity("run", 3, decision))
        result = self.run_fixture([decision, decision, {"answer": "Saved"}], ["save_item"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(store.items("note")), 1)

    def test_stale_review_cannot_approve_a_new_request(self):
        result = self.run_fixture([{"tool": "shell", "arguments": {"command": ["echo", "fixture"]}}], ["shell"])
        with self.assertRaisesRegex(ValueError, "changed"):
            runs.review(result["id"], True, False, "older-action")
        self.assertEqual(store.run_get(result["id"])["status"], "awaiting_review")

    def test_native_history_keeps_call_result_pairs(self):
        run = {"payload": {"prompt": "Read an item"}}
        steps = [{"type": "tool", "action_id": "abc", "tool": "read_item", "arguments": {"id": "fixture"}, "output": {"content": "data"}, "ok": True}]
        messages = context.native_messages(run, ["read_item"], steps, tools.TOOLS, 4096)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[-2]["tool_calls"][0]["id"], messages[-1]["tool_call_id"])
        with self.assertRaisesRegex(ValueError, "context profile"):
            context.native_messages({"payload": {"prompt": "x"*10000}}, ["read_item"], [], tools.TOOLS, 2048)

    def test_cancellation_stops_a_running_shell(self):
        import sys
        import threading
        import time
        from generation_stream import CURRENT, Stream, Cancelled
        stream = Stream(lambda event: None)
        token = CURRENT.set(stream)
        timer = threading.Timer(.1, stream.cancel.set)
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(Cancelled):
                tools.call("shell", {"command": [sys.executable, "-c", "import time; time.sleep(10)"]}, approved=True)
            self.assertLess(time.monotonic()-started, 3)
        finally:
            CURRENT.reset(token)
            timer.join()

    def test_action_identity_and_uncertain_effect_block_replay(self):
        pending = {"tool": "shell", "arguments": {"command": ["echo", "fixture"]}, "action_id": "fixture"}
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            effects.execute_action("run", pending, lambda: (_ for _ in ()).throw(TimeoutError()))
        with patch.object(tools, "call") as execute:
            with self.assertRaisesRegex(RuntimeError, "uncertain"):
                effects.execute_action("run", pending, execute)
            execute.assert_not_called()
        with self.assertRaisesRegex(ValueError, "identity"):
            effects.execute_action("run", {**pending, "arguments": {"command": ["different"]}}, lambda: {})

    def test_research_to_cited_note_and_reviewable_drafts(self):
        url = "https://example.com/fixture"
        steps = [{"tool": "read_url", "arguments": {"url": url}},
                 {"tool": "save_item", "arguments": {"kind": "note", "title": "Research", "content": f"Nana runs on macOS. Source: {url}"}},
                 {"tool": "save_item", "arguments": {"kind": "draft", "title": "Research email", "content": "Review this research before sending"}},
                 {"tool": "save_item", "arguments": {"kind": "event", "title": "Review research", "content": "Proposed calendar event"}},
                 {"answer": f"Saved the note and prepared drafts. Source: {url}"}]
        with patch.object(tools, "read_url", return_value={"url": url, "content": "Nana runs on macOS."}):
            result = self.run_fixture(steps, ["read_url", "save_item"], "research")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["sources"], [url])
        self.assertEqual([len(store.items(k)) for k in ["note", "draft", "event"]], [1, 1, 1])

    def test_research_rejects_unread_citations(self):
        result = self.run_fixture([{"answer": "Source: https://unread.example"}]*3, [], "research")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(runs.usage(result["steps"])["validation_errors"], 3)

    def test_mcp_uses_current_discovered_schema_before_execution(self):
        from workspace import mcp
        store.preferences({"mcp_servers": {"fixture": {"command": ["fixture"]}}})
        with patch.object(mcp, "Session") as session:
            call = session.return_value.__enter__.return_value.call
            call.side_effect = lambda method, args=None: {"tools": [{"name": "echo", "inputSchema": contracts.obj({"text": contracts.S}, ["text"])}]} if method == "tools/list" else {"content": [{"type": "text", "text": "fixture"}]}
            result = tools.call("mcp", {"server": "fixture", "tool": "echo", "arguments": {"text": "fixture"}}, approved=True)
            self.assertEqual(result["content"][0]["text"], "fixture")
            call.reset_mock()
            with self.assertRaises(ValueError):
                tools.call("mcp", {"server": "fixture", "tool": "echo", "arguments": {"text": 3}}, approved=True)
            self.assertEqual([c.args[0] for c in call.call_args_list], ["tools/list"])


if __name__ == "__main__":
    unittest.main()
