"""Behavioural checks for persistence, isolation, review and external boundaries."""
from datetime import datetime, timedelta, timezone
import json
import ssl
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit

from workspace import store, auth, connections, tools, runs, scheduler
from webui.server import make_server


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_root = store.ROOT
        store.ROOT = Path(self.temp.name)
        self.token = store.USER.set("local")
        auth.SESSIONS.clear()
        auth.FAILURES.clear()

    def tearDown(self):
        # Every started run in these tests is awaited before deleting its DB.
        store.USER.reset(self.token)
        store.ROOT = self.old_root
        self.temp.cleanup()

    def wait(self, run_id):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            run = store.run_get(run_id)
            if run["status"] not in {"queued", "running", "cancelling"}:
                return run
            time.sleep(.01)
        self.fail("Run did not finish")

    def test_revision_and_user_isolation(self):
        item = store.save({"kind": "document", "title": "Plan", "content": "Original"})
        store.save({"content": "Revision"}, item["id"])
        self.assertEqual(store.revisions(item["id"])[0]["content"], "Original")
        store.USER.set("other")
        self.assertEqual(store.items(), [])
        with self.assertRaises(LookupError):
            store.get(item["id"])
        store.USER.set("local")
        self.assertEqual(store.get(item["id"])["content"], "Revision")

    def test_tools_and_workspace_path_boundaries(self):
        with self.assertRaises(ValueError):
            tools.file_path("../../connections.json")
        with self.assertRaises(ValueError):
            tools.public_url("http://127.0.0.1:8642")
        with self.assertRaises(PermissionError):
            tools.call("shell", {"command": ["echo", "unsafe"]})
        item = tools.call("save_item", {"kind": "memory", "title": "Style", "content": "Concise"})
        self.assertEqual(tools.call("read_item", {"id": item["id"]})["content"], "Concise")
        with self.assertRaises(ValueError):
            tools.call("save_item", {"kind": "asset", "title": "Invalid"})

    def test_agent_tool_result_is_persisted_before_final_answer(self):
        replies = [json.dumps({"tool": "save_item", "arguments": {"kind": "note", "title": "Saved by agent", "content": "Works"}}), json.dumps({"answer": "Saved your note."})]
        def generate(payload, prompt, allow_real):
            return {"answer": replies.pop(0), "final_model": "fixture", "provider": "local", "route": "cheap", "tokens": {}, "estimated_cost_usd": 0, "mock": False}
        with patch.object(runs, "generate", side_effect=generate):
            run = runs.start("agent", {"prompt": "Save a note", "tools": ["save_item"]}, False)
            final = self.wait(run["id"])
        self.assertEqual(final["status"], "completed")
        self.assertEqual(len(store.items("note")), 1)
        self.assertEqual([s["type"] for s in final["steps"]], ["model", "tool", "model"])

    def test_shell_waits_for_review_and_decline_does_not_execute(self):
        reply = {"answer": json.dumps({"tool": "shell", "arguments": {"command": ["echo", "hello"]}}),
                 "final_model": "fixture", "provider": "local", "route": "cheap", "tokens": {}, "estimated_cost_usd": 0, "mock": False}
        with patch.object(runs, "generate", return_value=reply), patch.object(tools.subprocess, "run") as execute:
            run = runs.start("agent", {"prompt": "Print hello", "tools": ["shell"]}, False)
            waiting = self.wait(run["id"])
            self.assertEqual(waiting["status"], "awaiting_review")
            runs.review(run["id"], False, False)
            execute.assert_not_called()
        self.assertEqual(store.run_get(run["id"])["status"], "cancelled")

    def test_cancellation_prevents_follow_up_tool_action(self):
        entered, release = threading.Event(), threading.Event()
        def generate(*args):
            entered.set()
            release.wait(2)
            return {"answer": '{"tool":"save_item","arguments":{"kind":"note","title":"Do not save"}}', "final_model": "fixture", "provider": "local", "route": "cheap", "tokens": {}, "estimated_cost_usd": 0, "mock": False}
        with patch.object(runs, "generate", side_effect=generate):
            run = runs.start("agent", {"prompt": "Test", "tools": ["save_item"]}, False)
            self.assertTrue(entered.wait(2))
            runs.cancel(run["id"])
            release.set()
            self.assertEqual(self.wait(run["id"])["status"], "cancelled")
        self.assertEqual(store.items("note"), [])

    def test_reminder_is_delivered_once(self):
        store.job_save({"title": "Due", "due": (datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()})
        scheduler.tick(False)
        scheduler.tick(False)
        self.assertEqual(len(store.items("notification")), 1)
        self.assertEqual(store.jobs()[0]["status"], "completed")

    def test_oauth_state_and_secrets_are_not_exposed(self):
        connections.configure({"google_client_id": "test-client", "google_client_secret": "test-secret"})
        self.assertNotIn("test-secret", json.dumps(connections.status()))
        result = connections.google_start(8642)
        self.assertIn("code_challenge=", result["url"])
        with self.assertRaises(ValueError):
            connections.google_callback({"state": ["wrong"], "code": ["code"]})
        self.assertFalse(connections.status()["google_connected"])
        self.assertEqual((store.home()/"connections.json").stat().st_mode & 0o777, 0o600)

    def test_connection_uses_verified_ca_bundle_and_redacts_tls_failure(self):
        with patch.object(connections.urllib.request, "urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = b'{"ok":true}'
            self.assertEqual(connections.http_json("https://oauth2.googleapis.com/token"), {"ok": True})
            context = request.call_args.kwargs["context"]
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
            self.assertGreater(context.cert_store_stats()["x509_ca"], 0)
            request.side_effect = urllib.error.URLError(ssl.SSLCertVerificationError("private request detail"))
            with self.assertRaisesRegex(RuntimeError, "certificate verification failed") as error:
                connections.http_json("https://oauth2.googleapis.com/token", {"client_secret": "secret"})
            self.assertNotIn("private request detail", str(error.exception))
            self.assertNotIn("secret", str(error.exception))

    def test_oauth_tabs_and_replay_do_not_invalidate_other_attempts(self):
        connections.configure({"google_client_id": "fixture", "google_client_secret": "fixture-secret"})
        first = parse_qs(urlsplit(connections.google_start(8642)["url"]).query)["state"][0]
        second = parse_qs(urlsplit(connections.google_start(8642)["url"]).query)["state"][0]
        with self.assertRaises(ValueError):
            connections.google_callback({"state": ["unrelated"], "code": ["code"]})
        self.assertTrue(connections.google_has_pending(first))
        self.assertTrue(connections.google_has_pending(second))
        with patch.object(connections, "http_json", return_value={"access_token": "fixture-token", "expires_in": 3600}) as exchange:
            connections.google_callback({"state": [first], "code": ["first-code"]})
            self.assertFalse(connections.google_has_pending(first))
            self.assertTrue(connections.google_has_pending(second))
            with self.assertRaises(ValueError):
                connections.google_callback({"state": [first], "code": ["first-code"]})
            self.assertEqual(exchange.call_count, 1)
            connections.google_callback({"state": [second], "code": ["second-code"]})
            self.assertEqual(exchange.call_count, 2)
        self.assertTrue(connections.status()["google_connected"])

    def test_account_and_authenticator(self):
        auth.create("owner", "long-test-password")
        session = auth.login({"username": "owner", "password": "long-test-password"})
        self.assertEqual(auth.SESSIONS[session]["user_id"], "local")
        setup = auth.setup_totp("owner")
        auth.confirm_totp("owner", auth.totp(setup["secret"]))
        with self.assertRaises(PermissionError):
            auth.login({"username": "owner", "password": "long-test-password"})
        self.assertTrue(auth.login({"username": "owner", "password": "long-test-password", "code": auth.totp(setup["secret"])}))

    def test_http_persistence_origin_checks_and_account_gate(self):
        server = make_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        def request(path, data=None, headers=None):
            hdr = {"Content-Type": "application/json", **(headers or {})}
            req = urllib.request.Request(base+path, data=json.dumps(data).encode() if data is not None else None, headers=hdr)
            with urllib.request.urlopen(req) as response:
                return json.load(response)
        try:
            item = request("/api/workspace/items", {"kind": "note", "title": "From HTTP", "content": "Persists"})
            self.assertEqual(request("/api/workspace/items/"+item["id"])["content"], "Persists")
            with self.assertRaises(urllib.error.HTTPError) as err:
                request("/api/workspace/items", {"kind": "note", "title": "Blocked"}, {"Origin": "https://attacker.example"})
            self.assertEqual(err.exception.code, 403)
            request("/api/auth/create", {"username": "owner", "password": "long-test-password"})
            with self.assertRaises(urllib.error.HTTPError) as err:
                request("/api/workspace/items")
            self.assertEqual(err.exception.code, 401)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_mcp_stdio_handshake_and_discovery(self):
        from workspace.mcp import Session
        import sys
        script = Path(self.temp.name)/"mcp_fixture.py"
        script.write_text('import sys,json\nfor line in sys.stdin:\n d=json.loads(line)\n if "id" in d:\n  result={"tools":[{"name":"fixture"}]} if d["method"]=="tools/list" else {"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"fixture","version":"1"}}\n  print(json.dumps({"jsonrpc":"2.0","id":d["id"],"result":result}),flush=True)\n')
        with Session({"command": [sys.executable, str(script)]}, store.home()) as session:
            self.assertEqual(session.call("tools/list")["tools"][0]["name"], "fixture")


if __name__ == "__main__":
    unittest.main()
