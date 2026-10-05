"""Persistent chat behavior, including mode isolation and failure handling."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import api
from chat import database, repository, service
from config import settings
from schemas import Completion
from webui.server import make_server


class ChatCase(unittest.TestCase):
    _SETTINGS = (
        "mock_mode", "tier_mode", "router_mode", "local_model_name",
        "cheap_model_name", "strong_model_name", "confidence_threshold",
        "enable_escalation", "allowed_models", "usage_log_path",
    )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_patch = mock.patch.object(database, "DB_PATH", Path(self.temp.name) / "chat.db")
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.saved = {name: getattr(settings, name) for name in self._SETTINGS}
        self.addCleanup(self._restore_settings)
        settings.mock_mode = True
        settings.tier_mode = "remote_pair"
        settings.router_mode = "heuristic"
        settings.allowed_models = ""
        settings.usage_log_path = ""
        database.initialise_database()

    def _restore_settings(self):
        for name, value in self.saved.items():
            setattr(settings, name, value)


class TestChatService(ChatCase):
    def test_local_mode_never_calls_fireworks_and_saves_latency(self):
        conversation = repository.create_conversation(mode="local")
        with mock.patch("remote_client.FireworksClient.generate", side_effect=AssertionError("remote called")) as remote:
            result = service.send_message(conversation.id, "Hello", mock=True)
        remote.assert_not_called()
        self.assertEqual(result["routing"]["provider"], "local")
        self.assertEqual(result["routing"]["billable_tokens"], 0)
        self.assertEqual(result["assistant_message"]["estimated_cost_usd"], 0.0)
        self.assertIsInstance(result["assistant_message"]["latency_s"], float)

    def test_remote_mode_never_calls_local_backend(self):
        conversation = repository.create_conversation(mode="remote")
        with mock.patch("local_model.LocalModel.generate", side_effect=AssertionError("local called")) as local:
            result = service.send_message(conversation.id, "Hello", mock=True)
        local.assert_not_called()
        self.assertEqual(result["routing"]["provider"], "fireworks")
        self.assertTrue(result["mock"])

    def test_real_local_mode_cannot_fall_through_to_fireworks(self):
        conversation = repository.create_conversation(mode="local")
        completion = Completion(
            text="local answer", prompt_tokens=3, completion_tokens=2,
            source="local", model_name="test-local", provider="local",
        )
        with mock.patch("local_model.LocalModel.generate", return_value=completion), mock.patch(
            "remote_client.FireworksClient.generate", side_effect=AssertionError("remote called")
        ) as remote:
            result = service.send_message(conversation.id, "Hello", mock=False, allow_real=True)
        remote.assert_not_called()
        self.assertFalse(result["mock"])
        self.assertFalse(result["billable"])
        self.assertEqual(result["assistant_message"]["content"], "local answer")
        self.assertEqual(result["routing"]["billable_tokens"], 0)

    def test_real_request_is_forced_to_mock_without_server_opt_in(self):
        conversation = repository.create_conversation(mode="remote")
        result = service.send_message(conversation.id, "Hello", mock=False, allow_real=False)
        self.assertTrue(result["mock"])
        self.assertFalse(result["billable"])

    def test_failed_generation_leaves_no_partial_turn(self):
        conversation = repository.create_conversation()
        with mock.patch("chat.service.execute", return_value={"ok": False, "error": "model failed"}):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                service.send_message(conversation.id, "Keep this only on success")
        self.assertEqual(repository.list_messages(conversation.id), [])

    def test_prompt_keeps_recent_history_within_limit(self):
        conversation = repository.create_conversation()
        for number in range(5):
            repository.add_turn(conversation.id, f"turn-{number} " + "x" * 8000, "reply " + "y" * 1000)
        prompt = service.build_conversation_prompt(conversation.id, "latest question")
        self.assertLessEqual(len(prompt), service.MAX_CONTEXT_CHARS)
        self.assertIn("turn-4", prompt)
        self.assertIn("latest question", prompt)
        self.assertNotIn("turn-0", prompt)
        self.assertEqual(prompt.count("User:"), prompt.count("Assistant:"))


class TestFastAPI(ChatCase):
    def test_crud_search_and_mock_message(self):
        with TestClient(api.app) as client:
            created = client.post("/conversations", json={"title": "Review", "mode": "local"})
            self.assertEqual(created.status_code, 200)
            conversation_id = created.json()["id"]
            sent = client.post(f"/conversations/{conversation_id}/messages", json={"content": "Hello", "mock": False})
            self.assertEqual(sent.status_code, 200)
            self.assertEqual(sent.json()["routing"]["provider"], "local")
            self.assertTrue(sent.json()["mock"])
            self.assertEqual(len(client.get(f"/conversations/{conversation_id}").json()["messages"]), 2)
            self.assertEqual(len(client.get("/conversations/search?q=Review").json()), 1)
            self.assertEqual(client.patch(f"/conversations/{conversation_id}/settings", json={"mode": "remote"}).status_code, 200)
            self.assertEqual(client.delete(f"/conversations/{conversation_id}").status_code, 200)
            self.assertEqual(client.get(f"/conversations/{conversation_id}").status_code, 404)

    def test_rejects_bad_modes_models_and_blank_titles(self):
        with TestClient(api.app) as client:
            self.assertEqual(client.post("/conversations", json={"mode": "invalid"}).status_code, 422)
            self.assertEqual(client.post("/conversations", json={"title": "   "}).status_code, 400)
            settings.allowed_models = settings.strong_model_name
            self.assertEqual(client.post("/conversations", json={"mode": "remote", "model": "not-allowed"}).status_code, 400)


class TestDemoChatHTTP(ChatCase):
    def test_chat_page_and_mock_only_http_flow(self):
        server = make_server(port=0, allow_real=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def post(path, payload):
            request = urllib.request.Request(
                base + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                return json.load(response)

        with urllib.request.urlopen(base + "/chat") as response:
            self.assertIn(b"Nana chat", response.read())
        conversation = post("/conversations", {"mode": "local"})
        result = post(f"/conversations/{conversation['id']}/messages", {"content": "Hello"})
        self.assertEqual(result["routing"]["provider"], "local")
        self.assertTrue(result["mock"])
        with self.assertRaises(urllib.error.HTTPError) as error:
            post(f"/conversations/{conversation['id']}/messages", {"content": "Hello", "mock": False})
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(len(repository.list_messages(conversation["id"])), 2)
