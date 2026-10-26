"""Router choices are user-scoped, honest about preview/fallback, and bypassed
by explicit model modes. No actual model/provider requests in these tests.
"""
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config import ROUTE_LOCAL, ROUTE_REMOTE, settings
from generation_context import MESSAGES
from generation_stream import CURRENT, Stream, Cancelled
from schemas import Task
from workspace import store, runs
from chat.service import conversation_config
from webui import service
from routing.arch_router import ArchRouter


class RouterSettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = patch.object(store, "ROOT", Path(self.tmp.name))
        self.root.start()
        self.user = store.USER.set("local")
        self.saved = {k: getattr(settings, k) for k in (
            "router_mode", "mock_mode", "allowed_models", "usage_log_path", "ollama_base_url")}
        settings.router_mode = "heuristic"
        settings.mock_mode = False
        settings.allowed_models = ""
        settings.usage_log_path = ""
        settings.ollama_base_url = "http://127.0.0.1:11435"

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(settings, k, v)
        store.USER.reset(self.user)
        self.root.stop()
        self.tmp.cleanup()

    def test_choices_persist_and_reach_chat_and_agents(self):
        for choice in ("heuristic", "learned", "arch"):
            store.preferences({"router_kind": choice})
            self.assertEqual(store.preferences()["router_kind"], choice)
            self.assertEqual(conversation_config("hybrid", None, True).router_kind, choice)
            with patch("workspace.runs.execute", return_value={"ok": True}) as execute:
                runs.generate({"mode": "hybrid", "mock": True}, "hello", False)
            self.assertEqual(execute.call_args.args[0].router_kind, choice)

    def test_preferences_are_user_scoped(self):
        store.preferences({"router_kind": "arch", "system_prompt": "Keep this"})
        token = store.USER.set("other")
        try:
            self.assertNotIn("router_kind", store.preferences())
            self.assertEqual(conversation_config("hybrid", None, True).router_kind, "heuristic")
            store.preferences({"router_kind": "learned"})
        finally:
            store.USER.reset(token)
        self.assertEqual(store.preferences()["router_kind"], "arch")
        self.assertEqual(store.preferences()["system_prompt"], "Keep this")

    def test_invalid_preference_is_rejected_atomically(self):
        store.preferences({"theme": "light", "router_kind": "heuristic"})
        for value in ("auto", "invalid", None, [], 3):
            with self.assertRaises(ValueError):
                store.preferences({"theme": "dark", "router_kind": value})
            self.assertEqual(store.preferences()["theme"], "light")

    def test_factory_and_config_accept_arch(self):
        import main
        settings.router_mode = "arch"
        self.assertIsInstance(main.build_router(), ArchRouter)
        cfg, errors = service.config_from_payload({"router_kind": "arch"})
        self.assertEqual(errors, [])
        self.assertEqual(cfg.router_kind, "arch")

    def test_readiness_never_generates_or_downloads(self):
        with patch("ollama_client.request", side_effect=AssertionError("no model calls")), patch(
            "webui.service.artifact_status", return_value={"available": False, "detail": "missing"}):
            options = service.router_options(service.DemoConfig(), [{"name": settings.arch_router_model}])
        self.assertEqual([o["value"] for o in options], ["heuristic", "learned", "arch"])
        self.assertFalse(options[1]["available"])
        self.assertTrue(options[2]["available"])

    def test_mock_arch_is_labelled_and_does_not_contact_ollama(self):
        cfg = service.DemoConfig(router_kind="arch", mock=True)
        with patch("ollama_client.request", side_effect=AssertionError("preview model call")):
            result = service.execute(cfg, "hello")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["routing"]["kind"], "arch-preview")
        self.assertTrue(any("Preview" in s for s in result["warnings"]))

    def test_chat_response_preserves_preview_and_fallback_warnings(self):
        from chat import database, repository
        from chat.service import send_message
        from router import Router
        store.preferences({"router_kind": "arch"})
        with patch.object(database, "DB_PATH", Path(self.tmp.name) / "chat.db"):
            database.initialise_database()
            conversation = repository.create_conversation(mode="hybrid")
            with patch("ollama_client.request", side_effect=AssertionError("no inference")):
                result = send_message(conversation.id, "Hello", mock=True)
            self.assertEqual(result["routing"]["router"], "arch-preview")
            self.assertTrue(any("Preview" in warning for warning in result["warnings"]))
            fallback = Router().decide(Task("x", "Hello"))
            fallback.router_kind = "heuristic-fallback"
            fallback.reason = "Arch-Router unavailable; used heuristic fallback."
            with patch.object(ArchRouter, "decide", return_value=fallback):
                result = send_message(conversation.id, "Hello again", mock=True)
            self.assertEqual(result["routing"]["router"], "heuristic-fallback")
            self.assertIn(fallback.reason, result["warnings"])

    def test_explicit_modes_bypass_every_router(self):
        for mode in ("local_only", "remote_only"):
            for choice in ("heuristic", "learned", "arch"):
                with patch("main.build_router", side_effect=AssertionError("routing must be bypassed")), patch(
                    "ollama_client.request", side_effect=AssertionError("no router model call")):
                    result = service.execute(service.DemoConfig(mode=mode, router_kind=choice, mock=True), "hi")
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["routing"]["kind"], "bypassed")

    def test_route_parses_json_and_python_dict_without_tools(self):
        for raw, expected in [('{"route":"local"}', ROUTE_LOCAL), ("{'route': 'strong'}", ROUTE_REMOTE)]:
            with patch("ollama_client.request", return_value={"message": {"content": raw}, "prompt_eval_count": 40}) as request:
                d = ArchRouter().decide(Task("x", "A new request"))
            self.assertEqual(d.target, expected)
            self.assertEqual(d.router_kind, "arch")
            args, kwargs = request.call_args
            self.assertEqual(args[0], "/api/chat")
            self.assertEqual(args[1]["keep_alive"], 0)
            self.assertEqual(args[1]["options"]["num_ctx"], 2048)
            self.assertNotIn("tools", args[1])
            self.assertLessEqual(kwargs["timeout"], 15)

    def test_latest_intent_uses_bounded_messages_without_system_or_images(self):
        turns = [{"role": "system", "content": "PRIVATE SYSTEM"},
                 {"role": "user", "content": "old" * 2000},
                 {"role": "assistant", "content": "reply"},
                 {"role": "user", "content": "Name my cat", "images": ["PRIVATE IMAGE"]}]
        token = MESSAGES.set(turns)
        try:
            with patch("ollama_client.request", return_value={"message": {"content": '{"route":"local"}'}}) as request:
                ArchRouter().decide(Task("x", "legacy prompt"))
        finally:
            MESSAGES.reset(token)
        prompt = request.call_args.args[1]["messages"][0]["content"]
        conversation = json.loads(prompt.split("<conversation>\n")[1].split("\n</conversation>")[0])
        self.assertEqual(conversation[-1]["content"], "Name my cat")
        self.assertLessEqual(sum(len(t["content"]) for t in conversation), 2000)
        self.assertNotIn("PRIVATE", prompt)

    def test_malformed_unmatched_or_unavailable_router_reports_fallback(self):
        responses = [RuntimeError("model missing"), {}, {"message": {"content": None}},
                     {"message": {"content": "not a route"}},
                     {"message": {"content": '{"route":"other"}'}},
                     {"message": {"content": '{"route":"local"}'}, "done": False}]
        for response in responses:
            kw = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
            with patch("ollama_client.request", **kw), redirect_stderr(io.StringIO()) as stderr:
                d = ArchRouter().decide(Task("x", "Hello"))
            self.assertEqual(d.router_kind, "heuristic-fallback")
            self.assertIn("fallback", d.reason)
            self.assertIn("WARNING", stderr.getvalue())

    def test_remote_ollama_address_does_not_receive_router_context(self):
        settings.ollama_base_url = "https://example.org"
        with patch("ollama_client.request", side_effect=AssertionError("do not send context")), redirect_stderr(io.StringIO()):
            self.assertEqual(ArchRouter().decide(Task("x", "Hello")).router_kind, "heuristic-fallback")

    def test_cancellation_does_not_trigger_fallback_generation(self):
        stream = Stream(lambda e: None)
        token = CURRENT.set(stream)
        try:
            def response(*a, **kw):
                stream.cancel.set()
                return {"message": {"content": '{"route":"local"}'}}
            with patch("ollama_client.request", side_effect=response):
                with self.assertRaises(Cancelled):
                    ArchRouter().decide(Task("x", "Hello"))
        finally:
            CURRENT.reset(token)
