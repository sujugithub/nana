"""Deployment facts must reach Ollama in both live generation paths."""
import io
import json
import unittest
from unittest.mock import patch

from config import settings
from generation_stream import CURRENT, Stream
import ollama_client


class OllamaContextTests(unittest.TestCase):
    def test_nonstreaming_uses_selected_model_and_preserves_custom_instructions(self):
        with patch.object(settings, "ollama_base_url", "http://127.0.0.1:11435"), patch.object(
            settings, "system_prompt", "Keep the application's tool protocol."
        ), patch("ollama_client.request", return_value={"message": {"content": "Hello"}}) as request:
            ollama_client.generate("qwen3.5:4b", "Where are you running?")
        payload = request.call_args.args[1]
        system = payload["messages"][0]["content"]
        self.assertIn("Keep the application's tool protocol.", system)
        self.assertIn("selected model ID is qwen3.5:4b", system)
        self.assertIn("generated locally", system)
        self.assertIn("not a cloud API call", system)
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "Where are you running?"})

    def test_nonloopback_ollama_does_not_claim_to_run_on_this_computer(self):
        with patch.object(settings, "ollama_base_url", "http://192.0.2.5:11434"):
            system = ollama_client.local_system_prompt("another-model")
        self.assertIn("selected model ID is another-model", system)
        self.assertIn("physical location and model storage are not known", system)
        self.assertNotIn("generated locally", system)

    def test_streaming_sends_the_same_runtime_context(self):
        events = []
        token = CURRENT.set(Stream(events.append))
        response = io.BytesIO(json.dumps({"message": {"content": "Local"}, "done": True}).encode() + b"\n")
        try:
            with patch.object(settings, "ollama_base_url", "http://localhost:11435"), patch(
                "urllib.request.urlopen", return_value=response
            ) as urlopen:
                result = ollama_client.generate("qwen3.5:4b", "Are you local?")
            payload = json.loads(urlopen.call_args.args[0].data)
            self.assertTrue(payload["stream"])
            self.assertIn("generated locally", payload["messages"][0]["content"])
            self.assertEqual(result.text, "Local")
            self.assertEqual([event["text"] for event in events if event["type"] == "delta"], ["Local"])
        finally:
            CURRENT.reset(token)
