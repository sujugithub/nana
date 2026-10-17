import json
import unittest
from unittest.mock import patch
from config import settings
from generation_context import TOOLS, MESSAGES
from generation_stream import CURRENT, Stream
from workspace.contracts import catalog
from workspace.tools import TOOLS as REGISTRY
import ollama_client
from remote_client import StrongRemoteClient


class Response:
    status_code = 200
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def json(self):
        return {"choices": [{"message": {"content": None, "tool_calls": [{"id": "call-fixture", "function": {"name": "read_item", "arguments": '{"id":"fixture"}'}}]}}], "usage": {"prompt_tokens": 12, "completion_tokens": 5}}
    def iter_lines(self, **kwargs):
        for data in [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-fixture", "function": {"name": "read_item", "arguments": '{"id":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"fixture"}'}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}},
        ]:
            yield b"data: " + json.dumps(data).encode()
        yield b"data: [DONE]"


class NativeAdapters(unittest.TestCase):
    def setUp(self):
        self.token = TOOLS.set(catalog(["read_item"], REGISTRY))
    def tearDown(self):
        TOOLS.reset(self.token)

    def test_ollama_catalog_and_normalized_calls(self):
        reply = {"message": {"content": "", "tool_calls": [{"function": {"name": "read_item", "arguments": {"id": "fixture"}}}]}, "prompt_eval_count": 12, "eval_count": 5}
        with patch.object(ollama_client, "request", return_value=reply) as request:
            result = ollama_client.generate("fixture", "Read a note")
        self.assertEqual(request.call_args.args[1]["tools"], TOOLS.get())
        self.assertEqual(result.tool_calls[0]["function"]["arguments"], {"id": "fixture"})
        self.assertEqual(result.total_tokens, 17)

    def test_ollama_translates_history_without_mutating_shared_context(self):
        original = [{"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "read_item", "arguments": '{"id":"fixture"}'}}]}]
        token = MESSAGES.set(original)
        try:
            with patch.object(ollama_client, "request", return_value={"message": {"content": "Finished"}}) as request:
                ollama_client.generate("fixture", "x")
            self.assertEqual(request.call_args.args[1]["messages"][-1]["tool_calls"][0]["function"]["arguments"], {"id": "fixture"})
            self.assertIsInstance(original[0]["tool_calls"][0]["function"]["arguments"], str)
        finally:
            MESSAGES.reset(token)

    def test_remote_catalog_and_fragmented_tool_calls(self):
        with patch.object(settings, "mock_mode", False), patch("requests.post", return_value=Response()) as post:
            result = StrongRemoteClient(api_key="fixture").generate("Read a note")
            self.assertEqual(post.call_args.kwargs["json"]["tools"], TOOLS.get())
            self.assertEqual(result.tool_calls[0]["id"], "call-fixture")
            token = CURRENT.set(Stream(lambda event: None))
            try:
                result = StrongRemoteClient(api_key="fixture").generate("Read a note")
                self.assertEqual(json.loads(result.tool_calls[0]["function"]["arguments"]), {"id": "fixture"})
                self.assertEqual(result.tool_calls[0]["id"], "call-fixture")
                self.assertEqual(result.total_tokens, 17)
            finally:
                CURRENT.reset(token)


if __name__ == "__main__":
    unittest.main()
