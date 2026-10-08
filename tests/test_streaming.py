"""Provider token streaming preserves atomic turns and usage accounting."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request

from config import settings
from generation_stream import CURRENT, Stream, Cancelled
from remote_client import StrongRemoteClient
from webui.server import make_server
from workspace import store


class Response:
    status_code = 200
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def iter_lines(self, **kwargs):
        for data in [{"choices": [{"delta": {"content": "Hello"}}]},
                     {"choices": [{"delta": {"content": " world"}}]},
                     {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}]:
            yield b"data: " + json.dumps(data).encode()
        yield b"data: [DONE]"


class StreamingTests(unittest.TestCase):
    def test_live_deltas_and_provider_usage(self):
        events = []
        token = CURRENT.set(Stream(events.append))
        try:
            with patch.object(settings, "mock_mode", False), patch("requests.post", return_value=Response()):
                completion = StrongRemoteClient(api_key="fixture").generate("Hello")
            self.assertEqual(completion.text, "Hello world")
            self.assertEqual((completion.prompt_tokens, completion.completion_tokens), (7, 2))
            self.assertEqual([e["text"] for e in events if e["type"] == "delta"], ["Hello", " world"])
        finally:
            CURRENT.reset(token)

    def test_stop_propagates_instead_of_triggering_paid_fallback(self):
        stream = Stream(lambda event: stream.cancel.set() if event["type"] == "delta" else None)
        token = CURRENT.set(stream)
        try:
            with patch.object(settings, "mock_mode", False), patch("requests.post", return_value=Response()) as post:
                with self.assertRaises(Cancelled):
                    StrongRemoteClient(api_key="fixture").generate("Hello")
                self.assertEqual(post.call_count, 1)
        finally:
            CURRENT.reset(token)

    def test_sse_preview_turn_saves_both_messages(self):
        from chat import database
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(database, "DB_PATH", Path(directory)/"chat.db"), patch.object(store, "ROOT", Path(directory)/"workspace"):
                server = make_server(allow_real=False)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_address[1]}"
                def post(path, payload):
                    req = urllib.request.Request(base+path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                    return urllib.request.urlopen(req)
                try:
                    with post("/conversations", {"mode": "remote", "title": "Streaming"}) as response:
                        item = json.load(response)
                    with post(f"/conversations/{item['id']}/stream", {"content": "Hello", "mock": True, "stream_id": "a"*32}) as response:
                        frames = [json.loads(frame[6:]) for frame in response.read().decode().strip().split("\n\n")]
                    self.assertEqual(frames[-1]["type"], "complete")
                    with urllib.request.urlopen(base+f"/conversations/{item['id']}") as response:
                        self.assertEqual(len(json.load(response)["messages"]), 2)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join()


if __name__ == "__main__":
    unittest.main()
