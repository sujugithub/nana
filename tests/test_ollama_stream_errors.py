"""Ollama streaming failures and installed-model chat choices."""
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from generation_stream import Stream
from ollama_client import stream_generate
from webui.server import make_server


class TestOllamaStreamErrors(unittest.TestCase):
    def test_missing_model_http_error_is_not_reported_as_connection_failure(self):
        failure = urllib.error.HTTPError(
            "http://127.0.0.1:11435/api/chat", 404, "Not Found", {}, io.BytesIO(b"private detail")
        )
        with mock.patch("urllib.request.urlopen", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "model 'missing:latest' is not installed") as caught:
                stream_generate("missing:latest", [{"role": "user", "content": "disposable"}],
                                Stream(lambda event: None), 0)
        self.assertNotIn("private detail", str(caught.exception))

    def test_other_http_error_reports_status_without_exposing_response(self):
        failure = urllib.error.HTTPError(
            "http://127.0.0.1:11435/api/chat", 503, "Unavailable", {}, io.BytesIO(b"private detail")
        )
        with mock.patch("urllib.request.urlopen", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503") as caught:
                stream_generate("installed:latest", [{"role": "user", "content": "disposable"}],
                                Stream(lambda event: None), 0)
        self.assertNotIn("private detail", str(caught.exception))

    def test_chat_picker_lists_only_installed_ollama_models(self):
        server = make_server(port=0, allow_real=False)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/api/chat/config"
        prefs = {"local_backend": "ollama", "local_model": "uninstalled:model"}
        tags = {"models": [{"name": "qwen3.5:4b"}, {"name": "qwen3.8:27b"}]}
        with mock.patch("workspace.store.preferences", return_value=prefs), mock.patch(
            "ollama_client.request", return_value=tags
        ):
            with urllib.request.urlopen(url) as response:
                config = json.load(response)
        self.assertEqual(config["known_local_models"], ["qwen3.5:4b", "qwen3.8:27b"])
        self.assertEqual(config["default_models"]["local"], "qwen3.5:4b")


if __name__ == "__main__":
    unittest.main()
