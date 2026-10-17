"""Completed output downloads, format intent, isolation and actual file bytes."""
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from chat import artifacts, database, repository, service
from generation_context import completion_messages, MESSAGES
from webui.server import make_server
from workspace import store

HTML = '<!doctype html>\n<html><head><title>Snake</title></head><body><canvas id="game"></canvas><script>const score = "🐍 17";</script></body></html>\n'
ANSWER = 'Here is the game.\n\n```html\n' + HTML + '```\n\nOpen the downloaded file in your browser.'
ESSAY = '# Why Learning Matters\n\nLearning builds **confidence** and curiosity.\n\n## Practising Often\n\nSmall steps make progress. A & B < C.\n\n- Read daily\n- Reflect on mistakes\n'


class ChatArtifactTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for target, key, value in [(store, 'ROOT', root / 'workspace'), (database, 'DB_PATH', root / 'chat.db')]:
            p = patch.object(target, key, value)
            p.start(); self.addCleanup(p.stop)
        token = store.USER.set('local')
        self.addCleanup(store.USER.reset, token)
        database.initialise_database()
        self.chat = repository.create_conversation(mode='local', model='qwen3.5:4b')

    def test_html_bytes_match_code_and_survive_reload(self):
        _, reply = repository.add_turn(self.chat.id, 'Make me a snake game in HTML', ANSWER)
        self.assertEqual(reply.artifacts[0]['name'], 'snake-game.html')
        raw, mime, name = artifacts.download(self.chat.id, reply.id, 0)
        self.assertEqual(raw.decode(), HTML)
        self.assertTrue(mime.startswith('text/html'))
        self.assertEqual(asdict(repository.list_messages(self.chat.id)[1])['artifacts'], reply.artifacts)
        self.assertEqual(name, 'snake-game.html')

    def test_word_download_is_real_ooxml_with_title_and_escaped_text(self):
        _, reply = repository.add_turn(self.chat.id, 'Write an essay in docx', ESSAY)
        raw, mime, name = artifacts.download(self.chat.id, reply.id, 0)
        self.assertTrue(raw.startswith(b'PK'))
        self.assertTrue(name.endswith('.docx'))
        self.assertIn('wordprocessingml.document', mime)
        with ZipFile(BytesIO(raw)) as z:
            for filename in z.namelist(): ET.fromstring(z.read(filename))
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            document = ET.fromstring(z.read('word/document.xml'))
            self.assertEqual(document.find('.//w:pStyle', ns).attrib['{' + ns['w'] + '}val'], 'Title')
            words = ''.join(document.itertext())
            self.assertIn('Learning builds confidence', words)
            self.assertIn('A & B < C', words)
            self.assertNotIn('**', words)

    def test_simple_questions_essays_and_format_explanations_do_not_get_cards(self):
        for prompt in ['Hello', 'Write me an essay', 'What is DOCX?', 'Explain HTML', 'How do I create a DOCX file?', 'Write a short summary of the uploaded report.docx']:
            self.assertEqual(artifacts.describe(prompt, 'Ordinary answer.'), [], prompt)
        self.assertEqual(artifacts.requested_format('What is DOCX?'), None)
        self.assertEqual(artifacts.requested_format('Write a short summary of the uploaded report.docx'), None)
        self.assertEqual(artifacts.describe('Make snake in HTML, no downloads', ANSWER), [])

    def test_mock_refusal_and_incomplete_code_do_not_claim_finished_files(self):
        self.assertEqual(artifacts.describe('Write essay in docx', '[mock-local] Fake essay'), [])
        self.assertEqual(artifacts.describe('Write essay in docx', "I can't create Word files."), [])
        self.assertEqual(artifacts.describe('Make HTML', '```html\n<html>unfinished'), [])

    def test_multiple_code_files_preserve_source_and_language(self):
        prompt = 'Create index.html and app.js'
        reply = '```html filename=index.html\n<p>Snake</p>\n```\n```javascript filename=app.js\nconst x = 1;\n```'
        _, message = repository.add_turn(self.chat.id, prompt, reply)
        self.assertEqual([f['name'] for f in message.artifacts], ['index.html', 'app.js'])
        self.assertEqual(artifacts.download(self.chat.id, message.id, 1)[0], b'const x = 1;\n')

    def test_four_backtick_fence_keeps_inner_fences_and_unknown_languages(self):
        answer = '````text\nExample: ```html\nhello\n```\n````\n~~~unknown\nraw data\n~~~'
        _, reply = repository.add_turn(self.chat.id, 'Give code', answer)
        self.assertEqual(len(reply.artifacts), 2)
        self.assertIn(b'```html', artifacts.download(self.chat.id, reply.id, 0)[0])
        self.assertTrue(reply.artifacts[1]['name'].endswith('.txt'))

    def test_named_path_cannot_escape_and_raw_complete_html_is_supported(self):
        _, reply = repository.add_turn(self.chat.id, 'Make HTML save ../../game.html', HTML)
        name = artifacts.download(self.chat.id, reply.id, 0)[2]
        self.assertNotIn('/', name)
        self.assertNotIn('..', name)
        self.assertEqual(artifacts.download(self.chat.id, reply.id, 0)[0].decode(), HTML)

    def test_download_requires_correct_conversation_assistant_and_existing_file(self):
        user, assistant = repository.add_turn(self.chat.id, 'Make HTML', ANSWER)
        other = repository.create_conversation()
        for cid, mid, index in [(other.id, assistant.id, 0), (self.chat.id, user.id, 0), (self.chat.id, assistant.id, 99), (self.chat.id, assistant.id, -1)]:
            with self.assertRaises(LookupError): artifacts.download(cid, mid, index)
        repository.delete_conversation(self.chat.id)
        with self.assertRaises(LookupError): artifacts.download(self.chat.id, assistant.id, 0)

    def test_cross_account_reply_cannot_be_downloaded(self):
        _, assistant = repository.add_turn(self.chat.id, 'Make HTML', ANSWER)
        token = store.USER.set('guest')
        try:
            database.initialise_database()
            with self.assertRaises(LookupError): artifacts.download(self.chat.id, assistant.id, 0)
        finally: store.USER.reset(token)

    def test_markdown_wrapped_document_avoids_duplicate_download(self):
        meta = artifacts.describe('Give this in docx please', '```markdown\n' + ESSAY + '```')
        self.assertEqual(len(meta), 1)
        self.assertEqual(meta[0]['kind'], 'document')

    def test_docx_control_characters_are_removed_and_no_external_relationships(self):
        raw = artifacts.make_docx('# Title\n\nSafe\x00 and emoji 🐍. <script>alert(1)</script>')
        with ZipFile(BytesIO(raw)) as z:
            text = z.read('word/document.xml')
            ET.fromstring(text)
            self.assertIn(b'&lt;script&gt;', text)
            self.assertNotIn(b'TargetMode="External"', b''.join(z.read(n) for n in z.namelist()))

    def test_generation_instructions_enable_packaging_without_changing_roles(self):
        token = MESSAGES.set([{'role': 'user', 'content': 'Write essay in docx'}])
        try:
            turns = completion_messages('unused', 'Base instruction')
            self.assertIn('real DOCX download', turns[0]['content'])
            self.assertEqual(turns[-1], {'role': 'user', 'content': 'Write essay in docx'})
        finally: MESSAGES.reset(token)

    def test_download_http_is_attachment_and_static_renderer_is_served(self):
        _, assistant = repository.add_turn(self.chat.id, 'Make snake in HTML', ANSWER)
        server = make_server(0, allow_real=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_address[1]}'
            with urllib.request.urlopen(f'{base}/conversations/{self.chat.id}/messages/{assistant.id}/artifacts/0') as response:
                self.assertEqual(response.read(), HTML.encode())
                self.assertIn('attachment;', response.headers['Content-Disposition'])
                self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
            with urllib.request.urlopen(base + '/chat-response.js') as response:
                self.assertIn(b'window.NanaResponses', response.read())
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(f'{base}/conversations/{self.chat.id}/messages/{assistant.id}/artifacts/42')
            self.assertEqual(error.exception.code, 404)
            error.exception.close()
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_optional_fastapi_download_matches_native_server(self):
        try:
            from fastapi.testclient import TestClient
            from api import app
        except ImportError:
            self.skipTest('Optional FastAPI dependencies are absent')
        _, reply = repository.add_turn(self.chat.id, 'Write essay in docx', ESSAY)
        with TestClient(app) as client:
            response = client.get(f'/conversations/{self.chat.id}/messages/{reply.id}/artifacts/0')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.content.startswith(b'PK'))
            self.assertIn('attachment;', response.headers['content-disposition'])
            self.assertEqual(client.get('/chat-response.js').status_code, 200)
            self.assertEqual(client.get(f'/conversations/{self.chat.id}/messages/{reply.id}/artifacts/88').status_code, 404)
