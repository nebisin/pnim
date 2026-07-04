import http.client
import importlib
import json
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock


class FakeResponseError(Exception):
    def __init__(self, error: str, status_code: int = 500):
        super().__init__(error)
        self.error = error
        self.status_code = status_code


class FakeChatResponse:
    def __init__(self, content: str):
        self.message = types.SimpleNamespace(content=content)


class FakeEmbedResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeClient:
    def __init__(self, host=None):
        self.host = host
        self.chat_content = "# page text"
        self.last_chat_kwargs = None
        self.embed_response = FakeEmbedResponse([[0.1, 0.2, 0.3]])
        self.embed_error = None

    def chat(self, **kwargs):
        self.last_chat_kwargs = kwargs
        return FakeChatResponse(self.chat_content)

    def embed(self, **kwargs):
        if self.embed_error is not None:
            raise self.embed_error
        return self.embed_response


def load_watch_consume():
    fake_ollama = types.ModuleType("ollama")
    fake_ollama.Client = FakeClient
    fake_ollama.ResponseError = FakeResponseError

    fake_fitz = types.ModuleType("fitz")
    fake_fitz.Matrix = lambda x, y: (x, y)
    fake_fitz.open = lambda path: None

    with mock.patch.dict(sys.modules, {"ollama": fake_ollama, "fitz": fake_fitz}):
        sys.modules.pop("watch_consume", None)
        return importlib.import_module("watch_consume")


class WatchConsumeTests(unittest.TestCase):
    def setUp(self):
        self.module = load_watch_consume()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.module.CONSUME_DIR = root / "consume"
        self.module.IMAGES_DIR = root / "images"
        self.module.TEXTS_DIR = root / "texts"
        self.module.EMBEDDINGS_DB_PATH = root / "text_embeddings.sqlite3"
        self.module.OLLAMA_CLIENT = FakeClient(host="http://localhost:11434")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_embed_text_returns_first_embedding(self):
        vector = self.module.embed_text("hello")
        self.assertEqual(vector, [0.1, 0.2, 0.3])

    def test_embed_text_wraps_response_error(self):
        self.module.OLLAMA_CLIENT.embed_error = FakeResponseError("missing model", 404)

        with self.assertRaisesRegex(RuntimeError, "missing model"):
            self.module.embed_text("hello")

    def test_extract_embedding_accepts_dict_payload(self):
        vector = self.module.extract_embedding({"embeddings": [0.4, 0.5]})
        self.assertEqual(vector, [0.4, 0.5])

    def test_extract_embedding_rejects_invalid_shape(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported embedding shape"):
            self.module.extract_embedding({"embeddings": ["bad"]})

    def test_extract_chat_content_accepts_dict_payload(self):
        content = self.module.extract_chat_content({"message": {"content": " hello "}})
        self.assertEqual(content, "hello")

    def test_extract_chat_content_accepts_object_payload(self):
        content = self.module.extract_chat_content(FakeChatResponse(" hello "))
        self.assertEqual(content, "hello")

    def test_extract_chat_content_rejects_invalid_payload(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported chat response payload"):
            self.module.extract_chat_content({"message": {"content": None}})

    def test_initialize_database_is_idempotent(self):
        self.module.initialize_embedding_database()
        self.module.initialize_embedding_database()

        with sqlite3.connect(self.module.EMBEDDINGS_DB_PATH) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(text_embeddings)").fetchall()
            }

        self.assertTrue(
            {"source_pdf", "page_number", "text_path", "embedding_json"}.issubset(columns)
        )

    def test_save_embedding_upserts_text_path(self):
        self.module.ensure_directories()
        pdf_path = self.module.CONSUME_DIR / "sample.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4")

        text_path = self.module.write_markdown(pdf_path, 1, "first")
        self.module.save_embedding(pdf_path, 1, text_path, "first", [0.1, 0.2])
        self.module.save_embedding(pdf_path, 2, text_path, "second", [0.3, 0.4])

        with sqlite3.connect(self.module.EMBEDDINGS_DB_PATH) as connection:
            row = connection.execute(
                """
                SELECT page_number, content, embedding_json
                FROM text_embeddings
                WHERE text_path = ?
                """,
                (text_path.as_posix(),),
            ).fetchone()
            count = connection.execute(
                "SELECT COUNT(*) FROM text_embeddings WHERE text_path = ?",
                (text_path.as_posix(),),
            ).fetchone()[0]

        self.assertEqual(count, 1)
        self.assertEqual(row, (2, "second", "[0.3, 0.4]"))

    def test_semantic_search_returns_ranked_matches(self):
        self.module.ensure_directories()
        pdf_path = self.module.CONSUME_DIR / "sample.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4")

        first_path = self.module.write_markdown(pdf_path, 1, "alpha")
        second_path = self.module.write_markdown(pdf_path, 2, "beta")
        self.module.save_embedding(pdf_path, 1, first_path, "alpha", [1.0, 0.0])
        self.module.save_embedding(pdf_path, 2, second_path, "beta", [0.0, 1.0])

        with mock.patch.object(self.module, "embed_text", return_value=[1.0, 0.0]):
            results = self.module.semantic_search("find alpha", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "alpha")
        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_semantic_search_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            self.module.semantic_search("   ")

        with self.assertRaisesRegex(ValueError, "limit must be at least 1"):
            self.module.semantic_search("alpha", limit=0)

    def test_semantic_search_returns_empty_when_model_has_no_rows(self):
        self.module.ensure_directories()
        pdf_path = self.module.CONSUME_DIR / "sample.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4")

        text_path = self.module.write_markdown(pdf_path, 1, "alpha")
        self.module.save_embedding(pdf_path, 1, text_path, "alpha", [1.0, 0.0])
        self.module.EMBEDDING_MODEL = "other-model"

        with mock.patch.object(self.module, "embed_text", return_value=[1.0, 0.0]):
            results = self.module.semantic_search("find alpha", limit=1)

        self.assertEqual(results, [])

    def test_generate_rag_answer_uses_rag_model(self):
        self.module.OLLAMA_CLIENT.chat_content = "grounded answer"

        answer = self.module.generate_rag_answer(
            "What is alpha?",
            [
                {
                    "source_pdf": "sample.pdf",
                    "page_number": 1,
                    "text_path": "texts/sample_page_0001.md",
                    "content": "alpha content",
                    "score": 0.9,
                }
            ],
        )

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(self.module.OLLAMA_CLIENT.last_chat_kwargs["model"], self.module.RAG_MODEL)
        self.assertIn("alpha content", self.module.OLLAMA_CLIENT.last_chat_kwargs["messages"][1]["content"])

    def test_handle_search_request_validates_query(self):
        with self.assertRaisesRegex(ValueError, "query must be a non-empty string"):
            self.module.handle_search_request({"query": ""})

    def test_handle_search_request_returns_results_and_answer(self):
        with (
            mock.patch.object(self.module, "semantic_search", return_value=[{"content": "alpha"}]),
            mock.patch.object(self.module, "generate_rag_answer", return_value="answer"),
        ):
            response = self.module.handle_search_request({"query": "alpha"})

        self.assertEqual(response, {"query": "alpha", "results": [{"content": "alpha"}], "answer": "answer"})


class SearchApiHandlerTests(unittest.TestCase):
    TEST_HOST = "127.0.0.1"

    @classmethod
    def setUpClass(cls):
        cls.module = load_watch_consume()
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        cls.module.CONSUME_DIR = root / "consume"
        cls.module.IMAGES_DIR = root / "images"
        cls.module.TEXTS_DIR = root / "texts"
        cls.module.EMBEDDINGS_DB_PATH = root / "text_embeddings.sqlite3"
        cls.module.OLLAMA_CLIENT = FakeClient(host="http://localhost:11434")
        cls.server = ThreadingHTTPServer((cls.TEST_HOST, 0), cls.module.SearchApiHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.temp_dir.cleanup()

    def _post(self, path, body=None, raw_body=None):
        conn = http.client.HTTPConnection(self.TEST_HOST, self.port)
        if raw_body is not None:
            encoded = raw_body
        elif body is not None:
            encoded = json.dumps(body).encode()
        else:
            encoded = b""
        conn.request(
            "POST",
            path,
            body=encoded,
            headers={"Content-Length": str(len(encoded)), "Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        status = resp.status
        raw = resp.read()
        conn.close()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        return status, data

    def test_unknown_path_returns_404(self):
        status, _ = self._post("/not/found")
        self.assertEqual(status, 404)

    def test_invalid_json_returns_400(self):
        status, data = self._post("/api/search", raw_body=b"not json {")
        self.assertEqual(status, 400)
        self.assertIn("valid JSON", data["error"])

    def test_missing_query_returns_400(self):
        status, data = self._post("/api/search", body={})
        self.assertEqual(status, 400)
        self.assertIn("query", data["error"])

    def test_limit_too_large_returns_400(self):
        status, data = self._post("/api/search", body={"query": "test", "limit": 100})
        self.assertEqual(status, 400)
        self.assertIn("limit", data["error"])

    def test_include_answer_false_omits_answer(self):
        with mock.patch.object(self.module, "semantic_search", return_value=[{"content": "c"}]):
            status, data = self._post("/api/search", body={"query": "test", "include_answer": False})
        self.assertEqual(status, 200)
        self.assertNotIn("answer", data)
        self.assertEqual(data["results"], [{"content": "c"}])

    def test_runtime_error_returns_502(self):
        with mock.patch.object(
            self.module, "handle_search_request", side_effect=RuntimeError("db failure")
        ):
            status, data = self._post("/api/search", body={"query": "test"})
        self.assertEqual(status, 502)
        self.assertIn("db failure", data["error"])

    def test_success_returns_200_with_results_and_answer(self):
        results = [{"content": "alpha"}]
        with (
            mock.patch.object(self.module, "semantic_search", return_value=results),
            mock.patch.object(self.module, "generate_rag_answer", return_value="answer text"),
        ):
            status, data = self._post("/api/search", body={"query": "alpha query"})
        self.assertEqual(status, 200)
        self.assertEqual(data["query"], "alpha query")
        self.assertEqual(data["results"], results)
        self.assertEqual(data["answer"], "answer text")


if __name__ == "__main__":
    unittest.main()
