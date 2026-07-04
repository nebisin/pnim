import importlib
import sqlite3
import sys
import tempfile
import types
import unittest
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
        self.embed_response = FakeEmbedResponse([[0.1, 0.2, 0.3]])
        self.embed_error = None

    def chat(self, **kwargs):
        return FakeChatResponse("# page text")

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


if __name__ == "__main__":
    unittest.main()
