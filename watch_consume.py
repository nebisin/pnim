from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable

from ollama import Client, ResponseError

try:
    import fitz  # type: ignore[import-not-found]  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyMuPDF is required. Install it with: pip install -r requirements.txt"
    ) from exc

CONSUME_DIR = Path("consume")
IMAGES_DIR = Path("images")
TEXTS_DIR = Path("texts")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-ocr:latest")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "embeddinggemma:latest")
RAG_MODEL = os.environ.get("RAG_MODEL", "gemma4:26b")
EMBEDDINGS_DB_PATH = Path(os.environ.get("EMBEDDINGS_DB_PATH", "text_embeddings.sqlite3"))
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "2"))
STABLE_POLLS_REQUIRED = int(os.environ.get("STABLE_POLLS_REQUIRED", "2"))
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8080"))
APP_MODE = os.environ.get("APP_MODE", "watch")
DEFAULT_SEARCH_LIMIT = int(os.environ.get("DEFAULT_SEARCH_LIMIT", "5"))
API_ACCESS_LOGS = os.environ.get("API_ACCESS_LOGS", "").lower() in {"1", "true", "yes", "on"}
OCR_PROMPT = (
    "Extract all text from the image and return clean markdown only. "
    "Preserve headings, lists, tables, and reading order as faithfully as possible."
)
RAG_SYSTEM_PROMPT = (
    "You answer questions using only the provided context when possible. "
    "If the context is insufficient, say so clearly and do not invent facts."
)

OLLAMA_CLIENT = Client(host=OLLAMA_URL)


def response_error_message(exc: ResponseError) -> str:
    return getattr(exc, "error", str(exc))


def ensure_directories() -> None:
    CONSUME_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    initialize_embedding_database()


def pdf_files() -> Iterable[Path]:
    yield from sorted(CONSUME_DIR.glob("*.pdf"))


def stat_signature(path: Path) -> tuple[int, int]:
    stat_result = path.stat()
    return stat_result.st_mtime_ns, stat_result.st_size


def render_pdf_to_pngs(pdf_path: Path) -> list[Path]:
    document = fitz.open(pdf_path)
    png_paths: list[Path] = []
    try:
        for page_number in range(document.page_count):
            page = document.load_page(page_number)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            png_path = IMAGES_DIR / f"{pdf_path.stem}_page_{page_number + 1:04d}.png"
            pixmap.save(png_path.as_posix())
            png_paths.append(png_path)
    finally:
        document.close()
    return png_paths


def ocr_png_to_markdown(png_path: Path) -> str:
    try:
        response_payload = OLLAMA_CLIENT.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": OCR_PROMPT, "images": [str(png_path)]}],
            stream=False,
        )
    except ResponseError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Ollama request failed for {png_path.name}: {response_error_message(exc)}"
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to OCR {png_path.name} with Ollama at {OLLAMA_URL}."
        ) from exc

    markdown = extract_chat_content(response_payload)
    if not markdown:
        raise RuntimeError(
            f"Ollama returned an empty response for {png_path.name}. "
            "Ensure the model supports vision (multimodal) input and is correctly configured."
        )
    return markdown


def embed_text(text: str) -> list[float]:
    try:
        response_payload = OLLAMA_CLIENT.embed(model=EMBEDDING_MODEL, input=text)
    except ResponseError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Ollama embedding request failed for {EMBEDDING_MODEL}: {response_error_message(exc)}"
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to embed text with Ollama at {OLLAMA_URL} using {EMBEDDING_MODEL}."
        ) from exc

    return extract_embedding(response_payload)


def extract_embedding(response_payload: object) -> list[float]:
    if hasattr(response_payload, "embeddings"):
        embeddings = response_payload.embeddings
    elif isinstance(response_payload, dict):
        embeddings = response_payload.get("embeddings")
    else:
        embeddings = None

    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError(
            f"Ollama returned an empty embedding for {EMBEDDING_MODEL}. "
            "Ensure the model supports embeddings and is correctly configured."
        )

    first_embedding = embeddings[0]
    if isinstance(first_embedding, list) and first_embedding:
        return [float(value) for value in first_embedding]
    if all(isinstance(value, (int, float)) for value in embeddings):
        return [float(value) for value in embeddings]

    raise RuntimeError(
        f"Ollama returned an unsupported embedding shape for {EMBEDDING_MODEL}."
    )


def extract_chat_content(response_payload: object) -> str:
    message = getattr(response_payload, "message", None)
    if message is not None:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()

    if isinstance(response_payload, dict):
        payload_message = response_payload.get("message")
        if isinstance(payload_message, dict):
            content = payload_message.get("content", "")
            if isinstance(content, str):
                return content.strip()

    raise RuntimeError("Ollama returned an unsupported chat response payload.")


def write_markdown(pdf_path: Path, page_number: int, markdown: str) -> Path:
    output_path = TEXTS_DIR / f"{pdf_path.stem}_page_{page_number:04d}.md"
    content = markdown if not markdown or markdown.endswith("\n") else f"{markdown}\n"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def initialize_embedding_database() -> None:
    with sqlite3.connect(EMBEDDINGS_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS text_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_pdf TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text_path TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_embedding(
    pdf_path: Path, page_number: int, text_path: Path, markdown: str, embedding: list[float]
) -> None:
    with sqlite3.connect(EMBEDDINGS_DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO text_embeddings (
                source_pdf,
                page_number,
                text_path,
                content,
                embedding_model,
                embedding_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(text_path) DO UPDATE SET
                source_pdf = excluded.source_pdf,
                page_number = excluded.page_number,
                content = excluded.content,
                embedding_model = excluded.embedding_model,
                embedding_json = excluded.embedding_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                pdf_path.name,
                page_number,
                text_path.as_posix(),
                markdown,
                EMBEDDING_MODEL,
                json.dumps(embedding),
            ),
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if left_magnitude == 0.0 or right_magnitude == 0.0:
        return 0.0

    return dot_product / (left_magnitude * right_magnitude)


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict[str, object]]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    query_embedding = embed_text(normalized_query)
    with sqlite3.connect(EMBEDDINGS_DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT source_pdf, page_number, text_path, content, embedding_model, embedding_json
            FROM text_embeddings
            WHERE embedding_model = ?
            """,
            (EMBEDDING_MODEL,),
        ).fetchall()

    scored_results: list[dict[str, object]] = []
    for source_pdf, page_number, text_path, content, embedding_model, embedding_json in rows:
        try:
            stored_embedding = [float(value) for value in json.loads(embedding_json)]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        scored_results.append(
            {
                "source_pdf": source_pdf,
                "page_number": page_number,
                "text_path": text_path,
                "content": content,
                "embedding_model": embedding_model,
                "score": cosine_similarity(query_embedding, stored_embedding),
            }
        )

    scored_results.sort(key=lambda row: row["score"], reverse=True)
    return scored_results[:limit]


def generate_rag_answer(query: str, results: list[dict[str, object]]) -> str:
    context_blocks = []
    for result in results:
        context_blocks.append(
            "\n".join(
                [
                    f"Source PDF: {result['source_pdf']}",
                    f"Page: {result['page_number']}",
                    f"Text path: {result['text_path']}",
                    "Content:",
                    str(result["content"]),
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No matching context was found."
    user_prompt = (
        f"Question: {query.strip()}\n\n"
        f"Context:\n{context}\n\n"
        "Answer using the context above."
    )
    try:
        response_payload = OLLAMA_CLIENT.chat(
            model=RAG_MODEL,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
    except ResponseError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Ollama RAG request failed for {RAG_MODEL}: {response_error_message(exc)}"
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to generate a RAG answer with Ollama at {OLLAMA_URL} using {RAG_MODEL}."
        ) from exc

    answer = extract_chat_content(response_payload)
    if not answer:
        raise RuntimeError(f"Ollama returned an empty response for {RAG_MODEL}.")
    return answer


def handle_search_request(request_payload: object) -> dict[str, object]:
    if not isinstance(request_payload, dict):
        raise ValueError("request body must be a JSON object")

    query = request_payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    limit = request_payload.get("limit", DEFAULT_SEARCH_LIMIT)
    # bool is a subclass of int in Python, so explicitly reject it.
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")
    if limit > 50:
        raise ValueError("limit must be at most 50")

    include_answer = request_payload.get("include_answer", True)
    if not isinstance(include_answer, bool):
        raise ValueError("include_answer must be a boolean")

    results = semantic_search(query, limit=limit)
    response_payload: dict[str, object] = {
        "query": query.strip(),
        "results": results,
    }
    if include_answer:
        response_payload["answer"] = generate_rag_answer(query, results)

    return response_payload


class SearchApiHandler(BaseHTTPRequestHandler):
    """Handle POST /api/search requests for semantic search and RAG."""

    def do_POST(self) -> None:  # noqa: N802
        """Accept JSON with query/limit/include_answer and return JSON search results."""
        if self.path != "/api/search":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b""
            request_payload = json.loads(raw_body.decode("utf-8") or "{}")
            response_payload = handle_search_request(request_payload)
        except UnicodeDecodeError:
            self.write_json({"error": "request body must be UTF-8 encoded"}, HTTPStatus.BAD_REQUEST)
            return
        except json.JSONDecodeError:
            self.write_json({"error": "request body must contain valid JSON"}, HTTPStatus.BAD_REQUEST)
            return
        except ValueError as exc:
            self.write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as exc:
            self.write_json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            return

        self.write_json(response_payload, HTTPStatus.OK)

    def log_message(self, fmt: str, *args: object) -> None:
        """Suppress request logs by default, but allow opt-in access logs via env var."""
        if API_ACCESS_LOGS:
            super().log_message(fmt, *args)

    def write_json(self, payload: dict[str, object], status: HTTPStatus) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def process_pdf(pdf_path: Path) -> None:
    print(f"[{pdf_path.name}] Processing started")
    print(f"[{pdf_path.name}] Rendering PDF pages to PNG images")
    png_paths = render_pdf_to_pngs(pdf_path)
    print(f"[{pdf_path.name}] Rendered {len(png_paths)} page(s)")
    try:
        for page_number, png_path in enumerate(png_paths, start=1):
            print(f"[{pdf_path.name}] OCR page {page_number}/{len(png_paths)}: {png_path.name}")
            markdown = ocr_png_to_markdown(png_path)
            output_path = write_markdown(pdf_path, page_number, markdown)
            print(f"[{pdf_path.name}] Wrote markdown: {output_path.name}")
            print(f"[{pdf_path.name}] Embedding markdown with {EMBEDDING_MODEL}")
            embedding = embed_text(markdown)
            save_embedding(pdf_path, page_number, output_path, markdown, embedding)
            print(
                f"[{pdf_path.name}] Saved embedding for page {page_number} to "
                f"{EMBEDDINGS_DB_PATH.name}"
            )
    finally:
        print(f"[{pdf_path.name}] Cleaning up temporary images")
        for png_path in png_paths:
            if png_path.exists():
                png_path.unlink()

    print(f"[{pdf_path.name}] Removing source PDF")
    pdf_path.unlink()
    print(f"[{pdf_path.name}] Processing finished")


def watch_consume_folder() -> None:
    ensure_directories()
    pending: Dict[Path, tuple[int, int, int]] = {}

    while True:
        current_pdfs = set(pdf_files())

        for pdf_path in current_pdfs:
            try:
                signature = stat_signature(pdf_path)
            except FileNotFoundError:
                continue

            previous = pending.get(pdf_path)
            if previous is None:
                print(
                    f"[{pdf_path.name}] Detected in consume/; waiting for file stability "
                    f"({STABLE_POLLS_REQUIRED} poll(s) required)"
                )
                pending[pdf_path] = (signature[0], signature[1], 1)
                continue

            previous_mtime_ns, previous_size, stable_count = previous
            if (previous_mtime_ns, previous_size) == signature:
                stable_count += 1
            else:
                stable_count = 1

            pending[pdf_path] = (signature[0], signature[1], stable_count)

            if stable_count >= STABLE_POLLS_REQUIRED and (previous_mtime_ns, previous_size) == signature:
                print(f"[{pdf_path.name}] File is stable; starting processing")
                try:
                    process_pdf(pdf_path)
                except Exception as exc:
                    print(f"Failed to process {pdf_path.name}: {exc}")
                finally:
                    pending.pop(pdf_path, None)

        for tracked_path in list(pending):
            if tracked_path not in current_pdfs:
                pending.pop(tracked_path, None)

        time.sleep(POLL_INTERVAL_SECONDS)


def serve_api() -> None:
    ensure_directories()
    with ThreadingHTTPServer((API_HOST, API_PORT), SearchApiHandler) as server:
        print(f"Semantic search API listening on http://{API_HOST}:{API_PORT}/api/search")
        server.serve_forever()


if __name__ == "__main__":
    if APP_MODE == "api":
        serve_api()
    else:
        watch_consume_folder()
