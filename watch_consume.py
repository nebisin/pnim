from __future__ import annotations

import os
import time
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
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "2"))
STABLE_POLLS_REQUIRED = int(os.environ.get("STABLE_POLLS_REQUIRED", "2"))
OCR_PROMPT = (
    "Extract all text from the image and return clean markdown only. "
    "Preserve headings, lists, tables, and reading order as faithfully as possible."
)

OLLAMA_CLIENT = Client(host=OLLAMA_URL)


def ensure_directories() -> None:
    CONSUME_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)


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
            f"Ollama request failed for {png_path.name}: {exc.error}"
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to OCR {png_path.name} with Ollama at {OLLAMA_URL}."
        ) from exc

    markdown = (response_payload.message.content or "").strip()
    if not markdown:
        raise RuntimeError(
            f"Ollama returned an empty response for {png_path.name}. "
            "Ensure the model supports vision (multimodal) input and is correctly configured."
        )
    return markdown


def write_markdown(pdf_path: Path, page_number: int, markdown: str) -> Path:
    output_path = TEXTS_DIR / f"{pdf_path.stem}_page_{page_number:04d}.md"
    output_path.write_text(markdown + ("\n" if markdown and not markdown.endswith("\n") else ""), encoding="utf-8")
    return output_path


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


if __name__ == "__main__":
    watch_consume_folder()
