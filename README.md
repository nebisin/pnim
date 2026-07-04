# PDF OCR, Semantic Search, and RAG

A small Python app that monitors `./consume` for new PDF files, renders each page to PNG images, sends those images to Ollama using `deepseek-ocr:latest`, saves the OCR output as Markdown, generates embeddings with `embeddinggemma:latest`, stores them in SQLite, and can expose an HTTP API for semantic search and Retrieval-Augmented Generation with `gemma4:26b`.

## What it does

- Watches `./consume` for new `.pdf` files
- Splits each PDF into page images in `./images`
- Sends each PNG to Ollama for OCR using `deepseek-ocr:latest`
- Writes one Markdown file per page into `./texts`
- Generates one embedding per Markdown page using `embeddinggemma:latest`
- Saves embeddings and source text metadata into `./text_embeddings.sqlite3`
- Deletes the PNG files after OCR completes
- Deletes the original PDF after successful processing
- Exposes `POST /api/search` for semantic search and RAG responses over stored text

## Requirements

- Python 3.14+ in the project virtual environment
- Ollama running locally
- The `deepseek-ocr:latest` model pulled in Ollama
- The `embeddinggemma:latest` model pulled in Ollama
- The `gemma4:26b` model pulled in Ollama

Example:

```bash
ollama pull deepseek-ocr:latest
ollama pull embeddinggemma:latest
ollama pull gemma4:26b
ollama serve
```

## Setup

Create and activate a virtual environment, then install the Python dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Folder layout

The script creates these folders automatically if they do not exist:

- `consume/` — drop PDFs here
- `images/` — temporary PNG page images
- `texts/` — generated Markdown output

It also creates `text_embeddings.sqlite3` in the project root by default.

## Run

Start the watcher from the project root:

```bash
.venv/bin/python watch_consume.py
```

Start the API server from the project root:

```bash
APP_MODE=api .venv/bin/python watch_consume.py
```

Example request:

```bash
curl -X POST http://127.0.0.1:8080/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does the document say about refunds?","limit":3,"include_answer":true}'
```

## Environment variables

- `OLLAMA_URL` — Ollama base URL, defaults to `http://localhost:11434`
- `OLLAMA_MODEL` — model name, defaults to `deepseek-ocr:latest`
- `EMBEDDING_MODEL` — embedding model name, defaults to `embeddinggemma:latest`
- `RAG_MODEL` — chat model for grounded answers, defaults to `gemma4:26b`
- `EMBEDDINGS_DB_PATH` — SQLite database path, defaults to `text_embeddings.sqlite3`
- `POLL_INTERVAL_SECONDS` — scan interval in seconds, defaults to `2`
- `STABLE_POLLS_REQUIRED` — number of unchanged scans before processing, defaults to `2`
- `APP_MODE` — `watch` for the PDF watcher or `api` for the HTTP server, defaults to `watch`
- `API_HOST` — API bind host, defaults to `127.0.0.1`
- `API_PORT` — API bind port, defaults to `8080`
- `DEFAULT_SEARCH_LIMIT` — default number of retrieved chunks, defaults to `5`

Example:

```bash
OLLAMA_URL=http://localhost:11434 OLLAMA_MODEL=deepseek-ocr:latest EMBEDDING_MODEL=embeddinggemma:latest RAG_MODEL=gemma4:26b .venv/bin/python watch_consume.py
```

## Notes

- Drop only finished PDF files into `consume/`.
- The watcher waits for a file to remain unchanged across multiple polls before processing it.
- If OCR fails, the PDF is kept so you can fix the issue and try again.
- The API searches the stored page embeddings, returns ranked matches, and can optionally generate an answer grounded in those matches.
