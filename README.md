# PDF OCR Watcher

A small Python watcher that monitors `./consume` for new PDF files, renders each page to PNG images, sends those images to Ollama using `deepseek-ocr:latest`, saves the OCR output as Markdown, and removes the temporary files after processing.

## What it does

- Watches `./consume` for new `.pdf` files
- Splits each PDF into page images in `./images`
- Sends each PNG to Ollama for OCR using `deepseek-ocr:latest`
- Writes one Markdown file per page into `./texts`
- Deletes the PNG files after OCR completes
- Deletes the original PDF after successful processing

## Requirements

- Python 3.14+ in the project virtual environment
- Ollama running locally
- The `deepseek-ocr:latest` model pulled in Ollama

Example:

```bash
ollama pull deepseek-ocr:latest
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

## Run

Start the watcher from the project root:

```bash
.venv/bin/python watch_consume.py
```

## Environment variables

- `OLLAMA_URL` — Ollama base URL, defaults to `http://localhost:11434`
- `OLLAMA_MODEL` — model name, defaults to `deepseek-ocr:latest`
- `POLL_INTERVAL_SECONDS` — scan interval in seconds, defaults to `2`
- `STABLE_POLLS_REQUIRED` — number of unchanged scans before processing, defaults to `2`

Example:

```bash
OLLAMA_URL=http://localhost:11434 OLLAMA_MODEL=deepseek-ocr:latest .venv/bin/python watch_consume.py
```

## Notes

- Drop only finished PDF files into `consume/`.
- The watcher waits for a file to remain unchanged across multiple polls before processing it.
- If OCR fails, the PDF is kept so you can fix the issue and try again.
