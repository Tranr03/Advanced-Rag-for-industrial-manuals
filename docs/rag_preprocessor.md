# `rag_preprocessor.py`

Defines `RAGPreprocessor`, the class that ingests industrial-manual PDFs into a ChromaDB vector store for RAG retrieval. It combines plain text extraction with a vision model's description of diagrams/schematics, so both textual and visual content become searchable.

## Setup (`__init__`)

- Loads a `SentenceTransformer` embedding model (`EMBEDDING_MODEL` from `config.py`, default `BAAI/bge-base-en-v1.5`).
- Creates a `RecursiveCharacterTextSplitter` (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=80`).
- Opens (or creates) a persistent ChromaDB collection (`COLLECTION_NAME`, cosine similarity) at `VECTOR_DB_PATH`.

## Main pipeline — `process_pdf(pdf_path)`

For a single PDF, runs three stages and returns `(n_text_chunks, n_vision_chunks)`:

1. **Text extraction** (`_extract_text`) — opens the PDF with `pypdfium2` and pulls raw text per page. The full concatenated text is also saved as a `.md` file in `MARKDOWN_FOLDER`.
2. **Text indexing** (`_index_text`) — splits each page's text into chunks, embeds them, and adds them to ChromaDB with metadata (`source`, `page`, `type: "text"`). Chunks are also dumped to a JSON file in `CHUNKS_FOLDER`. Skips work if the PDF's text is already indexed.
3. **Vision indexing** (`_index_vision`) — detects pages containing image/graphic objects (`obj.type == 2` in pdfium), renders them to PNG, and sends each to a local vision LLM (via Ollama) for description. Descriptions are embedded and stored with `type: "image_description"`. Pages with no diagram are marked `type: "vision_skipped"` so they aren't retried. Supports resuming after a crash (checks what's already in the collection and what PNGs already exist on disk) and reports progress via optional callbacks.

## Vision description — `_vision_describe`

Resizes the page image (max width 1024px), base64-encodes it, and POSTs it to Ollama's `/api/generate` endpoint with `VISION_MODEL` (default `qwen3-vl:8b`) and a prompt instructing it to describe diagrams/schematics/tables in detail, or reply `NO_DIAGRAM` for plain text/decorative pages. Returns `""` on any request failure (network error, timeout, etc.), which is treated the same as "no diagram."

## Batch processing — `process_all(manuals_folder)`

Iterates every `*.pdf` in a folder, skipping any PDF whose stem is already present in the collection, and calls `process_pdf` on the rest. Returns a `{filename: (n_text, n_vision)}` dict.

## Other public methods

- **`is_indexed(pdf_stem)`** — checks whether a PDF has any documents in the collection.
- **`delete_pdf(pdf_stem)`** — removes all ChromaDB entries for a PDF plus its on-disk artifacts (chunk JSON, page images/JSON, markdown file). Returns the number of deleted documents.

## Data flow summary

```
PDF ──▶ per-page text ──▶ chunk + embed ──▶ ChromaDB (type=text)
     └─▶ pages w/ graphics ──▶ render PNG ──▶ vision LLM ──▶ embed ──▶ ChromaDB (type=image_description / vision_skipped)
```

Both text and vision entries live in the same collection, distinguished by the `type` metadata field, so retrieval can pull from either or both.

## Notable design points

- **Idempotent / resumable**: every stage checks existing ChromaDB entries (and, for vision, existing PNG files) before doing work, so re-running `process_pdf` on an already-processed PDF is cheap and crash recovery just continues where it left off.
- **Vision gating**: only pages pdfium reports as containing image objects are sent to the vision model, avoiding needless LLM calls on pure-text pages.
- **Local-only dependency**: the vision step depends on an Ollama server running locally (`OLLAMA_URL`); failures there degrade gracefully to "no diagram" rather than crashing the pipeline.
