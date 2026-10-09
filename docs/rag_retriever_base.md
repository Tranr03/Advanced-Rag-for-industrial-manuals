# `rag_retriever_base.py`

Defines `RAGRetriever`, the baseline (control) retriever for the RAG system — plain vector search plus answer generation, with no reranking, query rewriting, or few-shot prompting. Used as the comparison point for evaluating fancier retrieval strategies.

## Setup

Loads the embedding model and connects to the same persistent ChromaDB collection used by `rag_preprocessor.py`, so it searches over whatever has already been indexed.

## Core functions

- **`retrieve(query, ...)`** — embeds the query, runs a single cosine-similarity search against ChromaDB, filters out low-score results and `vision_skipped` entries, and returns a ranked list of matching chunks (text + metadata + score).
- **`generate_answer(query, docs)`** — builds a context block from the retrieved chunks and sends it, with a strict instruction-only system prompt, to a local LLM (via Ollama) to produce a concise answer.
- **`query(question, ...)`** — convenience wrapper that just calls `retrieve` then `generate_answer` and returns both.
- **`list_sources()`** — lists all distinct PDF sources currently indexed.
- **`count()`** — returns how many documents are in the collection.

## Supporting behavior

- Results are capped per source (`_build_context`) so one manual can't dominate the answer context.
- Each context snippet is labeled as "Text" or "Visual Element" depending on whether it came from page text or a vision-model image description.

In short: query → embed → search → filter → build context → prompt LLM → answer.
