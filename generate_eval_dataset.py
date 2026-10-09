#!/usr/bin/env python3
"""
generate_eval_dataset.py — Auto-generate evaluation Q&A pairs from ChromaDB chunks.

Samples text chunks spread across all indexed manuals, sends each to the local
LLM asking it to produce one question + reference answer, then writes the result
to data/evaluation_questions.xlsx.

Usage:
    python generate_eval_dataset.py
    python generate_eval_dataset.py --per-source 5 --min-score 0.3
    python generate_eval_dataset.py --output data/my_eval.xlsx
"""

import argparse
import json
import random
import re
import sys
import requests
import pandas as pd
import chromadb

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    VECTOR_DB_PATH, COLLECTION_NAME,
    OLLAMA_URL, OLLAMA_MODEL,
    EVAL_QUESTIONS_PATH,
)

# Minimum chunk length to be worth generating a question from
MIN_CHUNK_CHARS = 200


def _ask_llm(prompt: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "think":   False,
                "options": {"num_ctx": 2048, "temperature": 0.4, "num_gpu": 99},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def _generate_qa(chunk_text: str, source: str) -> tuple[str, str] | None:
    """Ask the LLM to produce one Q&A pair from a chunk. Returns (question, answer) or None."""
    prompt = (
        "You are creating evaluation data for a RAG system about industrial machinery.\n"
        "Read the following excerpt from a machinery manual and write:\n"
        "1. ONE specific, answerable question about a fact, value, procedure, or component mentioned. "
        f"The question must explicitly name the source manual ({source}), e.g. "
        f"\"According to {source}, what is...\".\n"
        "2. The correct answer, drawn only from the text.\n\n"
        "Rules:\n"
        "- The question must be answerable from the text alone.\n"
        "- Do NOT ask vague questions like 'What is this about?'\n"
        "- Do NOT ask yes/no questions.\n"
        "- Output ONLY this exact format (two lines, nothing else):\n"
        "QUESTION: <your question>\n"
        "ANSWER: <your answer>\n\n"
        f"EXCERPT (from {source}):\n{chunk_text[:800]}\n"
    )

    raw = _ask_llm(prompt)
    if raw.startswith("ERROR"):
        return None

    q_match = re.search(r"QUESTION:\s*(.+)", raw)
    a_match = re.search(r"ANSWER:\s*(.+)", raw, re.DOTALL)

    if not q_match or not a_match:
        return None

    question = q_match.group(1).strip()
    answer   = a_match.group(1).strip().split("\n")[0].strip()

    if len(question) < 15 or len(answer) < 10:
        return None

    if source.lower() not in question.lower():
        question = f"According to {source}, {question[0].lower()}{question[1:]}"

    return question, answer


def sample_chunks(collection, per_source: int) -> list[dict]:
    """Sample up to per_source text chunks from each indexed source."""
    all_data = collection.get(include=["documents", "metadatas"])
    ids       = all_data["ids"]
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    # Group by source, keep only text chunks with enough content
    by_source: dict[str, list[dict]] = {}
    for cid, doc, meta in zip(ids, documents, metadatas):
        if not meta or meta.get("type") != "text":
            continue
        if len(doc) < MIN_CHUNK_CHARS:
            continue
        src = meta.get("source", "unknown")
        by_source.setdefault(src, []).append({"id": cid, "text": doc, "source": src, "page": meta.get("page", "?")})

    sampled = []
    for src, chunks in by_source.items():
        random.shuffle(chunks)
        sampled.extend(chunks[:per_source])

    return sampled


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate evaluation Q&A pairs from ChromaDB")
    p.add_argument("--per-source", type=int, default=4,
                   help="chunks to sample per indexed manual (default: 4)")
    p.add_argument("--output", default=str(EVAL_QUESTIONS_PATH),
                   help=f"output Excel path (default: {EVAL_QUESTIONS_PATH})")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for reproducible sampling")
    args = p.parse_args(argv)

    random.seed(args.seed)

    client     = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Collection: {collection.count()} total documents")

    chunks = sample_chunks(collection, args.per_source)
    print(f"Sampled {len(chunks)} chunks across {len({c['source'] for c in chunks})} sources\n")

    MAX_RETRIES = 5

    rows = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {chunk['source']} p.{chunk['page']} — generating Q&A...", end=" ", flush=True)
        result = None
        for attempt in range(1, MAX_RETRIES + 1):
            result = _generate_qa(chunk["text"], chunk["source"])
            if result is not None:
                break
            print(f"retry {attempt}/{MAX_RETRIES}...", end=" ", flush=True)
        if result is None:
            print(f"failed after {MAX_RETRIES} attempts — skipping")
            continue
        question, answer = result
        rows.append({
            "Question": question,
            "Answer":   answer,
            "Source":   chunk["source"],
            "Page":     chunk["page"],
        })
        print(f"OK  →  {question[:60]}...")

    df = pd.DataFrame(rows, columns=["Question", "Answer", "Source", "Page"])
    df.to_excel(args.output, index=False)
    print(f"\nWrote {len(df)} Q&A pairs to {args.output}")
    print("Review the file, delete weak questions, then run the evaluation notebook.")


if __name__ == "__main__":
    sys.exit(main())
