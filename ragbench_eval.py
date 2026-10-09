#!/usr/bin/env python3
"""
ragbench_eval.py — Evaluate the v2 RAG pipeline (GA reranker, no query rewrite)
against 3 RAGBench subsets (emanual, delucionqa, techqa) using the TRACe
framework with GPT-4o as annotator, per mds/ragbench_eval_instructions.md.

Builds one throwaway ChromaDB collection per dataset ("ragbench_<name>") in
the same persistent store as production — never touches the "rag_manuals"
collection.

Stages (run independently so a long job can be resumed/inspected):
  index    — build/reuse the per-dataset ChromaDB collections from the train split
  retrieve — run retrieve() + generate_answer() over the test split (local Ollama, free)
  evaluate — annotate each result with GPT-4o via the TRACe prompt (costs money)
  all      — index + retrieve + evaluate (default)

Usage:
    python ragbench_eval.py --stage retrieve --datasets emanual --limit 5   # smoke test, no cost
    python ragbench_eval.py --stage all                                    # full run, ~$18 in GPT-4o calls
"""

from __future__ import annotations

import os
# Must be set before `import config`, which does os.environ.setdefault(...) —
# HF_HUB_OFFLINE=1 (config's default) would block load_dataset() from fetching
# RAGBench off the Hub.
os.environ["HF_HUB_OFFLINE"] = "0"

import argparse
import json
import re
import sys
import chromadb
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from config import EMBEDDING_MODEL, VECTOR_DB_PATH, OPENAI_API_KEY, DATA_DIR
from rag_retriever_v2 import RAGRetriever

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAGBENCH_SUBSETS     = ["emanual", "delucionqa", "techqa"]
RAGBENCH_RESULTS_DIR = DATA_DIR / "ragbench_results"

RAGBENCH_BASELINES = {
    # RAGBench paper (Friel et al. 2024) Table 3 — Hallucination detection AUROC
    "emanual":    {"ragas_hal_auroc": 0.57, "deberta_auroc": 0.76},
    "delucionqa": {"ragas_hal_auroc": 0.70, "deberta_auroc": 0.64},
    "techqa":     {"ragas_hal_auroc": 0.52, "deberta_auroc": 0.86},
}

TRACE_PROMPT = """I asked someone to answer a question based on one or more documents.
Your task is to review their response and assess whether or not each sentence
in that response is supported by text in the documents. And if so, which
sentences in the documents provide that support. You will also tell me which
of the documents contain useful information for answering the question, and
which of the documents the answer was sourced from.

Here are the documents, each of which is split into sentences. Alongside each
sentence is associated key, such as '0a.' or '0b.' that you can use to refer to it:
'''
{documents}
'''

The question was:
'''
{question}
'''

Here is their response, split into sentences. Alongside each sentence is
associated key, such as 'a.' or 'b.' that you can use to refer to it:
'''
{answer}
'''

You must respond with a JSON object matching this schema:
'''
{{
  "relevance_explanation": string,
  "all_relevant_sentence_keys": [string],
  "overall_supported_explanation": string,
  "overall_supported": boolean,
  "sentence_support_information": [
    {{
      "response_sentence_key": string,
      "explanation": string,
      "supporting_sentence_keys": [string],
      "fully_supported": boolean
    }}
  ],
  "all_utilized_sentence_keys": [string]
}}
'''
The relevance_explanation field is a string explaining which documents
contain useful information for answering the question. Provide a step-by-step
breakdown of information provided in the documents and how it is useful for
answering the question.
The all_relevant_sentence_keys field is a list of all document sentences keys
(e.g. ’0a’) that are revant to the question. Include every sentence that is
useful and relevant to the question, even if it was not used in the response,
or if only parts of the sentence are useful. Ignore the provided response when
making this judgement and base your judgement solely on the provided documents
and question. Omit sentences that, if removed from the document, would not
impact someone’s ability to answer the question.
The overall_supported_explanation field is a string explaining why the response
*as a whole* is or is not supported by the documents. In this field, provide a
step-by-step breakdown of the claims made in the response and the support (or
lack thereof) for those claims in the documents. Begin by assessing each claim
separately, one by one; don’t make any remarks about the response as a whole
until you have assessed all the claims in isolation.
The overall_supported field is a boolean indicating whether the response as a
whole is supported by the documents. This value should reflect the conclusion
you drew at the end of your step-by-step breakdown in overall_supported_explanation.
In the sentence_support_information field, provide information about the support
*for each sentence* in the response.
The sentence_support_information field is a list of objects, one for each sentence
in the response. Each object MUST have the following fields:- response_sentence_key: a string identifying the sentence in the response.
This key is the same as the one used in the response above.- explanation: a string explaining why the sentence is or is not supported by the
documents.- supporting_sentence_keys: keys (e.g. ’0a’) of sentences from the documents that
support the response sentence. If the sentence is not supported, this list MUST
be empty. If the sentence is supported, this list MUST contain one or more keys.
In special cases where the sentence is supported, but not by any specific sentence,
you can use the string "supported_without_sentence" to indicate that the sentence
is generally supported by the documents. Consider cases where the sentence is
expressing inability to answer the question due to lack of relevant information in
the provided contex as "supported_without_sentence". In cases where the sentence
is making a general statement (e.g. outlining the steps to produce an answer, or
summarizing previously stated sentences, or a transition sentence), use the
sting "general".In cases where the sentence is correctly stating a well-known fact,
like a mathematical formula, use the string "well_known_fact". In cases where the
sentence is performing numerical reasoning (e.g. addition, multiplication), use
the string "numerical_reasoning".- fully_supported: a boolean indicating whether the sentence is fully supported by
the documents.- This value should reflect the conclusion you drew at the end of your step-by-step
breakdown in explanation.- If supporting_sentence_keys is an empty list, then fully_supported must be false.
- Otherwise, use fully_supported to clarify whether everything in the response
sentence is fully supported by the document text indicated in supporting_sentence_keys
(fully_supported = true), or whether the sentence is only partially or incompletely
supported by that document text (fully_supported = false).
The all_utilized_sentence_keys field is a list of all sentences keys (e.g. ’0a’) that
were used to construct the answer. Include every sentence that either directly supported
the answer, or was implicitly used to construct the answer, even if it was not used
in its entirety. Omit sentences that were not used, and could have been removed from
the documents without affecting the answer.
You must respond with a valid JSON string. Use escapes for quotes, e.g. ‘\\"‘, and
newlines, e.g. ‘\\n‘. Do not write anything before or after the JSON string. Do not
wrap the JSON string in backticks like ‘‘‘ or ‘‘‘json.
As a reminder: your task is to review the response and assess which documents contain
useful information pertaining to the question, and how each sentence in the response
is supported by the text in the documents.\
"""


# ─── Sentence splitting + keying ───────────────────────────────────────────────
# nltk isn't installed in this venv and would need a network download for
# punkt data; a regex splitter is good enough for TRACe's sentence keys
# ("use nltk.sent_tokenize or similar" per the instructions).

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _letter(i: int) -> str:
    """0->a, 1->b, ..., 25->z, 26->aa, ..."""
    i += 1
    letters = ""
    while i > 0:
        i, rem = divmod(i - 1, 26)
        letters = chr(97 + rem) + letters
    return letters


def key_documents(documents: list[str]) -> tuple[str, int]:
    """Formats retrieved contexts with '0a.', '0b.', '1a.' ... keys. Returns (block, total_sentences)."""
    blocks = []
    total = 0
    for doc_idx, doc_text in enumerate(documents):
        sents = split_sentences(doc_text)
        lines = [f"{doc_idx}{_letter(i)}. {s}" for i, s in enumerate(sents)]
        total += len(sents)
        blocks.append(f"Document {doc_idx}:\n" + "\n".join(lines))
    return "\n\n".join(blocks), total


def key_response(response: str) -> str:
    """Formats the answer with 'a.', 'b.', ... keys."""
    sents = split_sentences(response)
    return "\n".join(f"{_letter(i)}. {s}" for i, s in enumerate(sents))


# ─── TRACe scoring ──────────────────────────────────────────────────────────────

def compute_trace(annotation: dict, total_doc_sents: int) -> dict:
    relevant_set = set(annotation.get("all_relevant_sentence_keys", []))
    utilized_set = set(annotation.get("all_utilized_sentence_keys", []))

    return {
        "relevance":    len(relevant_set) / total_doc_sents if total_doc_sents else 0.0,
        "utilization":  len(utilized_set) / total_doc_sents if total_doc_sents else 0.0,
        "completeness": len(relevant_set & utilized_set) / len(relevant_set) if relevant_set else 0.0,
        "adherence":    bool(annotation.get("overall_supported", False)),
    }


def annotate_trace(client: OpenAI, documents_block: str, question: str, response_block: str) -> dict | None:
    prompt = TRACE_PROMPT.format(documents=documents_block, question=question, answer=response_block)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                print(f"    TRACe annotation failed after 3 attempts: {e}")
                return None
    return None


# ─── Stage 1: index (per-dataset ChromaDB collection from the train split) ────

def build_ragbench_collection(client, dataset_name: str, embedding_model) -> "chromadb.Collection":
    collection_name = f"ragbench_{dataset_name}"
    try:
        collection = client.get_collection(collection_name)
        if collection.count() > 0:
            print(f"[{dataset_name}] collection '{collection_name}' already has {collection.count()} docs — reusing")
            return collection
    except Exception:
        collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    print(f"[{dataset_name}] loading train split for corpus...")
    train = load_dataset("rungalileo/ragbench", dataset_name, split="train")

    unique_docs: dict[str, int] = {}
    for row in train:
        for doc in row["documents"]:
            if doc and doc not in unique_docs:
                unique_docs[doc] = len(unique_docs)

    texts = list(unique_docs.keys())
    print(f"[{dataset_name}] embedding {len(texts)} unique passages...")

    BATCH = 64
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        embeddings = embedding_model.encode(batch).tolist()
        ids       = [f"{dataset_name}_{unique_docs[t]}" for t in batch]
        # Each passage gets its OWN "source" id — _build_context() caps context
        # to 2 chunks per source, so a shared source value would collapse the
        # whole corpus down to 2 chunks regardless of top_k.
        metadatas = [{"source": f"{dataset_name}_doc{unique_docs[t]}", "page": 1, "type": "text"} for t in batch]
        collection.add(ids=ids, embeddings=embeddings, documents=batch, metadatas=metadatas)

    print(f"[{dataset_name}] indexed {collection.count()} passages")
    return collection


# ─── Stage 2: retrieve + generate (local Ollama, no API cost) ─────────────────

def run_pipeline(dataset_name: str, retriever: RAGRetriever, limit: int | None = None) -> list[dict]:
    out_path = RAGBENCH_RESULTS_DIR / f"{dataset_name}_pipeline.json"

    test = load_dataset("rungalileo/ragbench", dataset_name, split="test")
    if limit:
        test = test.select(range(min(limit, len(test))))

    results = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)

    for i in range(len(results), len(test)):
        row      = test[i]
        question = row["question"]
        docs     = retriever.retrieve(question, n_results=10)
        answer   = retriever.generate_answer(question, docs)

        results.append({
            "question":             question,
            "ground_truth_response": row.get("response", ""),
            "retrieved_contexts":    [d["text"] for d in docs],
            "rag_answer":            answer,
        })

        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{dataset_name}] retrieve+generate {i + 1}/{len(test)}")

    return results


# ─── Stage 3: TRACe evaluation (GPT-4o, costs money) ──────────────────────────

def run_trace_eval(dataset_name: str, limit: int | None = None) -> list[dict] | None:
    pipeline_path = RAGBENCH_RESULTS_DIR / f"{dataset_name}_pipeline.json"
    if not pipeline_path.exists():
        print(f"[{dataset_name}] no pipeline output found — run --stage retrieve first")
        return None

    with open(pipeline_path, encoding="utf-8") as f:
        pipeline_results = json.load(f)
    if limit:
        pipeline_results = pipeline_results[:limit]

    scores_path = RAGBENCH_RESULTS_DIR / f"{dataset_name}_trace_scores.json"
    scores = []
    if scores_path.exists():
        with open(scores_path, encoding="utf-8") as f:
            scores = json.load(f)

    client = OpenAI(api_key=OPENAI_API_KEY)

    for i in range(len(scores), len(pipeline_results)):
        item = pipeline_results[i]
        documents_block, total_doc_sents = key_documents(item["retrieved_contexts"])
        response_block = key_response(item["rag_answer"])

        annotation = annotate_trace(client, documents_block, item["question"], response_block)
        if annotation is None:
            metrics = {"relevance": None, "utilization": None, "completeness": None, "adherence": None}
        else:
            metrics = compute_trace(annotation, total_doc_sents)

        scores.append({"question": item["question"], **metrics})
        scores_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{dataset_name}] TRACe {i + 1}/{len(pipeline_results)} — "
              f"rel={metrics['relevance']} util={metrics['utilization']} "
              f"comp={metrics['completeness']} adh={metrics['adherence']}")

    return scores


# ─── Excel export ───────────────────────────────────────────────────────────────

def export_dataset_excel(dataset_name: str, scores: list[dict]) -> dict:
    df = pd.DataFrame(scores)

    means = {
        col: df[col].mean(skipna=True)
        for col in ("relevance", "utilization", "completeness")
    }
    means["adherence"] = df["adherence"].map(
        lambda v: float(v) if isinstance(v, bool) else float("nan")
    ).mean(skipna=True)

    out_path = RAGBENCH_RESULTS_DIR / f"{dataset_name}_trace_scores.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        df.to_excel(writer, sheet_name="per_query", index=False)
        pd.DataFrame([means]).to_excel(writer, sheet_name="summary", index=False)

    print(f"[{dataset_name}] wrote {out_path}")
    return means


def export_combined_excel(all_means: dict[str, dict]):
    rows = []
    for name, means in all_means.items():
        rows.append({"dataset": name, **means, **RAGBENCH_BASELINES.get(name, {})})

    out_path = RAGBENCH_RESULTS_DIR / "ragbench_combined_summary.xlsx"
    pd.DataFrame(rows).to_excel(out_path, index=False)
    print(f"\nWrote combined summary: {out_path}")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="RAGBench evaluation of the v2 pipeline via TRACe + GPT-4o")
    p.add_argument("--datasets", nargs="+", default=RAGBENCH_SUBSETS, choices=RAGBENCH_SUBSETS)
    p.add_argument("--stage", choices=["index", "retrieve", "evaluate", "all"], default="all")
    p.add_argument("--limit", type=int, default=None,
                   help="limit test examples per dataset (smoke test)")
    args = p.parse_args(argv)

    RAGBENCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    need_pipeline = args.stage in ("index", "retrieve", "all")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL) if need_pipeline else None
    chroma_client   = chromadb.PersistentClient(path=str(VECTOR_DB_PATH)) if need_pipeline else None

    all_means = {}
    for name in args.datasets:
        collection = None
        if need_pipeline:
            collection = build_ragbench_collection(chroma_client, name, embedding_model)

        if args.stage in ("retrieve", "all"):
            retriever = RAGRetriever(embedding_model=embedding_model, collection=collection)
            run_pipeline(name, retriever, limit=args.limit)

        if args.stage in ("evaluate", "all"):
            scores = run_trace_eval(name, limit=args.limit)
            if scores:
                all_means[name] = export_dataset_excel(name, scores)

    if all_means:
        export_combined_excel(all_means)


if __name__ == "__main__":
    sys.exit(main())
