# RAGBench Evaluation — Instructions for Claude Code

## Goal
Run our v2 RAG pipeline (GA reranker) against 3 RAGBench subsets, evaluated with the **TRACe framework** (not RAGAS) using GPT-4o as annotator. Do NOT touch the existing production ChromaDB collection.

## 1. Download Datasets

```python
from datasets import load_dataset

emanual = load_dataset("rungalileo/ragbench", "emanual")      # 132 test, TV manual, 3 docs/q, ~165 tok/doc
delucionqa = load_dataset("rungalileo/ragbench", "delucionqa") # 182 test, Jeep manual, 3 docs/q, ~296 tok/doc
techqa = load_dataset("rungalileo/ragbench", "techqa")         # 310 test, IBM technotes, 5 docs/q, ~1800 tok/doc
```

Each example has: `question`, `documents` (list of context strings), `response` (ground truth), and TRACe ground-truth labels: `relevance`, `utilization`, `adherence` (boolean), `completeness`.

## 2. Create Separate ChromaDB Collections

One NEW collection per dataset. Never modify existing collections:

```python
emanual_col = client.create_collection("ragbench_emanual")
delucionqa_col = client.create_collection("ragbench_delucionqa")
techqa_col = client.create_collection("ragbench_techqa")
```

Index the `documents` from each dataset's **train** split as chunks. Keep metadata (dataset name, doc index).

## 3. Run v2 Pipeline (GA Reranker)

For each dataset's **test** split:
1. Use `question` as input
2. Retrieve from dataset-specific collection (NOT production)
3. Apply GA reranker (no post-reranker score filtering)
4. Generate answer with Qwen3.5:27b (no-think mode)
5. Store: question, retrieved_contexts, rag_answer

## 4. Evaluate with TRACe + GPT-4o

TRACe has 4 metrics (from RAGBench paper, Friel et al. 2024):

| Metric | Formula | Measures |
|--------|---------|----------|
| **Relevance** | Len(relevant spans) / Len(all context) | Retriever quality — what fraction of context is useful |
| **Utilization** | Len(utilized spans) / Len(all context) | Generator efficiency — what fraction of context was used |
| **Completeness** | Len(relevant ∩ utilized) / Len(relevant) | Did the generator use ALL relevant info? |
| **Adherence** | Boolean per example | Is the response fully grounded? (= faithfulness) |

### GPT-4o Annotation Prompt

Use the exact TRACe annotation prompt from RAGBench Appendix 7.4 (reproduced below). It produces a single JSON per example with all 4 metrics:

```python
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
"""
```

### Computing TRACe Metrics from JSON

```python
def compute_trace(annotation_json, doc_sentences, response_sentences):
    total_doc_sents = len(doc_sentences)
    relevant = len(annotation_json["all_relevant_sentence_keys"])
    utilized = len(annotation_json["all_utilized_sentence_keys"])
    
    relevant_set = set(annotation_json["all_relevant_sentence_keys"])
    utilized_set = set(annotation_json["all_utilized_sentence_keys"])
    
    relevance = relevant / total_doc_sents if total_doc_sents > 0 else 0
    utilization = utilized / total_doc_sents if total_doc_sents > 0 else 0
    completeness = len(relevant_set & utilized_set) / len(relevant_set) if relevant_set else 0
    adherence = annotation_json["overall_supported"]  # boolean
    
    return {
        "relevance": relevance,
        "utilization": utilization,
        "completeness": completeness,
        "adherence": adherence  # True/False → 1/0 for averaging
    }
```

### Sentence Splitting

Before calling GPT-4o, split documents and response into sentences with keys:
- Document sentences: "0a.", "0b.", "0c." (doc 0), "1a.", "1b." (doc 1), etc.
- Response sentences: "a.", "b.", "c.", etc.

Use nltk.sent_tokenize or similar for splitting.

## 5. Output

Per dataset: Excel with TRACe scores per query + summary means.
Combined Excel comparing all 3 datasets.

Compare our Adherence rate (% non-hallucinated) against RAGBench baselines from paper Table 3:
- EManual: RAGAS Hal AUROC 0.57, DeBERTa 0.76
- DelucionQA: RAGAS Hal AUROC 0.70, DeBERTa 0.64
- TechQA: RAGAS Hal AUROC 0.52, DeBERTa 0.86

## Cost Estimate (GPT-4o, 1 call per example)

TRACe uses ONE comprehensive call per example (not 6 like RAGAS).

| Dataset | Test | Avg input tokens | Avg output tokens | Cost |
|---------|------|------------------|-------------------|------|
| EManual | 132 | ~2,500 | ~1,500 | ~$2.8 |
| DelucionQA | 182 | ~3,000 | ~1,500 | ~$3.9 |
| TechQA | 310 | ~6,000 | ~2,000 | ~$10.9 |
| **Total** | **624** | | | **~$18** |

Pricing: GPT-4o input $2.50/1M, output $10/1M.
Cheaper than RAGAS (~$25) because TRACe is 1 call/example vs 6.
