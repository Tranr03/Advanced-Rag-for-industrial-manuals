# Reducing Prompt Dependency in the RAG Pipeline

Two targeted changes to `rag_retriever.py` that reduce how much the system relies on precise prompt wording, without touching the overall architecture.

---

## 1. HyDE — Hypothetical Document Embeddings (Query Rewriting)

### The Problem

The current retrieval embeds the user's raw question and searches for semantically similar chunks in ChromaDB. The issue is that questions and manual text live in different linguistic spaces:

- **User asks:** *"What do I do if the conveyor belt slips?"*
- **Manual says:** *"If lateral displacement occurs during operation, adjust the tensioning roller..."*

These don't embed close together, so relevant chunks get low cosine scores and may fall below the `score_threshold`.

### The Solution

Before embedding, send the query to the LLM and ask it to produce a **short hypothetical answer** — the kind of text that would appear in a manual. Embed that instead of the raw question.

```
User query  →  LLM  →  Hypothetical manual excerpt  →  embed  →  ChromaDB search
```

Because the hypothetical answer uses manual-style language, it embeds much closer to real chunks.

### Where to Change It

All changes go inside `retrieve()` in [rag_retriever.py](rag_retriever.py), before the `self.embedding_model.encode(...)` call.

```python
def _hypothetical_document(self, query: str) -> str:
    """Ask the LLM to produce a short manual-style passage for the query."""
    prompt = (
        "Write a short technical paragraph (2-4 sentences) as it would appear "
        "in an industrial machinery manual, answering the following question. "
        "Do not add any preamble.\n\n"
        f"Question: {query}"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "think":   False,       # skip chain-of-thought for speed
                "options": {"num_ctx": 512, "temperature": 0.2, "num_gpu": 99}
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("response", query).strip()
    except Exception:
        return query  # graceful fallback to original query
```

Then replace the embedding line in `retrieve()`:

```python
# Before
query_embedding = self.embedding_model.encode([query]).tolist()[0]

# After
hyde_text = self._hypothetical_document(query)
query_embedding = self.embedding_model.encode([hyde_text]).tolist()[0]
```

### Trade-offs

| Pro | Con |
|-----|-----|
| Retrieval quality improves for short/vague questions | Adds one extra LLM call (~1-3 s on local hardware) |
| No changes to the GA reranker or ChromaDB schema | HyDE can hallucinate and retrieve wrong chunks if the LLM misunderstands the domain |
| Fallback to raw query on LLM error — zero breakage risk | Needs `think: False` to keep latency acceptable |

### When It Helps Most

- Short, conversational questions ("What is the max RPM?")
- Questions using different terminology than the manual
- Questions with implicit context ("Why does it vibrate?")

### When It Helps Least

- Queries that already mirror manual language
- Very specific part numbers or error codes (embed well as-is)

---

## 2. Minimal System Prompt + Few-Shot Examples

### The Problem

The current system prompt is behavioral and verbose:

```
"You are a technical assistant specialized in industrial machinery manuals.
Answer using only the provided context. Be precise and concise.
If the context contains no relevant information, say:
'I could not find this information in the available manuals.'"
```

Small local models (qwen3:8b) are brittle to instruction wording — slight rephrasing produces different output styles, and instructions like "be concise" are interpreted inconsistently. The fallback phrase is especially fragile: if the model paraphrases it, downstream parsing or evaluation breaks.

### The Solution

Replace verbose instructions with a **minimal domain statement** plus **one or two concrete Q&A examples**. Models generalize far more reliably from examples than from rules.

### Where to Change It

All changes go inside `generate_answer()` in [rag_retriever.py](rag_retriever.py).

```python
def generate_answer(self, query: str, docs: list[dict]) -> str:
    if not docs:
        return "No relevant information found in the indexed manuals."

    context = self._build_context(docs)

    # Minimal domain anchor — just one sentence
    system = (
        "You are a technical assistant for industrial machinery. "
        "Answer only from the provided context."
    )

    # Few-shot examples teach format better than instructions do
    few_shot = (
        "CONTEXT:\n"
        "[Text — ExampleManual.pdf, p.12]\n"
        "The oil filter must be replaced every 500 operating hours or annually, whichever comes first.\n\n"
        "QUESTION: How often should the oil filter be replaced?\n"
        "ANSWER: The oil filter must be replaced every 500 operating hours or once a year, "
        "whichever comes first (ExampleManual.pdf, p.12).\n\n"
        "---\n\n"
        "CONTEXT:\n"
        "[Text — ExampleManual.pdf, p.34]\n"
        "Lubrication schedule: bearings every 200 h, gearbox every 1000 h.\n\n"
        "QUESTION: What is the maintenance interval for the conveyor motor?\n"
        "ANSWER: The available manuals do not contain information about the conveyor motor maintenance interval.\n\n"
        "---\n\n"
    )

    prompt = (
        f"{few_shot}"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n"
        "ANSWER:"
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "system":  system,
                "prompt":  prompt,
                "stream":  False,
                "think":   True,
                "options": {"num_ctx": 4096, "temperature": 0.1, "num_gpu": 99}
            },
            timeout=200
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        return f"Error generating answer: {e}"
```

### What the Few-Shot Examples Teach

1. **Citation style** — the model sees that answers include `(source, page)` without being told to
2. **"Not found" phrasing** — the exact fallback phrase is demonstrated, not instructed; the model copies it reliably
3. **Conciseness** — the model observes short answers without an instruction to "be concise"

### Trade-offs

| Pro | Con |
|-----|-----|
| Output format becomes consistent and predictable | Few-shot examples consume ~200-300 extra tokens per call |
| "Not found" behavior is stable regardless of instruction phrasing | Example content must be domain-plausible; bad examples hurt more than no examples |
| Works better on smaller models (qwen3:8b) than long instruction lists | Examples need updating if the context format in `_build_context()` ever changes |

### When It Helps Most

- Inconsistent answer formatting across different question types
- The model ignoring or paraphrasing the fallback "not found" message
- Output style varying between short and long queries

---

## Combining Both Changes

These two changes are fully independent and stack cleanly:

- **HyDE** improves *what gets retrieved* — better chunks reach the LLM
- **Few-shot prompt** improves *how the answer is generated* — more consistent output from whatever chunks arrive

Together they reduce prompt sensitivity at both pipeline stages (retrieval and generation) with no changes to ChromaDB, the GA reranker, the embedding model, or the Streamlit UI.
