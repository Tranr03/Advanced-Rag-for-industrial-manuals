# The Bag-of-Words (BoW) Part of `rag_retriever_v3.py`

`rag_retriever_v3.py` is otherwise identical to the v2 retriever, with one addition: it
injects a **BM25-weighted bag-of-words vocabulary** into the query-rewriting step. This note
covers only that piece — where the vocabulary comes from and how v3 uses it.

## Where the vocabulary comes from: `bm25_bow.py`

The BoW itself is **not built by the retriever**. It's a separate offline script,
`bm25_bow.py`, run once (or whenever the manuals change) to produce `data/bow.json`:

```
python bm25_bow.py
```

What it does:

1. **Load the corpus** — every preprocessed manual page/chunk under `data/markdown_temp/`
   (`.txt`/`.md` files) becomes one "document."
2. **Tokenize** each document (lowercase, strip stopwords in English/Greek, drop short/numeric
   tokens).
3. **Compute BM25 term weights** per document — the same BM25 formula used elsewhere in this
   project, but here applied at the *corpus* level rather than per-chunk:
   ```
   w(t,d) = IDF(t) * [ f(t,d)·(k1+1) / (f(t,d) + k1·(1 - b + b·|d|/avgdl)) ]
   IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))     # non-negative Lucene variant
   ```
   `f(t,d)` = how often term `t` appears in doc `d`, `df(t)` = how many docs contain `t`,
   `N` = total docs. High weight = the term appears often in a given document but is rare
   across the whole corpus — i.e. it's *discriminative*, not just common.
4. **Rank corpus-wide terms by IDF** into `corpus_top_terms` — the vocabulary's most
   domain-specific words (technical terms like part names, error-code patterns, component
   names), since generic words that appear in most documents get a low/near-zero IDF and sink
   to the bottom.
5. **Write `data/bow.json`**, which stores this `corpus_top_terms` ranking plus per-document
   term weights (the per-document data isn't used by the retriever — only the corpus-level
   ranking is).

In short: `bm25_bow.py` distills "what are the most technically distinctive words used across
all the manuals" into a ranked list, independent of any specific query.

## How `rag_retriever_v3.py` uses it

### Loading ([rag_retriever_v3.py:182-191](../rag_retriever_v3.py#L182-L191))
```python
def _load_bow_terms(self) -> list[str]:
    with open(BOW_PATH, encoding="utf-8") as f:
        bow = json.load(f)
    return [entry["term"] for entry in bow.get("corpus_top_terms", [])[:BOW_TOP_N]]
```
At startup, `RAGRetriever` reads `bow.json` once and keeps just the top **40** terms
(`BOW_TOP_N = 40`) — the most discriminative words in the whole corpus — as a flat list on
`self._bow_terms`. If the file doesn't exist yet (nobody ran `bm25_bow.py`), this silently
returns an empty list and v3 degrades to behaving like a rewrite without vocabulary hints.

### Injection into query rewriting ([rag_retriever_v3.py:193-227](../rag_retriever_v3.py#L193-L227))
Before every retrieval, `_rewrite_query()` asks the LLM to turn the user's question into 3
alternate search queries phrased like manual section titles. The BoW terms are pasted directly
into that prompt as a comma-separated list under a `KEY TECHNICAL TERMS` heading:
```python
terms_block = ", ".join(self._bow_terms) if self._bow_terms else "(not available)"
...
f"KEY TECHNICAL TERMS (most domain-specific words in the corpus):\n{terms_block}\n\n"
```
This is a form of **corpus-grounded prompt conditioning**: instead of letting the LLM rewrite
the query using whatever vocabulary it thinks is plausible, it's shown a cheat-sheet of words
that are *statistically known* to actually appear in the indexed manuals. The rewritten
queries the LLM produces are more likely to reuse this vocabulary, which in turn are embedded
and searched against ChromaDB in the same multi-query retrieval loop v2 doesn't have.

### What it is *not* doing
- It's not a second retrieval channel — there's no BM25/lexical search running against the
  BoW at query time in v3's `retrieve()`. The BoW's only job is to steer the LLM's query
  rewrite phrasing.
- `bm25_bow.py` also ships a `rewrite_query()` function that does classic pseudo-relevance
  feedback (score docs by shared BoW terms, expand the query with the best-matching doc's
  top terms) — but that function is a standalone demo/example in the script (see
  `--demo-rewrite`), and `rag_retriever_v3.py` does not call it. v3 only reads
  `corpus_top_terms` for the prompt injection described above.

## Why it exists
Manual-specific error codes, part names, and mechanism terminology (e.g. "gearbox", "E47",
"axes zeroing") are exactly the kind of vocabulary a general-purpose LLM might paraphrase away
during query rewriting. Grounding the rewrite prompt in real corpus terms nudges the generated
search queries back toward the literal wording used in the manuals, which the embedding model
can then match more precisely against ChromaDB.
