# `rag_retriever_v2.py` — What It Does

`RAGRetriever` is the retrieval + answer-generation layer of the RAG pipeline. It wires
together three stages:

1. **Vector search** in ChromaDB
2. **GA (genetic-algorithm) reranking** of the candidate pool
3. **Answer generation** via a local Ollama model

## 1. Setup (`__init__`)

- Loads the sentence-embedding model (`BAAI/bge-base-en-v1.5`, from `config.py`) unless one
  is injected — used to embed queries.
- Opens (or creates) the persistent ChromaDB collection `rag_manuals` that holds the indexed
  manual chunks.
- Instantiates `GAReranker(weights_path=GA_WEIGHTS_PATH)`, loading a **pre-trained** set of
  reranking weights from `data/ga_weights.json`. No GA search happens at query time — the GA
  optimization is a one-time offline training step (see `ga_reranker/`); at inference the
  reranker just applies the weights it already learned.

## 2. Retrieval (`retrieve`)

```
query → embed → ChromaDB top-N (cosine) → filter out vision-skipped chunks → GA rerank → top_k
```

- Embeds the query with the same SentenceTransformer used at indexing time.
- Pulls a **candidate pool** from ChromaDB — `max(GA_RETRIEVAL_POOL, n_results * 2)` results
  (`GA_RETRIEVAL_POOL = 50` by default). This pool is deliberately larger than what's finally
  returned, because reranking needs a wide net of plausible candidates to pick the best subset
  from — a pure top-5 cosine search wouldn't leave the reranker anything to rerank.
- Chunks tagged `type == "vision_skipped"` (images the vision model failed to caption) are
  dropped before reranking.
- Converts ChromaDB's distance to a similarity score (`1.0 - distance`).
- Hands the candidate list to `GAReranker.rerank(query, candidates)`.
- Falls back to plain cosine top-3 if the reranker returns nothing.

## 3. The GA Reranker — what it actually does at query time

This is the interesting part, and it's important to separate **training** from **inference**:

### Training (offline, not run by `RAGRetriever`)
A genetic algorithm (`ga_reranker/genetic_algorithm.py`, based on Dao, Abhary & Marian 2017)
searches for the best combination of 6 numbers — a "chromosome":

```
[w_cosine, w_bm25, w_diversity, w_position, w_coverage, top_k]
```

It evolves a population of these weight-vectors over up to 200 generations using selection,
crossover, mutation, adaptive restarts, and local neighborhood search, scoring each candidate
weight-set by how well it reranks a labeled training set (questions with known relevant
chunk IDs), measured via NDCG / Precision / Recall / MRR. The winning weights are normalized
(the 5 `w_*` values sum to 1) and saved to `data/ga_weights.json`. This is what
`GAReranker.train()` does — it's a separate, offline calibration step, not something that
happens per query.

### Inference (what `RAGRetriever.retrieve()` triggers every query)
`GAReranker.rerank()` loads those pre-learned weights and does a **greedy, iterative
selection** of `top_k` chunks (typically 3–15, learned by the GA) out of the candidate pool.
For each slot, it scores every remaining candidate on 5 criteria (each normalized to `[0,1]`)
and picks the highest weighted sum:

| Criterion    | What it measures                                                                 |
|--------------|-----------------------------------------------------------------------------------|
| `cosine`     | The original embedding similarity from ChromaDB (recomputed as `1 - distance`).   |
| `bm25`       | Lexical term-overlap between query and chunk (BM25-style, without IDF).           |
| `diversity`  | Jaccard-distance of this chunk's words vs. chunks *already selected* — penalizes near-duplicate chunks so the final set doesn't waste slots repeating the same content. |
| `position`   | Favors chunks from earlier pages (`1/(1+log(page))`) — a heuristic that safety warnings/intros tend to appear early in manuals. |
| `coverage`   | Fraction of meaningful query terms (stopwords stripped) that literally appear in the chunk text. |

`diversity` is the only criterion that depends on what's already been picked, which is why
selection is done greedily one slot at a time rather than by sorting once: after each pick,
every remaining candidate's diversity score is recomputed against the updated selected set.

The final weighted score per chunk is:

```
score = w_cosine·cosine + w_bm25·bm25 + w_diversity·diversity
      + w_position·position + w_coverage·coverage
```

### Why this matters for retrieval quality
Plain vector search only optimizes for embedding similarity, which can surface chunks that
are topically close but lexically vague, redundant with each other, or from irrelevant late
pages. The GA reranker re-scores the same candidate pool along axes that pure cosine
similarity ignores — literal keyword coverage (`bm25`, `coverage`), non-redundancy
(`diversity`), and manual structure (`position`) — using weights that were *empirically
tuned* against real relevance judgments rather than guessed. In effect: ChromaDB casts a wide,
cheap net (50 candidates); the GA-learned weights decide, in a smarter and non-redundant way,
which handful of those candidates actually go into the LLM's context window.

## 4. Answer generation (`generate_answer` / `query`)

- `_build_context` caps the reranked docs to at most 2 chunks per source (further diversity
  control) and formats them with `[Source: ... — p.N]` headers, truncating each chunk to 1200
  characters.
- Sends a strict, no-external-knowledge system prompt plus the context and question to a local
  Ollama model (`qwen3.5:27b` by default) and returns the generated answer text.
- `query()` is the convenience wrapper: retrieve → generate → return `(answer, docs)`.

## Other methods
- `list_sources()` — distinct source manual names in the collection.
- `count()` — number of indexed chunks.
