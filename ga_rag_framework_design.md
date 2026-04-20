# GA-Optimized RAG Retrieval Framework — Design Document

## 1. Problem Statement

We have a **naive RAG pipeline** for reading complex industrial documents (manuals). The current pipeline uses:
- **LangChain** as orchestration framework
- **ChromaDB** as vector store
- **"BAAI/bge-base-en-v1.5"** for vectorization
- **Cosine similarity** as the sole retrieval criterion

The retriever returns the top-N chunks ranked purely by cosine similarity. This is insufficient because:
- Chunks with high cosine scores may be **semantically redundant** (covering the same information)
- They may **not cover all aspects** of the query
- They may lack the **correct ordering** for the LLM to synthesize an answer
- Performance **degrades as the database grows**

## 2. Proposed Solution: GA-Based Reranker

### Core Idea

A **Genetic Algorithm (GA)** learns **offline** the optimal weights of a multi-criteria scoring function. At inference time, these learned weights are applied instantly — **no GA runs during inference**, so there is zero latency overhead.

### Architecture Overview

```
[Query] → [ChromaDB Retriever (top-N)] → [GA-Optimized Reranker] → [Top-K Reranked Chunks] → [LLM Generator] → [Answer]
```

The GA-Optimized Reranker sits **between** the retriever and the generator as a plug-in module.

### Multi-Criteria Scoring

Each retrieved chunk is scored using multiple criteria:

| Criterion | Description | Why It Matters |
|-----------|-------------|----------------|
| `cosine_similarity` | Original vector similarity score from ChromaDB | Baseline semantic relevance |
| `bm25_score` | BM25/keyword overlap between query and chunk | Captures exact keyword matches that embeddings may miss |
| `diversity_score` | How different this chunk is from already-selected chunks | Reduces redundancy in the context window |
| `position_score` | Normalized position of the chunk in the original document | Structural context (e.g., introductions, conclusions) |
| `query_coverage` | How many distinct aspects/entities of the query the chunk addresses | Ensures comprehensive coverage |

### Chromosome Encoding

Each chromosome is a **real-valued vector** encoding the weights and the top-k parameter:

```
chromosome = [w_cosine, w_bm25, w_diversity, w_position, w_coverage, top_k]
```

- `w_cosine`, `w_bm25`, `w_diversity`, `w_position`, `w_coverage`: floats in [0, 1], normalized
- `top_k`: integer, the number of chunks to pass to the LLM (e.g., range 3–15)

### Scoring Formula (Inference Time)

For each candidate chunk `c` in the retrieved set, the reranking score is:

```
score(c) = w_cosine * cosine_sim(c) + w_bm25 * bm25(c) + w_diversity * diversity(c) + w_position * position(c) + w_coverage * coverage(c)
```

Chunks are sorted by `score(c)` descending, and the top `top_k` are passed to the LLM.

### Fitness Function (Offline Training)

> **Important Design Decision**: RAGAS is NOT used as fitness function for the GA.
> RAGAS requires LLM calls per evaluation, making it too slow (100K+ LLM calls),
> non-deterministic (noisy fitness landscape), and indirect (measures answer quality,
> not retrieval quality). Instead, we use **ground-truth retrieval metrics**.

The fitness of a chromosome is evaluated by:
1. Taking a **ground-truth dataset** of `{question, relevant_chunk_ids}` pairs
2. For each question, applying the chromosome's weights to rerank the retrieved chunks
3. Comparing the reranked order against the known relevant chunks
4. Computing **retrieval-level metrics** — no LLM calls needed

```
fitness(chromosome) = α * NDCG@k + β * Precision@k + γ * Recall@k + δ * MRR
```

Where (default: α=0.3, β=0.2, γ=0.3, δ=0.2):
- **NDCG@k** (Normalized Discounted Cumulative Gain): measures ranking quality — are relevant chunks near the top?
- **Precision@k**: of the top-k chunks selected, how many are actually relevant?
- **Recall@k**: of all relevant chunks, how many made it into the top-k?
- **MRR** (Mean Reciprocal Rank): how high is the first relevant chunk?

This approach is **fast** (milliseconds per evaluation), **deterministic** (same input = same score),
and **directly measures retrieval quality** (what the GA is actually optimizing).

### GA Structure (Based on Dao et al., 2017)

The GA uses the innovative structure from the reference paper:

1. **Adaptive Restarting Mechanism**: If the best fitness doesn't improve for `A` consecutive generations, the GA restarts with a new random population (keeping elite chromosomes). This prevents getting stuck in local optima of the weight space.

2. **Local Solution Generation**: In each generation, the best chromosome is slightly modified (one weight at a time, increased or decreased by a random amount) to generate local candidate solutions. This fine-tunes weights near the current best.

3. **Taguchi Parameter Tuning**: The GA's own parameters (population size, crossover rate, mutation rate, restarting condition, local solution %, step size, reduction rate) are tuned using Taguchi Orthogonal Array Design.

## 3. Project Architecture

### File Structure

```
project/
├── main_pipeline.ipynb          # Main notebook — orchestrates everything
├── preprocessor.py              # Document preprocessing (chunking, cleaning)
├── retriever.py                 # ChromaDB retriever class
├── ga_reranker/                 # NEW: GA-optimized reranker framework
│   ├── __init__.py
│   ├── reranker.py              # GAReranker class (plug-in to pipeline)
│   ├── scoring.py               # Multi-criteria scoring functions
│   ├── genetic_algorithm.py     # GA engine (adaptive restart, local search)
│   ├── fitness.py               # Fitness evaluation using retrieval metrics
│   ├── chromosome.py            # Chromosome encoding/decoding
│   └── config.py                # GA parameters and search ranges
├── evaluation/
│   ├── retrieval_metrics.py     # NDCG, Precision, Recall, MRR computation
│   ├── ragas_evaluator.py       # RAGAS wrapper (end-to-end eval only)
│   └── results/                 # Evaluation results and plots
├── datasets/
│   ├── ground_truth_train.json  # GA training set (35-40 questions)
│   ├── ground_truth_test.json   # Held-out test set (15-20 questions)
│   └── README.md                # Dataset creation methodology
└── trained_weights/
    └── best_weights.json        # Saved optimal weights after GA training
```

### Class Interfaces

#### GAReranker (the plug-in class)

```python
class GAReranker:
    """
    Plug-in reranker for the RAG pipeline.
    - During training: runs GA to find optimal weights
    - During inference: applies learned weights for instant reranking
    """
    
    def __init__(self, weights_path: str = None):
        """Load pre-trained weights or use defaults."""
        pass
    
    def train(self, retriever, ground_truth: list[dict], ga_config: dict):
        """
        Run GA optimization offline using ground-truth retrieval metrics.
        
        Args:
            retriever: the existing retriever class instance
            ground_truth: list of {
                "question": str,
                "relevant_chunk_ids": list[str],
                "ground_truth_answer": str  # optional, for RAGAS eval later
            }
            ga_config: GA hyperparameters (population_size, crossover_rate, etc.)
        
        Returns:
            best_weights: dict with optimal scoring weights
        """
        pass
    
    def rerank(self, query: str, retrieved_chunks: list[dict]) -> list[dict]:
        """
        Apply learned weights to rerank chunks. Called at inference time.
        
        Args:
            query: the user's question
            retrieved_chunks: list of chunks from retriever, each with metadata
        
        Returns:
            reranked_chunks: top-k chunks sorted by multi-criteria score
        """
        pass
    
    def save_weights(self, path: str):
        """Save trained weights to JSON."""
        pass
    
    def load_weights(self, path: str):
        """Load pre-trained weights from JSON."""
        pass
```

#### GeneticAlgorithm (the GA engine)

```python
class GeneticAlgorithm:
    """
    GA with adaptive restarting and local solution generation.
    Based on the framework from Dao et al. (2017).
    """
    
    def __init__(self, config: GAConfig):
        pass
    
    def run(self, fitness_fn: Callable) -> Chromosome:
        """
        Execute GA optimization loop.
        
        Args:
            fitness_fn: callable that takes a chromosome and returns fitness score
        
        Returns:
            best_chromosome: the chromosome with highest fitness
        """
        pass
    
    def _generate_population(self, size: int) -> list[Chromosome]:
        pass
    
    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple:
        pass
    
    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        pass
    
    def _generate_local_solutions(self, best: Chromosome) -> list[Chromosome]:
        """Generate neighbors of the best chromosome by modifying one weight at a time."""
        pass
    
    def _should_restart(self, generation_history: list[float]) -> bool:
        """Check adaptive restarting condition."""
        pass
    
    def _restart(self, elite_chromosomes: list[Chromosome]) -> list[Chromosome]:
        """Restart with new random population, keeping elites."""
        pass
```

### Integration in main_pipeline.ipynb

```python
# Existing imports
from preprocessor import Preprocessor
from retriever import Retriever

# New import
from ga_reranker import GAReranker

# === TRAINING PHASE (run once) ===
import json
with open("datasets/ground_truth_train.json") as f:
    train_data = json.load(f)

reranker = GAReranker()
reranker.train(
    retriever=my_retriever,
    ground_truth=train_data,
    ga_config={"population_size": 50, "max_generations": 100, ...}
)
reranker.save_weights("trained_weights/best_weights.json")

# === INFERENCE PHASE ===
reranker = GAReranker(weights_path="trained_weights/best_weights.json")

query = "What is the torque specification for the main bearing cap bolts?"
raw_chunks = my_retriever.retrieve(query, top_n=30)  # Get more than needed
reranked_chunks = reranker.rerank(query, raw_chunks)  # GA-optimized selection
answer = my_llm.generate(query, context=reranked_chunks)
```

## 4. Key Design Decisions

### Why Offline Training with Retrieval Metrics (not RAGAS)?
Using RAGAS as fitness function would require ~100K+ LLM calls (50 chromosomes × 100 generations × 20 questions). Ground-truth retrieval metrics (NDCG, Precision, Recall, MRR) are computed in milliseconds, are deterministic, and directly measure what the GA optimizes: retrieval quality. RAGAS is used only for final end-to-end evaluation.

### Why These Specific Scoring Criteria?
- **Cosine similarity**: preserves the baseline semantic signal
- **BM25**: industrial manuals have specific terminology (part numbers, codes) where exact match matters
- **Diversity**: avoids feeding the LLM redundant context chunks
- **Position**: in manuals, section headers, introductions, and safety warnings have structural significance
- **Query coverage**: complex queries about procedures may need chunks from multiple document sections

### Extensibility
The scoring criteria are modular. New criteria can be added (e.g., `recency_score`, `section_type_score`) by extending the chromosome with additional weights. The GA will automatically learn their optimal contribution.

## 5. Ground-Truth Dataset Creation

### Why Ground Truth (not RAGAS) for GA Training

| Aspect | RAGAS as Fitness | Ground Truth as Fitness |
|--------|-----------------|------------------------|
| Speed per eval | ~seconds (LLM calls) | ~milliseconds (no LLM) |
| Determinism | Non-deterministic (LLM variance) | Fully deterministic |
| What it measures | End-to-end answer quality (indirect) | Retrieval quality (direct) |
| Cost for GA | ~100K+ LLM calls | Zero LLM calls |
| Fitness landscape | Noisy → hard convergence | Clean → fast convergence |

RAGAS is reserved for **final end-to-end evaluation** (before/after comparison in the thesis).

### Manual Sampling Strategy — Avoiding Overfitting

The full database will contain 20-30 manuals. The GA trains on a **representative sample** of 4-5 manuals. The key risk is that the GA overfits to characteristics of the sampled manuals.

**Mitigation: Stratified sampling by manual type.** Select manuals that cover the structural diversity of the full database:

| Slot | Manual Characteristic | What It Tests |
|------|----------------------|---------------|
| 1 | Large manual (many chapters/chunks) | Diversity scoring, redundancy handling |
| 2 | Small/concise manual | Sparse retrieval, precision under few candidates |
| 3 | Specification-heavy (tables, part numbers) | BM25/keyword scoring, exact match importance |
| 4 | Procedure-heavy (step-by-step instructions) | Position scoring, sequential chunk relevance |
| 5 | Mixed content (specs + procedures + safety) | General balance across all criteria |

This ensures the GA encounters all content archetypes present in the full database.

### Question Types

Create 50-60 questions covering 5 types, roughly balanced:

| Type | Example | What It Stresses |
|------|---------|-----------------|
| Factual lookup | "What is the torque spec for X bolt?" | Precision, BM25 (exact terms) |
| Procedural | "How do I replace component Y?" | Position score, sequential chunks |
| Cross-section | "What tools are needed for Z maintenance?" | Query coverage, diversity |
| Troubleshooting | "Engine makes X noise — what's wrong?" | Semantic similarity, broad retrieval |
| Safety | "What precautions before doing Y?" | Position score (safety sections) |

### Dataset Format

```json
{
  "questions": [
    {
      "id": "q001",
      "question": "What is the recommended oil viscosity for the hydraulic system?",
      "source_manual": "manual_003_hydraulic_pump.pdf",
      "question_type": "factual_lookup",
      "relevant_chunk_ids": ["chunk_003_042", "chunk_003_043", "chunk_003_107"],
      "ground_truth_answer": "The recommended oil viscosity is ISO VG 46 for operating temperatures between 15°C and 40°C.",
      "notes": "Answer spans two adjacent chunks (042-043) plus a reference table (107)"
    }
  ]
}
```

### Data Split

```
Total: 50-60 questions from 4-5 manuals
├── Training set (35-40 questions)
│   └── Used as fitness function for GA optimization
├── Test set (15-20 questions)
│   ├── ~10 questions from manuals IN the training set (in-distribution)
│   └── ~5-10 questions from manuals NOT in training set (out-of-distribution)
```

The **out-of-distribution test questions** are critical — they measure whether the learned weights generalize beyond the training manuals. This directly addresses the overfitting concern and is very convincing in the thesis.

### Annotation Process

1. For each selected manual, read the table of contents and identify key sections
2. Write 10-12 questions per manual, covering all 5 question types
3. For each question, **run the retriever with high top-N (e.g., 50)** and manually identify which returned chunks are truly relevant
4. Record the `chunk_ids` of relevant chunks — this is the ground truth
5. Optionally, use an LLM to help generate candidate questions, but **always manually verify** the relevant_chunk_ids

## 6. Evaluation Plan

### Two-Level Evaluation

#### Level 1: Retrieval Quality (primary, used throughout development)

| Metric | What It Measures |
|--------|-----------------|
| NDCG@k | Are relevant chunks ranked near the top? |
| Precision@k | Of the top-k, how many are relevant? |
| Recall@k | Of all relevant chunks, how many are in top-k? |
| MRR | How high is the first relevant chunk? |

Computed on both in-distribution and out-of-distribution test sets separately.

#### Level 2: End-to-End Quality (RAGAS, final comparison only)

| Metric | What It Measures |
|--------|-----------------|
| Faithfulness | Is the answer grounded in the provided context? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are the retrieved chunks relevant? |
| Context Recall | Do the chunks cover what's needed for the answer? |

Run once on the test set: baseline (cosine-only) vs proposed (GA-reranked).

### Experiments Table (for thesis)

| Experiment | Description | Metrics |
|-----------|-------------|---------|
| Baseline | Naive RAG, cosine-only top-k | Retrieval + RAGAS |
| Proposed GA | Full GA-optimized reranker | Retrieval + RAGAS |
| Ablation: no BM25 | Remove w_bm25, retrain GA | Retrieval only |
| Ablation: no diversity | Remove w_diversity, retrain GA | Retrieval only |
| Ablation: no position | Remove w_position, retrain GA | Retrieval only |
| Ablation: no coverage | Remove w_coverage, retrain GA | Retrieval only |
| Generalization | Test on out-of-distribution manuals | Retrieval + RAGAS |
| Convergence | GA fitness over generations | Fitness curve plot |

### Convergence Analysis
Plot GA fitness over generations to show convergence behavior and the effect of adaptive restarting.

## 7. Known Limitations & Future Work

- **Weights are optimized on a sample**: The 4-5 manual sample may not capture all edge cases in 20-30 manuals. Future work: periodic retraining as the database grows, or online weight adaptation.
- **Static weights**: The same weights apply to all query types. Future work: query-type-aware weight selection (e.g., different weights for factual vs procedural queries).
- **Unconstrained optimization**: The GA optimizes continuous weights. Future work: explore discrete choices (e.g., which scoring criteria to include/exclude) via mixed-integer GA.
- **Single embedding model**: Results depend on the chosen embedding model. Future work: test robustness across different embedding models.
## 8. Αλλαγή του κώδικα για αναγνώριση των σωστών chunks 
- Τροποποίηση του κώδικα του retriever ώστε να εκτυπώνει τα chunks τα οποία διαλέγει , με πλήρες κείμενο και όνομα , για να μπορώ να διαλεξώ τα σωστά για το dataset. 