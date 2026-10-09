"""
Fitness evaluation for GA chromosomes.
Pre-retrieves chunks for all training questions once, then only does
scoring + reranking per fitness call (fast inner loop).

Adaptation note: uses query_embeddings (not query_texts) because the
ChromaDB collection was built with a custom SentenceTransformer encoder.
"""
import re
import numpy as np
from collections import Counter
from .scoring import compute_all_scores
from .retrieval_metrics import compute_all_metrics, composite_fitness
from .config import ScoringConfig, GAConfig


class FitnessEvaluator:
    """
    Evaluates chromosome fitness against a ground-truth dataset.
    Chunks are pre-retrieved once during __init__ to keep GA iterations fast.
    """

    def __init__(self, collection, embedding_model, training_data: list,
                 scoring_config: ScoringConfig = None,
                 ga_config: GAConfig = None,
                 retrieval_top_n: int = 50):
        """
        Args:
            collection:      ChromaDB collection object
            embedding_model: SentenceTransformer instance (same one used by RAGRetriever)
            training_data:   list of dicts with "question" and "relevant_chunk_ids"
            scoring_config:  ScoringConfig instance
            ga_config:       GAConfig instance (for fitness weights)
            retrieval_top_n: how many chunks to retrieve per question
        """
        self.collection     = collection
        self.scoring_config = scoring_config or ScoringConfig()
        self.ga_config      = ga_config      or GAConfig()
        self.retrieval_top_n = retrieval_top_n

        self.query_data = []
        for item in training_data:
            question = item["question"]

            raw_ids = item.get("relevant_chunk_ids", "")
            if isinstance(raw_ids, str):
                relevant_ids = {cid.strip() for cid in raw_ids.split(",") if cid.strip()}
            else:
                relevant_ids = set(raw_ids)

            if not relevant_ids:
                continue

            # Embed question and retrieve from ChromaDB
            query_emb = embedding_model.encode([question],
                                               show_progress_bar=False).tolist()
            safe_n = min(retrieval_top_n, collection.count())
            raw = collection.query(
                query_embeddings=query_emb,
                n_results=safe_n,
                include=["documents", "metadatas", "distances"],
            )

            chunks = [
                {
                    "chunk_id":     raw["ids"][0][i],
                    "content":      raw["documents"][0][i],
                    "cosine_score": 1.0 - raw["distances"][0][i],
                    "metadata":     raw["metadatas"][0][i] if raw["metadatas"] else {},
                }
                for i in range(len(raw["ids"][0]))
            ]

            for c in chunks:
                words = re.findall(r'\b\w+\b', c["content"].lower())
                c["_words"]        = words
                c["_word_set"]     = set(words)
                c["_word_counter"] = Counter(words)

            avg_len = (
                np.mean([len(c["_words"]) for c in chunks])
                if chunks else 1.0
            )

            self.query_data.append({
                "question":       question,
                "relevant_ids":   relevant_ids,
                "chunks":         chunks,
                "avg_chunk_length": avg_len,
            })

        print(f"FitnessEvaluator ready: {len(self.query_data)} questions, "
              f"{retrieval_top_n} candidates each.")

    def evaluate(self, chromosome) -> float:
        """
        Evaluate a chromosome's fitness averaged over all training questions.
        For each question: rerank candidates with chromosome's weights → metrics → fitness.
        """
        weights = chromosome.to_weights_dict()
        top_k   = weights["top_k"]

        scores = []
        for qd in self.query_data:
            reranked_ids = self._greedy_rerank(
                query=qd["question"],
                chunks=qd["chunks"],
                weights=weights,
                avg_chunk_length=qd["avg_chunk_length"],
                top_k=top_k,
            )
            metrics = compute_all_metrics(reranked_ids, qd["relevant_ids"], top_k)
            scores.append(composite_fitness(
                metrics,
                self.ga_config.fitness_alpha,
                self.ga_config.fitness_beta,
                self.ga_config.fitness_gamma,
                self.ga_config.fitness_delta,
            ))
        return float(np.mean(scores)) if scores else 0.0

    def _greedy_rerank(self, query: str, chunks: list, weights: dict,
                       avg_chunk_length: float, top_k: int) -> list:
        """
        Select chunks greedily, updating diversity scores after each pick.
        Returns ordered list of chunk IDs.
        """
        remaining  = list(range(len(chunks)))
        selected   = []
        selected_ids = []

        for _ in range(min(top_k, len(chunks))):
            best_idx, best_score = None, -1.0
            for idx in remaining:
                chunk  = chunks[idx]
                scores = compute_all_scores(query, chunk, selected,
                                            avg_chunk_length, self.scoring_config)
                weighted = (
                    weights["w_cosine"]    * scores["cosine"] +
                    weights["w_bm25"]      * scores["bm25"] +
                    weights["w_diversity"] * scores["diversity"] +
                    weights["w_position"]  * scores["position"] +
                    weights["w_coverage"]  * scores["coverage"]
                )
                if weighted > best_score:
                    best_score, best_idx = weighted, idx

            if best_idx is not None:
                selected.append(chunks[best_idx])
                selected_ids.append(chunks[best_idx]["chunk_id"])
                remaining.remove(best_idx)

        return selected_ids

    def evaluate_baseline(self, k: int = 5) -> dict:
        """Evaluate cosine-only baseline (ChromaDB default order). Used for comparison."""
        all_metrics = {"ndcg": [], "precision": [], "recall": [], "mrr": []}
        for qd in self.query_data:
            baseline_ids = [c["chunk_id"] for c in qd["chunks"]]
            metrics = compute_all_metrics(baseline_ids, qd["relevant_ids"], k)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
        return {key: float(np.mean(vals)) for key, vals in all_metrics.items()}
