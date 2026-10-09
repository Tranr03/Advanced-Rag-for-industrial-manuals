"""
GAReranker — plug-in module for the RAG pipeline.
Training: runs GA to learn optimal reranking weights.
Inference: applies learned weights for instant reranking (no GA at query time).

Adaptation note: all ChromaDB queries use query_embeddings (not query_texts)
because the collection was built with a custom SentenceTransformer encoder.
The rerank() method accepts chunks in the RAGRetriever format
{text, metadata, score} as well as the internal format {content, cosine_score,
chunk_id, metadata}.
"""
import json
import re
import numpy as np
import pandas as pd

from .config import GAConfig, ChromosomeConfig, ScoringConfig
from .chromosome import Chromosome
from .genetic_algorithm import GeneticAlgorithm
from .fitness import FitnessEvaluator
from .scoring import compute_all_scores
from .retrieval_metrics import compute_all_metrics, composite_fitness


class GAReranker:

    def __init__(self, weights_path: str = None):
        """
        Args:
            weights_path: path to pre-trained weights JSON.
                          If None, initialises with equal weights.
        """
        self.weights    = None
        self.ga_history = None

        if weights_path:
            self.load_weights(weights_path)
        else:
            self.weights = {
                "w_cosine":    0.2,
                "w_bm25":      0.2,
                "w_diversity": 0.2,
                "w_position":  0.2,
                "w_coverage":  0.2,
                "top_k":       5,
            }

    # ── Training ───────────────────────────────────────────────────────────

    def train(self, collection, embedding_model, training_data,
              ga_config: dict = None, scoring_config: dict = None,
              verbose: bool = True) -> dict:
        """
        Run GA optimisation to find best reranking weights.

        Args:
            collection:      ChromaDB collection (retriever.collection)
            embedding_model: SentenceTransformer (retriever.embedding_model)
            training_data:   DataFrame or list of dicts with
                             "question" and "relevant_chunk_ids" columns
            ga_config:       dict of GAConfig overrides
            scoring_config:  dict of ScoringConfig overrides
            verbose:         print progress

        Returns:
            dict of best weights found
        """
        ga_cfg      = GAConfig(**(ga_config or {}))
        scoring_cfg = ScoringConfig(**(scoring_config or {}))
        chromo_cfg  = ChromosomeConfig()

        train_list = training_data.to_dict("records") \
            if isinstance(training_data, pd.DataFrame) else list(training_data)

        if verbose:
            print("Pre-retrieving chunks for all training questions...")

        evaluator = FitnessEvaluator(
            collection=collection,
            embedding_model=embedding_model,
            training_data=train_list,
            scoring_config=scoring_cfg,
            ga_config=ga_cfg,
            retrieval_top_n=scoring_cfg.retrieval_top_n,
        )

        if verbose:
            baseline = evaluator.evaluate_baseline()
            print(f"\nBaseline (cosine-only):")
            print(f"  NDCG: {baseline['ndcg']:.4f} | "
                  f"P: {baseline['precision']:.4f} | "
                  f"R: {baseline['recall']:.4f} | "
                  f"MRR: {baseline['mrr']:.4f}")
            print(f"\nStarting GA optimisation...\n")

        ga = GeneticAlgorithm(ga_cfg, chromo_cfg)
        best = ga.run(fitness_fn=evaluator.evaluate, verbose=verbose)

        self.weights    = best.to_weights_dict()
        self.ga_history = ga.get_convergence_data()

        if verbose:
            print(f"\nOptimal weights found:")
            for k, v in self.weights.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return self.weights

    # ── Inference ──────────────────────────────────────────────────────────

    def rerank(self, query: str, chunks: list,
               scoring_config: ScoringConfig = None) -> list:
        """
        Apply learned weights to rerank chunks at inference time.

        Accepts chunks in either format:
          - RAGRetriever format: {text, metadata, score}
          - Internal format:     {chunk_id, content, cosine_score, metadata}

        Returns top_k chunks in reranked order (same format as input).
        """
        if self.weights is None:
            raise ValueError("No weights loaded. Call train() or load_weights() first.")

        cfg   = scoring_config or ScoringConfig()
        top_k = self.weights["top_k"]

        # Normalise chunk format
        normalised = []
        for i, c in enumerate(chunks):
            normalised.append({
                "chunk_id":     c.get("chunk_id", f"chunk_{i}"),
                "content":      c.get("content") or c.get("text", ""),
                "cosine_score": c.get("cosine_score") or c.get("score", 0.0),
                "metadata":     c.get("metadata", {}),
                "_original":    c,
            })

        avg_len = (
            np.mean([len(re.findall(r'\b\w+\b', c["content"])) for c in normalised])
            if normalised else 1.0
        )

        # Greedy reranking
        remaining = list(range(len(normalised)))
        selected_norm = []
        selected_orig = []

        for _ in range(min(top_k, len(normalised))):
            best_idx, best_score = None, -1.0
            for idx in remaining:
                chunk  = normalised[idx]
                scores = compute_all_scores(query, chunk, selected_norm, avg_len, cfg)
                weighted = (
                    self.weights["w_cosine"]    * scores["cosine"] +
                    self.weights["w_bm25"]      * scores["bm25"] +
                    self.weights["w_diversity"] * scores["diversity"] +
                    self.weights["w_position"]  * scores["position"] +
                    self.weights["w_coverage"]  * scores["coverage"]
                )
                if weighted > best_score:
                    best_score, best_idx = weighted, idx

            if best_idx is not None:
                selected_norm.append(normalised[best_idx])
                selected_orig.append(normalised[best_idx]["_original"])
                remaining.remove(best_idx)

        return selected_orig

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(self, collection, embedding_model, test_data,
                 use_weights: bool = True,
                 scoring_config: ScoringConfig = None) -> dict:
        """
        Evaluate reranking performance on a test dataset.

        Args:
            collection:      ChromaDB collection
            embedding_model: SentenceTransformer
            test_data:       DataFrame or list of dicts
            use_weights:     True = GA weights, False = cosine-only baseline
            scoring_config:  optional ScoringConfig

        Returns:
            dict with average metrics: ndcg, precision, recall, mrr, composite
        """
        cfg       = scoring_config or ScoringConfig()
        test_list = test_data.to_dict("records") \
            if isinstance(test_data, pd.DataFrame) else list(test_data)

        all_metrics = {"ndcg": [], "precision": [], "recall": [], "mrr": []}

        for item in test_list:
            question = item["question"]
            raw_ids  = item.get("relevant_chunk_ids", "")
            if isinstance(raw_ids, str):
                relevant_ids = {cid.strip() for cid in raw_ids.split(",") if cid.strip()}
            else:
                relevant_ids = set(raw_ids)

            if not relevant_ids:
                continue

            query_emb = embedding_model.encode([question],
                                               show_progress_bar=False).tolist()
            safe_n = min(cfg.retrieval_top_n, collection.count())
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

            if use_weights and self.weights:
                reranked     = self.rerank(question, chunks, cfg)
                reranked_ids = [c.get("chunk_id", "") for c in reranked]
                k            = self.weights["top_k"]
            else:
                reranked_ids = [c["chunk_id"] for c in chunks]
                k            = 5

            metrics = compute_all_metrics(reranked_ids, relevant_ids, k)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

        result = {key: float(np.mean(vals)) for key, vals in all_metrics.items() if vals}
        result["composite"] = composite_fitness(result)
        return result

    # ── Persistence ────────────────────────────────────────────────────────

    def save_weights(self, path: str):
        with open(path, "w") as f:
            json.dump(self.weights, f, indent=2)
        print(f"Weights saved to {path}")

    def load_weights(self, path: str):
        with open(path, "r") as f:
            self.weights = json.load(f)
        print(f"Weights loaded from {path}: {self.weights}")
