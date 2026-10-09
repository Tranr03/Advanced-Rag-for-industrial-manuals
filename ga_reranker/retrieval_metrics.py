"""
Retrieval quality metrics for evaluating chunk ranking.
All functions operate on lists of chunk IDs — no LLM calls needed.
"""
import math


def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    if k <= 0:
        return 0.0
    relevant_found = sum(1 for cid in retrieved_ids[:k] if cid in relevant_ids)
    return relevant_found / k


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of all relevant chunks that appear in top-k."""
    if not relevant_ids:
        return 0.0
    relevant_found = sum(1 for cid in retrieved_ids[:k] if cid in relevant_ids)
    return relevant_found / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    Measures whether relevant chunks appear near the top of the ranking.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, cid in enumerate(retrieved_ids[:k])
        if cid in relevant_ids
    )

    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved_ids: list, relevant_ids: set) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant chunk."""
    for i, cid in enumerate(retrieved_ids):
        if cid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_all_metrics(retrieved_ids: list, relevant_ids: set, k: int) -> dict:
    """Compute all 4 retrieval metrics."""
    return {
        "ndcg":      ndcg_at_k(retrieved_ids, relevant_ids, k),
        "precision": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall":    recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr":       mrr(retrieved_ids, relevant_ids),
    }


def composite_fitness(metrics: dict, alpha=0.3, beta=0.2,
                      gamma=0.3, delta=0.2) -> float:
    """
    Combine 4 metrics into a single fitness score.
    fitness = α*NDCG + β*Precision + γ*Recall + δ*MRR
    """
    return (alpha * metrics["ndcg"] +
            beta  * metrics["precision"] +
            gamma * metrics["recall"] +
            delta * metrics["mrr"])
