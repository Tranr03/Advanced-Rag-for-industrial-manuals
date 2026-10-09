"""
Multi-criteria scoring functions for chunk reranking.
Each function returns a float in [0, 1].
"""
import math
import re
from collections import Counter


def cosine_score(chunk: dict) -> float:
    """Return the cosine similarity score from ChromaDB (already 1 - distance)."""
    return max(0.0, min(1.0, chunk.get("cosine_score", 0.0)))


def bm25_score(query: str, chunk: dict, avg_chunk_length: float,
               k1: float = 1.5, b: float = 0.75) -> float:
    """
    BM25-like score between query and chunk text (without IDF).
    Normalized to [0, 1].
    """
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    query_terms = tokenize(query)
    chunk_terms = chunk.get("_words") or tokenize(chunk.get("content", ""))
    if not query_terms or not chunk_terms:
        return 0.0

    chunk_len     = len(chunk_terms)
    chunk_counter = chunk.get("_word_counter") or Counter(chunk_terms)
    score = 0.0
    for term in set(query_terms):
        tf = chunk_counter.get(term, 0)
        if tf > 0:
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (chunk_len / max(avg_chunk_length, 1)))
            score += numerator / denominator

    score = score / len(set(query_terms))
    score = score / (k1 + 1)
    return max(0.0, min(1.0, score))


def diversity_score(chunk: dict, already_selected: list) -> float:
    """
    Jaccard-based diversity: how different is this chunk from already-selected ones.
    Returns 1 - max(Jaccard similarity to any selected chunk).
    Order-dependent — must be used in a greedy selection loop.
    """
    if not already_selected:
        return 1.0

    def word_set(text):
        return set(re.findall(r'\b\w+\b', text.lower()))

    chunk_words = chunk.get("_word_set") or word_set(chunk.get("content", ""))
    if not chunk_words:
        return 0.0

    max_similarity = 0.0
    for selected in already_selected:
        selected_words = selected.get("_word_set") or word_set(selected.get("content", ""))
        if not selected_words:
            continue
        intersection = len(chunk_words & selected_words)
        union = len(chunk_words | selected_words)
        if union > 0:
            max_similarity = max(max_similarity, intersection / union)

    return 1.0 - max_similarity


def position_score(chunk: dict) -> float:
    """
    Inverse log score based on page number.
    Earlier pages score higher (safety warnings, introductions tend to appear first).
    """
    page = chunk.get("metadata", {}).get("page", 1)
    try:
        page = max(1, int(page))
    except (ValueError, TypeError):
        page = 1
    return max(0.0, min(1.0, 1.0 / (1.0 + math.log(1.0 + page))))


def query_coverage_score(query: str, chunk: dict) -> float:
    """
    Fraction of meaningful query terms (stopwords removed) that appear in the chunk.
    """
    STOPWORDS = {
        "what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "under",
        "how", "do", "does", "did", "can", "could", "would", "should", "must",
        "will", "shall", "may", "might", "are", "was", "were", "been", "being",
        "have", "has", "had", "having", "this", "that", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
        "her", "its", "our", "their", "which", "who", "whom", "when", "where",
        "why", "if", "then", "than", "so", "no", "not", "only", "very",
        "be", "describe", "explain", "list", "identify", "specific", "step",
        "procedure", "recommended", "required", "located", "regarding",
    }

    def extract_terms(text):
        return [w for w in re.findall(r'\b\w+\b', text.lower())
                if w not in STOPWORDS and len(w) > 2]

    query_terms = extract_terms(query)
    if not query_terms:
        return 0.0

    chunk_text_lower = chunk.get("content", "").lower()
    covered = sum(1 for term in query_terms if term in chunk_text_lower)
    return covered / len(query_terms)


def compute_all_scores(query: str, chunk: dict, already_selected: list,
                       avg_chunk_length: float, scoring_config=None) -> dict:
    """Compute all 5 scoring criteria for a single chunk."""
    from .config import ScoringConfig
    cfg = scoring_config or ScoringConfig()
    return {
        "cosine":    cosine_score(chunk),
        "bm25":      bm25_score(query, chunk, avg_chunk_length, cfg.bm25_k1, cfg.bm25_b),
        "diversity": diversity_score(chunk, already_selected),
        "position":  position_score(chunk),
        "coverage":  query_coverage_score(query, chunk),
    }
