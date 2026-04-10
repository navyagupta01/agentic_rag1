# rag/ranker.py
"""
Re-ranking layer.

After the hybrid retriever returns ~10 candidates, the ranker
refines the selection by applying additional signals:

1. Length penalty  – very short chunks are penalised (likely noise).
2. Query-term coverage – bonus if the chunk contains many distinct
   query tokens (not captured by TF-IDF).
3. Diversity selection for analytical queries – Maximal Marginal
   Relevance (MMR) to avoid redundant chunks.

The ranker operates *after* retrieval and *before* LLM generation.
"""

import re
import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Type alias: (text, combined_score, semantic_score, keyword_score)
Candidate = Tuple[str, float, float, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_token_coverage(chunk: str, query_tokens: set) -> float:
    """
    Fraction of unique query tokens that appear in the chunk.
    Rewards chunks that explicitly mention query keywords.
    """
    if not query_tokens:
        return 0.0
    chunk_lower = chunk.lower()
    hits = sum(1 for tok in query_tokens if tok in chunk_lower)
    return hits / len(query_tokens)


def _length_score(chunk: str, ideal_min: int = 80, ideal_max: int = 800) -> float:
    """
    Returns 1.0 when chunk length is in [ideal_min, ideal_max],
    with a smooth ramp-down outside that range.
    """
    n = len(chunk)
    if n < ideal_min:
        return max(0.0, n / ideal_min)
    if n > ideal_max:
        # Slight penalty for very long chunks (may be noisy)
        return max(0.5, ideal_max / n)
    return 1.0


def _tokenize_query(query: str) -> set:
    """Lower-case word tokens; filter stopwords and very short tokens."""
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "can", "could", "should", "may", "might", "of", "in", "on",
        "at", "to", "for", "with", "by", "from", "about", "into",
        "what", "who", "when", "where", "why", "how", "which", "that",
        "this", "these", "those", "it", "its", "and", "or", "not", "no",
    }
    tokens = re.findall(r"[a-z]+", query.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


# ---------------------------------------------------------------------------
# Main re-ranker
# ---------------------------------------------------------------------------

def rerank(
    candidates: List[Candidate],
    query: str,
    final_k: int,
    mode: str = "standard",   # "standard" | "diverse"
) -> List[Candidate]:
    """
    Re-rank candidates and return the best *final_k*.

    Parameters
    ----------
    candidates : output of HybridRetriever.retrieve()
    query      : original user query
    final_k    : how many results to return
    mode       : "standard" (score-based) or "diverse" (MMR)

    Returns
    -------
    Re-ranked list of Candidate tuples, length ≤ final_k.
    """
    if not candidates:
        return []

    final_k = min(final_k, len(candidates))
    query_tokens = _tokenize_query(query)

    # ---- Compute adjusted scores ----
    adjusted: List[Tuple[float, Candidate]] = []
    for cand in candidates:
        text, combined, sem, kw = cand
        coverage = _query_token_coverage(text, query_tokens)
        length = _length_score(text)

        # Weighted combination
        adj_score = (
            0.65 * combined
            + 0.25 * coverage
            + 0.10 * length
        )
        adjusted.append((adj_score, cand))

    adjusted.sort(key=lambda x: x[0], reverse=True)

    if mode == "standard":
        result = [cand for _, cand in adjusted[:final_k]]
        logger.debug(
            f"Standard rerank top scores: {[round(s, 3) for s, _ in adjusted[:final_k]]}"
        )
        return result

    # ---- Diverse mode: Maximal Marginal Relevance (MMR) ----
    # Use the re-ranked scores as relevance; use text overlap as proxy for similarity.
    selected: List[Candidate] = []
    remaining = [(score, cand) for score, cand in adjusted]

    while len(selected) < final_k and remaining:
        if not selected:
            # First pick: highest relevance
            _, best = remaining.pop(0)
            selected.append(best)
            continue

        # MMR: relevance - λ * max_similarity_to_selected
        lambda_ = 0.5
        mmr_scores = []
        for score, cand in remaining:
            sim_to_selected = max(
                _text_overlap(cand[0], sel[0]) for sel in selected
            )
            mmr = score - lambda_ * sim_to_selected
            mmr_scores.append((mmr, cand))

        mmr_scores.sort(key=lambda x: x[0], reverse=True)
        _, best = mmr_scores[0]

        # Remove best from remaining
        remaining = [
            (s, c) for s, c in remaining if c[0] != best[0]
        ]
        selected.append(best)

    logger.debug(f"MMR rerank selected {len(selected)} diverse chunks.")
    return selected


def _text_overlap(a: str, b: str) -> float:
    """Jaccard similarity of word sets as a diversity proxy."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
