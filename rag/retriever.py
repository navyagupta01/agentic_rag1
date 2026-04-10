# rag/retriever.py
"""
Hybrid retriever: semantic (FAISS) + keyword (TF-IDF).

Combines both similarity scores with a configurable weight α:
    combined_score = α * semantic_score + (1-α) * keyword_score

All scores are normalised to [0, 1] before combining so that
neither signal dominates due to scale differences.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.embedder import Embedder

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Manages both the FAISS semantic index and TF-IDF keyword index.

    Parameters
    ----------
    semantic_weight : float in [0, 1]
        Weight given to semantic similarity (α).
        Keyword weight = 1 - α.
    """

    def __init__(self, semantic_weight: float = 0.6):
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be in [0, 1].")
        self.alpha = semantic_weight
        self.embedder = Embedder()
        self.tfidf: TfidfVectorizer = None
        self.tfidf_matrix = None          # shape (n_chunks, vocab)
        self.chunks: List[str] = []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build(self, chunks: List[str]) -> None:
        """
        Index a list of text chunks for both retrieval methods.
        Must be called before any search.
        """
        if not chunks:
            raise ValueError("Chunk list is empty.")

        self.chunks = chunks

        # 1. Semantic index
        self.embedder.build_index(chunks)

        # 2. TF-IDF keyword index
        logger.info("Fitting TF-IDF vectorizer …")
        self.tfidf = TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),      # unigrams + bigrams
            sublinear_tf=True,       # apply log(tf) scaling
            min_df=1,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(chunks)
        logger.info(f"TF-IDF vocabulary size: {len(self.tfidf.vocabulary_)}")

    # ------------------------------------------------------------------
    # Internal search helpers
    # ------------------------------------------------------------------

    def _semantic_scores(self, query: str, top_k: int) -> Dict[int, float]:
        """Return {chunk_index: semantic_score} for top_k semantic hits."""
        results = self.embedder.search(query, top_k=top_k)
        # Map text → index (chunks list is canonical)
        text_to_idx = {text: i for i, text in enumerate(self.chunks)}
        scores = {}
        for text, score in results:
            idx = text_to_idx.get(text)
            if idx is not None:
                scores[idx] = score
        return scores

    def _keyword_scores(self, query: str) -> np.ndarray:
        """Return cosine similarity between query TF-IDF vector and all chunks."""
        query_vec = self.tfidf.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]  # shape (n_chunks,)
        return sims

    @staticmethod
    def _normalise(scores: np.ndarray) -> np.ndarray:
        """Min-max normalise an array to [0, 1]."""
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float, float, float]]:
        """
        Hybrid retrieval.

        Returns
        -------
        List of (chunk_text, combined_score, semantic_score, keyword_score)
        sorted by descending combined_score, length = top_k.
        """
        if not self.chunks:
            raise RuntimeError("Retriever not built. Call build() first.")

        n = len(self.chunks)

        # --- Semantic scores (sparse dict over candidates) ---
        sem_dict = self._semantic_scores(query, top_k=min(top_k * 2, n))

        # Convert to dense array
        sem_arr = np.zeros(n, dtype=np.float32)
        for idx, score in sem_dict.items():
            sem_arr[idx] = score

        # --- Keyword scores (dense array over all chunks) ---
        kw_arr = self._keyword_scores(query).astype(np.float32)

        # --- Normalise both signals ---
        sem_norm = self._normalise(sem_arr)
        kw_norm = self._normalise(kw_arr)

        # --- Combine ---
        combined = self.alpha * sem_norm + (1 - self.alpha) * kw_norm

        # --- Select top_k ---
        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                (
                    self.chunks[idx],
                    float(combined[idx]),
                    float(sem_norm[idx]),
                    float(kw_norm[idx]),
                )
            )

        logger.debug(
            f"Hybrid retrieval: top combined scores = "
            f"{[round(r[1], 3) for r in results[:5]]}"
        )
        return results

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks) and self.embedder.is_ready
