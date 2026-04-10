# rag/embedder.py
"""
Embedding layer backed by SentenceTransformers and FAISS.

Responsibilities:
  - Encode text chunks into dense vectors.
  - Build / update a FAISS flat-L2 index.
  - Perform semantic nearest-neighbour search.
"""

import logging
import numpy as np
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model — lightweight yet strong on retrieval tasks
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Wraps SentenceTransformer + FAISS index.

    Usage
    -----
    embedder = Embedder()
    embedder.build_index(chunks)
    results = embedder.search("What is X?", top_k=5)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        self.index: faiss.IndexFlatIP = None   # inner-product (cosine after norm)
        self.chunks: List[str] = []
        logger.info(f"Embedding dimension: {self.dimension}")

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts and L2-normalise for cosine similarity via inner product."""
        vectors = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalise → IP == cosine
        )
        return vectors.astype(np.float32)

    def build_index(self, chunks: List[str]) -> None:
        """
        Encode all chunks and load them into a fresh FAISS index.
        Call this whenever the document set changes.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        logger.info(f"Building FAISS index for {len(chunks)} chunks …")
        self.chunks = chunks
        vectors = self._encode(chunks)

        # IndexFlatIP: exact inner-product search (cosine since vectors normalised)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        logger.info(f"FAISS index built. Total vectors: {self.index.ntotal}")

    def add_chunks(self, new_chunks: List[str]) -> None:
        """Incrementally add more chunks to an existing index."""
        if self.index is None:
            return self.build_index(new_chunks)
        vectors = self._encode(new_chunks)
        self.index.add(vectors)
        self.chunks.extend(new_chunks)
        logger.info(f"Added {len(new_chunks)} chunks. Total: {self.index.ntotal}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Return (chunk_text, cosine_score) pairs for the top-k nearest chunks.

        Parameters
        ----------
        query : user query string
        top_k : number of results to retrieve

        Returns
        -------
        List of (text, score) tuples sorted by descending score.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty — returning no results.")
            return []

        k = min(top_k, self.index.ntotal)
        query_vec = self._encode([query])          # shape (1, dim)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))

        logger.debug(f"Semantic search returned {len(results)} results.")
        return results

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0
