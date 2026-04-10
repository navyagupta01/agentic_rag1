# rag/pipeline.py
"""
Adaptive Multi-Stage RAG Pipeline — orchestrator.

Ties together all modules:
  preprocessing → chunking → embedder/retriever → classifier
  → adaptive retrieval → ranker → generator
"""

import logging
from typing import List, Optional

from utils.preprocessing import load_document, clean_text
from utils.chunking import chunk_text
from rag.retriever import HybridRetriever
from rag.classifier import classify_query, get_retrieval_k
from rag.ranker import rerank
from rag.generator import generate_answer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end pipeline.

    Usage
    -----
    pipeline = RAGPipeline()
    pipeline.ingest_file("research.pdf")
    result = pipeline.query("What are the key findings?")
    print(result["answer"])
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        semantic_weight: float = 0.6,
        llm_model: str = "mistralai/mistral-7b-instruct",
        candidate_pool: int = 10,    # how many candidates retriever fetches
        use_cache: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model = llm_model
        self.candidate_pool = candidate_pool
        self.use_cache = use_cache

        self.retriever = HybridRetriever(semantic_weight=semantic_weight)
        self.all_chunks: List[str] = []
        self.ingested_files: List[str] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_file(self, filepath: str) -> int:
        """
        Load, clean, chunk, and index a document.

        Returns
        -------
        Number of chunks added.
        """
        logger.info(f"=== Ingesting: {filepath} ===")

        raw_text = load_document(filepath)
        clean = clean_text(raw_text)
        chunks = chunk_text(clean, self.chunk_size, self.chunk_overlap)

        if not chunks:
            logger.warning(f"No chunks extracted from {filepath}.")
            return 0

        self.all_chunks.extend(chunks)
        self.retriever.build(self.all_chunks)   # rebuild full index
        self.ingested_files.append(filepath)

        logger.info(f"Ingested {len(chunks)} chunks. Total in index: {len(self.all_chunks)}")
        return len(chunks)

    def ingest_text(self, text: str, source_name: str = "manual_input") -> int:
        """
        Ingest raw text directly (useful for testing or pasted content).

        Returns
        -------
        Number of chunks added.
        """
        logger.info(f"=== Ingesting text: '{source_name}' ===")
        clean = clean_text(text)
        chunks = chunk_text(clean, self.chunk_size, self.chunk_overlap)

        if not chunks:
            return 0

        self.all_chunks.extend(chunks)
        self.retriever.build(self.all_chunks)
        self.ingested_files.append(source_name)
        logger.info(f"Ingested {len(chunks)} chunks from text. Total: {len(self.all_chunks)}")
        return len(chunks)

    def reset(self) -> None:
        """Clear all ingested data."""
        self.all_chunks = []
        self.ingested_files = []
        self.retriever = HybridRetriever(
            semantic_weight=self.retriever.alpha
        )
        logger.info("Pipeline reset. All data cleared.")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        override_type: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict:
        """
        Run the full RAG pipeline for a user query.

        Parameters
        ----------
        user_query    : the question to answer
        override_type : force a query type ("factual" | "summarization" | "analytical")
        temperature   : LLM temperature

        Returns
        -------
        dict with keys:
          answer       : str
          query_type   : str
          sources      : List[str]
          model        : str
          cached       : bool
          chunks_used  : int
        """
        if not self.retriever.is_ready:
            raise RuntimeError(
                "No documents have been ingested. "
                "Call ingest_file() or ingest_text() first."
            )

        # 1. Classify
        q_type = override_type if override_type else classify_query(user_query)
        final_k = get_retrieval_k(q_type)
        logger.info(f"Query type: {q_type} | Will use top {final_k} chunks")

        # 2. Hybrid retrieval (fetch candidate_pool candidates)
        candidates = self.retriever.retrieve(
            user_query, top_k=self.candidate_pool
        )
        logger.info(f"Retrieved {len(candidates)} candidates from hybrid retriever.")

        # 3. Adaptive re-ranking
        rank_mode = "diverse" if q_type == "analytical" else "standard"
        ranked = rerank(candidates, user_query, final_k=final_k, mode=rank_mode)
        logger.info(f"Re-ranked to {len(ranked)} final chunks (mode={rank_mode}).")

        # 4. Generate answer
        result = generate_answer(
            query=user_query,
            context_chunks=ranked,
            query_type=q_type,
            model=self.llm_model,
            temperature=temperature,
            use_cache=self.use_cache,
        )

        result["chunks_used"] = len(ranked)
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        return len(self.ingested_files)

    @property
    def chunk_count(self) -> int:
        return len(self.all_chunks)

    @property
    def is_ready(self) -> bool:
        return self.retriever.is_ready
