# utils/chunking.py
"""
Text chunking with configurable size and overlap.

Strategy:
  1. Split document into sentences (regex-based, fast).
  2. Group sentences into chunks of ~chunk_size characters.
  3. Overlap between consecutive chunks is achieved by carrying
     the tail of the previous chunk into the next one.

This preserves sentence boundaries, which improves embedding quality
compared to hard character splits.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    """
    Naive but effective sentence splitter.
    Splits on '.', '!', '?' followed by whitespace + uppercase letter,
    or on paragraph boundaries (double newlines).
    """
    # Treat paragraph breaks as hard sentence boundaries
    text = re.sub(r"\n{2,}", " <PARA> ", text)

    # Split on sentence-ending punctuation
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)

    sentences = []
    for s in raw:
        # Further split on <PARA> markers
        parts = s.split("<PARA>")
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[str]:
    """
    Split *text* into overlapping chunks.

    Parameters
    ----------
    text       : cleaned document text
    chunk_size : target character length per chunk
    overlap    : number of characters to repeat from the previous chunk

    Returns
    -------
    List of non-empty string chunks.
    """
    if not text.strip():
        return []

    sentences = _split_sentences(text)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds chunk_size, hard-split it
        if sentence_len > chunk_size:
            # First flush whatever we have
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

            # Hard split the long sentence
            for start in range(0, sentence_len, chunk_size - overlap):
                part = sentence[start : start + chunk_size]
                if part.strip():
                    chunks.append(part.strip())
            continue

        # If adding this sentence exceeds the limit, finalise current chunk
        if current_len + sentence_len > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            # Build the overlap: take sentences from the end of the current chunk
            # until we have ~overlap chars
            overlap_sentences: List[str] = []
            overlap_len = 0
            for prev_sent in reversed(current_chunk):
                if overlap_len + len(prev_sent) <= overlap:
                    overlap_sentences.insert(0, prev_sent)
                    overlap_len += len(prev_sent)
                else:
                    break

            current_chunk = overlap_sentences
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += sentence_len

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(
        f"Chunked document into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={overlap})."
    )
    return chunks
