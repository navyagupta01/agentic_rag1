# rag/classifier.py
"""
Query classifier: determines the *intent type* of a user query
so that the retrieval strategy can be adapted.

Three classes are supported:
  - factual      → short precise answers; fetch top 3 chunks
  - summarization → broad overview; fetch top 8 chunks
  - analytical   → multi-facet analysis; fetch top 5 diverse chunks

Approach
--------
Rule-based keyword/pattern matching + lightweight heuristics.
No ML model dependency — works offline, no training needed.
Extremely fast (microseconds per query).

Extend by adding patterns to the PATTERNS dict below.
"""

import re
import logging
from typing import Literal

logger = logging.getLogger(__name__)

QueryType = Literal["factual", "summarization", "analytical"]

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each entry: (compiled_regex, query_type)
# Patterns are checked in order; first match wins.
_RAW_PATTERNS = [
    # ---- Summarization signals ----
    (r"\b(summar(ize|ise|y)|overview|outline|brief|synopsis|gist|tldr|tl;dr|recap|highlights?)\b", "summarization"),
    (r"\bwhat (is|are) (the )?(main|key|primary|overall|general)\b", "summarization"),
    (r"\btell me (about|everything about)\b", "summarization"),
    (r"\bdescribe (the |this )?(document|paper|article|text|report|content)\b", "summarization"),
    (r"\bwhat does (the |this )?(document|paper|article|report) (say|cover|discuss|talk about)\b", "summarization"),

    # ---- Analytical signals ----
    (r"\b(analyz|analys|compar|contrast|evaluat|assess|discuss|examin|investigat|review|critique|interpret)\b", "analytical"),
    (r"\bwhy (is|are|does|do|did|was|were)\b", "analytical"),
    (r"\bhow does .* (work|function|operate|affect|influence|impact)\b", "analytical"),
    (r"\bwhat (are|is) the (impact|effect|implication|consequence|cause|reason|advantage|disadvantage|benefit|risk)\b", "analytical"),
    (r"\bpros? and cons?\b", "analytical"),
    (r"\bstrengths? (and|&|or) weaknesses?\b", "analytical"),
    (r"\brelationship between\b", "analytical"),
    (r"\bhow (does|did|do) .* (relate|connect|link|contribute)\b", "analytical"),
    (r"\b(difference|distinction|similarity) between\b", "analytical"),

    # ---- Factual signals (default bucket, but also explicit patterns) ----
    (r"\b(who|what|when|where|which)\b", "factual"),
    (r"\bhow (many|much|long|old|far|tall|big|large|small|fast|often)\b", "factual"),
    (r"\b(define|definition|meaning|name|list|enumerate|identify)\b", "factual"),
    (r"\bis .* (true|false|correct|accurate)\b", "factual"),
]

PATTERNS = [(re.compile(pat, re.IGNORECASE), qtype) for pat, qtype in _RAW_PATTERNS]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

# How many retrieval chunks each type should use
RETRIEVAL_K: dict = {
    "factual": 3,
    "summarization": 8,
    "analytical": 5,
}


def classify_query(query: str) -> QueryType:
    """
    Classify a query string into one of three intent types.

    Parameters
    ----------
    query : the user's question / instruction

    Returns
    -------
    One of "factual", "summarization", "analytical"
    """
    q = query.strip()
    if not q:
        return "factual"

    # Check each pattern in priority order
    for pattern, qtype in PATTERNS:
        if pattern.search(q):
            logger.debug(f"Query classified as '{qtype}' via pattern: {pattern.pattern[:40]}")
            return qtype

    # Default: factual (short specific queries with no clear signal)
    logger.debug("Query classification: no pattern matched → defaulting to 'factual'")
    return "factual"


def get_retrieval_k(query_type: QueryType) -> int:
    """Return the recommended number of chunks to retrieve for a query type."""
    return RETRIEVAL_K[query_type]
