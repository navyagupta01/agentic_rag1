# rag/generator.py
"""
LLM answer generation via OpenRouter API.

Features:
  - Uses OpenRouter's /v1/chat/completions endpoint
  - Adaptive prompts per query type (merged into user role for compatibility)
  - Simple in-memory cache for repeated queries
  - Auto-retry on 429 rate limits
  - Graceful error handling
"""

import os
import time
import hashlib
import logging
import requests
from typing import List, Tuple

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-2-9b-it:free"

# In-memory cache
_cache: dict = {}

# ---------------------------------------------------------------------------
# System prompt templates per query type
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "factual": (
        "You are a precise, factual assistant. "
        "Answer the user's question using ONLY the provided context. "
        "Be concise and direct. If the answer is not in the context, say so. "
        "Format your answer in clear bullet points where appropriate."
    ),
    "summarization": (
        "You are a skilled summarizer. "
        "Using the provided context, produce a comprehensive yet concise summary. "
        "Organise the response under clear headings or bullet points. "
        "Cover all key points present in the context."
    ),
    "analytical": (
        "You are a sharp analytical assistant. "
        "Analyse the user's question in depth using the provided context. "
        "Identify patterns, causes, effects, and relationships. "
        "Structure your response with: (1) Key Observations, (2) Analysis, "
        "(3) Conclusions. Use bullet points within each section."
    ),
}

# ---------------------------------------------------------------------------
# Prompt builder — single user message (no system role for compatibility)
# ---------------------------------------------------------------------------

def _build_user_message(query: str, context_chunks: List[Tuple], query_type: str) -> str:
    system_prompt = SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS["factual"])

    ctx_lines = []
    for i, cand in enumerate(context_chunks, start=1):
        text = cand[0]
        ctx_lines.append(f"[Source {i}]\n{text.strip()}")
    context_str = "\n\n".join(ctx_lines)

    return (
        f"{system_prompt}\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"---\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer based solely on the context above. "
        f"Reference source numbers (e.g. [Source 1]) where relevant."
    )

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(query: str, query_type: str, n_chunks: int) -> str:
    raw = f"{query}|{query_type}|{n_chunks}"
    return hashlib.md5(raw.encode()).hexdigest()

def clear_cache() -> None:
    _cache.clear()
    logger.info("Answer cache cleared.")

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    context_chunks: List[Tuple],
    query_type: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 512,
    use_cache: bool = True,
) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Export it with: export OPENROUTER_API_KEY='your_key_here'"
        )

    # Cache lookup
    ck = _cache_key(query, query_type, len(context_chunks))
    if use_cache and ck in _cache:
        logger.info("Returning cached answer.")
        return {**_cache[ck], "cached": True}

    user_message = _build_user_message(query, context_chunks, query_type)

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Retry loop — handles 429 rate limits
    last_error = None
    for attempt in range(3):
        try:
            logger.info(f"Calling OpenRouter (model={model}, attempt={attempt+1}/3) ...")
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )

            # Rate limited — wait and retry
            if response.status_code == 429:
                wait = 20 * (attempt + 1)
                logger.warning(f"Rate limited (429). Waiting {wait}s ...")
                time.sleep(wait)
                continue

            # Bad request — don't retry, fail fast
            if response.status_code == 400:
                detail = ""
                try:
                    detail = response.json().get("error", {}).get("message", "")
                except Exception:
                    detail = response.text[:300]
                raise RuntimeError(
                    f"OpenRouter API error 400: {detail or response.text[:200]}"
                )

            # Model not found — don't retry, fail fast
            if response.status_code == 404:
                detail = ""
                try:
                    detail = response.json().get("error", {}).get("message", "")
                except Exception:
                    detail = response.text[:300]
                raise RuntimeError(
                    f"OpenRouter API error 404: {detail or 'No endpoints found for this model'}. "
                    "Try switching to a different model in the sidebar."
                )

            response.raise_for_status()

            data = response.json()

            # Guard against empty or malformed response
            choices = data.get("choices")
            if not choices:
                last_error = (
                    f"Model returned an empty response (no choices). "
                    f"Full response: {data}"
                )
                logger.warning(f"Attempt {attempt+1}: {last_error}")
                time.sleep(10)
                continue

            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                finish_reason = choices[0].get("finish_reason", "unknown")
                last_error = (
                    f"Model returned empty content "
                    f"(finish_reason={finish_reason}). "
                    f"Full response: {data}"
                )
                logger.warning(f"Attempt {attempt+1}: {last_error}")
                time.sleep(10)
                continue

            result = {
                "answer": content,
                "sources": [cand[0] for cand in context_chunks],
                "query_type": query_type,
                "model": model,
                "cached": False,
            }

            if use_cache:
                _cache[ck] = {k: v for k, v in result.items()}

            logger.info("Answer generated successfully.")
            return result

        except RuntimeError:
            raise
        except requests.exceptions.Timeout:
            last_error = "Request timed out after 60 seconds."
            logger.warning(f"Attempt {attempt+1} timed out.")
            time.sleep(10)
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = response.json().get("error", {}).get("message", "")
            except Exception:
                detail = response.text[:300]
            last_error = (
                f"OpenRouter API error {response.status_code}: "
                f"{detail or str(e) or 'unknown error'}"
            )
            logger.warning(f"Attempt {attempt+1} failed: {last_error}")
            # Don't retry client errors other than 429
            if response.status_code in (401, 403, 404):
                break
            time.sleep(10)
        except requests.exceptions.ConnectionError:
            last_error = "Connection error. Check your internet connection."
            logger.warning(f"Attempt {attempt+1} connection error.")
            time.sleep(10)
        except Exception as e:
            last_error = f"Unexpected error: {type(e).__name__}: {e}"
            logger.exception(f"Attempt {attempt+1} unexpected error: {e}")
            time.sleep(10)

    raise RuntimeError(
        f"Failed after 3 attempts. Last error: {last_error} "
        "Try switching to a different model in the sidebar."
    )
