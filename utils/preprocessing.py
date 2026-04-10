# utils/preprocessing.py
"""
Text preprocessing utilities.
Handles loading PDFs and TXT files, cleaning raw text,
and normalizing whitespace / special characters.
"""

import re
import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def load_pdf(filepath: str) -> str:
    """Extract raw text from every page of a PDF."""
    logger.info(f"Loading PDF: {filepath}")
    doc = fitz.open(filepath)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
        else:
            logger.debug(f"  Page {page_num + 1} has no extractable text (may be scanned).")
    doc.close()
    full_text = "\n\n".join(pages)
    logger.info(f"  Extracted {len(full_text)} characters from {len(pages)} pages.")
    return full_text


def load_txt(filepath: str) -> str:
    """Read a plain-text file."""
    logger.info(f"Loading TXT: {filepath}")
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    logger.info(f"  Read {len(text)} characters.")
    return text


def load_document(filepath: str) -> str:
    """
    Dispatch to the correct loader based on file extension.
    Supports .pdf and .txt files.
    """
    lower = filepath.lower()
    if lower.endswith(".pdf"):
        return load_pdf(filepath)
    elif lower.endswith(".txt"):
        return load_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}. Only .pdf and .txt are supported.")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Perform lightweight cleaning:
    - Collapse repeated blank lines to one
    - Normalize whitespace within lines
    - Strip leading/trailing whitespace
    - Remove non-printable control characters
    """
    # Remove null bytes and non-printable control chars (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ consecutive blank lines → double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse spaces/tabs within a single line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()
