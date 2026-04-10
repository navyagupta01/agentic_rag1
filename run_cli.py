#!/usr/bin/env python3
# run_cli.py
"""
Command-line interface for the Adaptive RAG System.
Useful for quick testing without the Streamlit UI.

Usage
-----
python run_cli.py --file path/to/document.pdf --query "What is the main topic?"
python run_cli.py --text "Some long text here..." --query "Summarize this"
"""

import os
import sys
import json
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from rag.pipeline import RAGPipeline


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adaptive Multi-Stage RAG System — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", "-f", help="Path to PDF or TXT document")
    source.add_argument("--text", "-t", help="Raw text to ingest")

    p.add_argument("--query", "-q", required=True, help="Question to ask")
    p.add_argument(
        "--type",
        choices=["factual", "summarization", "analytical"],
        default=None,
        help="Override query type (default: auto-detect)",
    )
    p.add_argument("--model", default="mistralai/mistral-7b-instruct", help="OpenRouter model ID")
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY is not set.\n"
            "Run: export OPENROUTER_API_KEY='your_key_here'",
            file=sys.stderr,
        )
        sys.exit(1)

    print("🚀 Initialising RAG pipeline …")
    pipeline = RAGPipeline(llm_model=args.model)

    if args.file:
        print(f"📄 Ingesting file: {args.file}")
        n = pipeline.ingest_file(args.file)
        print(f"   → {n} chunks indexed")
    else:
        print("📝 Ingesting provided text …")
        n = pipeline.ingest_text(args.text)
        print(f"   → {n} chunks indexed")

    print(f"\n🔍 Query: {args.query}")
    result = pipeline.query(args.query, override_type=args.type)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Pretty print
    q_type = result["query_type"]
    type_symbols = {"factual": "💡", "summarization": "📑", "analytical": "🔬"}
    sym = type_symbols.get(q_type, "❓")

    print(f"\n{sym} Query type : {q_type.upper()}")
    print(f"🤖 Model      : {result['model']}")
    print(f"📦 Chunks used: {result['chunks_used']}")
    print(f"🔄 Cached     : {result['cached']}")
    print("\n" + "─" * 60)
    print("ANSWER:")
    print("─" * 60)
    print(result["answer"])
    print("─" * 60)

    print(f"\n📚 Context sources ({len(result['sources'])} chunks):")
    for i, src in enumerate(result["sources"], 1):
        preview = src[:120].replace("\n", " ")
        print(f"  [{i}] {preview}{'…' if len(src) > 120 else ''}")


if __name__ == "__main__":
    main()
