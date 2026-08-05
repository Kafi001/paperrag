"""Command-line interface for PaperRAG.

Usage:
    python cli.py ingest data/sample_docs
    python cli.py ask "What is C2PSA in YOLOv11?"
    python cli.py stats
    python cli.py reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import SAMPLE_DOCS_DIR
from src.rag import RAGPipeline


def cmd_ingest(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        return 1

    pipeline = RAGPipeline()
    print(f"Loading and embedding from {path} ...")
    count = (
        pipeline.ingest_directory(path)
        if path.is_dir()
        else pipeline.ingest_file(path)
    )
    print(f"Indexed {count} chunks. Total in store: {pipeline.store.count()}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline()
    if pipeline.store.count() == 0:
        print("No documents indexed. Run: python cli.py ingest data/sample_docs")
        return 1

    print("Retrieving and generating (first run downloads the model) ...\n")
    answer = pipeline.ask(args.question, top_k=args.top_k)

    print(f"Q: {answer.question}\n")
    print(f"A: {answer.answer}\n")
    print("Sources:")
    for chunk in answer.sources:
        print(f"  - {chunk.source} (chunk {chunk.chunk_index}, "
              f"similarity {chunk.similarity})")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    """Retrieval only. Useful for debugging why an answer was wrong."""
    pipeline = RAGPipeline()
    hits = pipeline.retrieve(args.question, top_k=args.top_k)
    if not hits:
        print("No results. Is anything indexed?")
        return 1
    for i, chunk in enumerate(hits, 1):
        print(f"\n[{i}] {chunk.source} #{chunk.chunk_index} "
              f"(similarity {chunk.similarity})")
        print(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    stats = RAGPipeline().stats()
    print(f"Chunks indexed : {stats['chunks_indexed']}")
    print(f"Documents      : {len(stats['documents'])}")
    for doc in stats["documents"]:
        print(f"  - {doc}")
    print(f"Embedding model: {stats['embedding_model']}")
    print(f"Generator model: {stats['generator_model']}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    pipeline = RAGPipeline()
    n = pipeline.store.count()
    pipeline.store.reset()
    print(f"Cleared {n} chunks from the index.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperrag",
        description="Ask questions over your documents, with cited sources.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a file or directory")
    p_ingest.add_argument("path", nargs="?", default=str(SAMPLE_DOCS_DIR))
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.set_defaults(func=cmd_ask)

    p_ret = sub.add_parser("retrieve", help="Show retrieved chunks only")
    p_ret.add_argument("question")
    p_ret.add_argument("--top-k", type=int, default=None)
    p_ret.set_defaults(func=cmd_retrieve)

    sub.add_parser("stats", help="Show index statistics").set_defaults(func=cmd_stats)
    sub.add_parser("reset", help="Clear the index").set_defaults(func=cmd_reset)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
