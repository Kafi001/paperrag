"""The RAG pipeline itself.

Ties retrieval and generation together, and — importantly — returns the
sources alongside the answer. An answer without provenance is not much use
in a research context, and citation is what separates RAG from a chatbot
that might be confabulating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, default_config
from .generator import Generator, LocalGenerator
from .ingest import load_directory, load_document
from .vectorstore import RetrievedChunk, VectorStore


@dataclass
class Answer:
    """A generated answer plus the evidence behind it."""

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)

    @property
    def citations(self) -> list[str]:
        """Unique source documents, in retrieval order."""
        seen: list[str] = []
        for chunk in self.sources:
            if chunk.source not in seen:
                seen.append(chunk.source)
        return seen

    def to_dict(self) -> dict:
        """JSON-serialisable form, used by the API layer."""
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "sources": [
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "similarity": c.similarity,
                    "excerpt": c.text[:300] + ("..." if len(c.text) > 300 else ""),
                }
                for c in self.sources
            ],
        }


class RAGPipeline:
    """Ingest documents, retrieve relevant context, generate grounded answers."""

    def __init__(
        self,
        config: Config = default_config,
        store: VectorStore | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.config = config
        self.store = store or VectorStore(config)
        # Generator is injected so tests can pass EchoGenerator instead
        self.generator = generator or LocalGenerator(config)

    # ---------- ingestion ----------

    def ingest_file(self, path: Path) -> int:
        """Index a single document. Returns chunks added."""
        return self.store.add_chunks(load_document(path, self.config))

    def ingest_directory(self, directory: Path) -> int:
        """Index every supported file in a directory. Returns chunks added."""
        return self.store.add_chunks(load_directory(directory, self.config))

    # ---------- querying ----------

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieval only — useful for debugging why an answer went wrong."""
        return self.store.search(question, top_k)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        """Full RAG: retrieve, build context, generate, return with citations."""
        chunks = self.retrieve(question, top_k)

        if not chunks:
            return Answer(
                question=question,
                answer=(
                    "No documents have been indexed yet, so I can't answer that. "
                    "Ingest some documents first."
                ),
                sources=[],
            )

        context = self._build_context(chunks)
        answer_text = self.generator.generate(question, context)
        return Answer(question=question, answer=answer_text, sources=chunks)

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Concatenate retrieved chunks, labelled so the model can attribute them."""
        return "\n\n".join(
            f"[{i + 1}] (from {c.source}) {c.text}" for i, c in enumerate(chunks)
        )

    # ---------- introspection ----------

    def stats(self) -> dict:
        """Index summary, surfaced by the CLI and the API health endpoint."""
        return {
            "chunks_indexed": self.store.count(),
            "documents": self.store.sources(),
            "embedding_model": self.config.embedding_model,
            "generator_model": self.config.generator_model,
        }
