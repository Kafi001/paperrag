"""Vector storage and semantic retrieval.

Wraps ChromaDB so the rest of the app never touches Chroma's API directly.
That isolation matters: if we later swap to FAISS, pgvector, or a hosted
store, only this file changes.

Embeddings are generated with sentence-transformers rather than Chroma's
default, so the model is explicit and configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings

from .config import Config, default_config, VECTORSTORE_DIR
from .ingest import Chunk


class Embedder(Protocol):
    """Anything that can turn a list of strings into a list of vectors."""

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Production embedder backed by sentence-transformers.

    Imported lazily inside __init__ so that merely importing this module
    doesn't pull in torch, keeping test collection and API startup fast.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        ).tolist()


@dataclass
class RetrievedChunk:
    """A search result: the chunk text, where it came from, and how close it was."""

    text: str
    source: str
    chunk_index: int
    distance: float

    @property
    def similarity(self) -> float:
        """Convert cosine distance to a 0-1 similarity for readability."""
        return round(max(0.0, 1.0 - self.distance), 4)


class VectorStore:
    """Persistent semantic index over document chunks."""

    def __init__(
        self,
        config: Config = default_config,
        persist_dir: Path | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """
        Args:
            config: Model names and retrieval settings.
            persist_dir: Where Chroma writes its index. Defaults to data/chroma_db.
            embedder: Optional embedding backend. Injected in tests so the suite
                runs without downloading a model; defaults to sentence-transformers.
        """
        self.config = config
        self.persist_dir = persist_dir or VECTORSTORE_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedder = embedder  # loaded lazily if None
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            # Cosine suits normalised sentence embeddings better than L2
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_embedder(self) -> None:
        """Load the sentence-transformers model on first use, not at import."""
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.config.embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Turn text into vectors. Exposed so tests can inspect embeddings."""
        self._ensure_embedder()
        return self._embedder.encode(texts)

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Embed and store chunks. Returns the number added.

        IDs are deterministic (source + index), so re-ingesting the same
        document updates rather than duplicates it.
        """
        if not chunks:
            return 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[f"{c.source}::{c.chunk_index}" for c in batch],
                documents=[c.text for c in batch],
                embeddings=self.embed([c.text for c in batch]),
                metadatas=[c.to_chroma_metadata() for c in batch],
            )
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the chunks most semantically similar to the query."""
        k = top_k or self.config.top_k
        if self.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=self.embed([query]),
            n_results=min(k, self.count()),
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", -1),
                distance=dist,
            )
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def count(self) -> int:
        """Number of chunks currently indexed."""
        return self._collection.count()

    def sources(self) -> list[str]:
        """Distinct document names in the index."""
        if self.count() == 0:
            return []
        records = self._collection.get(include=["metadatas"])
        return sorted({m.get("source", "unknown") for m in records["metadatas"]})

    def reset(self) -> None:
        """Delete everything in the collection. Used by tests and the CLI."""
        self._client.delete_collection(self.config.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
