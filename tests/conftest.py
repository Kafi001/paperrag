"""Shared test fixtures.

The key idea: a FakeEmbedder that produces deterministic vectors without
downloading a model. This keeps the suite fast (no network, no 80MB download)
and means CI can run it in seconds.
"""

import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from src.config import Config
from src.generator import EchoGenerator
from src.rag import RAGPipeline
from src.vectorstore import VectorStore


class FakeEmbedder:
    """Deterministic bag-of-words embedder.

    Not semantically clever, but it is consistent and captures word overlap,
    which is enough to assert that retrieval returns the right document.
    """

    DIM = 64

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.DIM
            for word in text.lower().split():
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
                vec[idx] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            vectors.append([x / norm for x in vec])
        return vectors


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Deterministic embedder, so API tests need no model download either."""
    return FakeEmbedder()


@pytest.fixture
def config() -> Config:
    return Config(chunk_size=300, chunk_overlap=50, top_k=3)


@pytest.fixture
def temp_dir():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def store(config, temp_dir) -> VectorStore:
    return VectorStore(config=config, persist_dir=temp_dir, embedder=FakeEmbedder())


@pytest.fixture
def pipeline(config, store) -> RAGPipeline:
    return RAGPipeline(config=config, store=store, generator=EchoGenerator())


@pytest.fixture
def sample_docs(temp_dir) -> Path:
    docs = temp_dir / "docs"
    docs.mkdir()
    (docs / "detection.md").write_text(
        "YOLOv11 introduces C2PSA parallel spatial attention blocks. "
        "Spatial attention reweights feature map regions by relevance, "
        "which helps detect small or occluded objects in cluttered scenes.\n\n"
        "YOLOv8 uses an anchor free detection head with a decoupled design.",
        encoding="utf-8",
    )
    (docs / "metrics.md").write_text(
        "Mean Average Precision averages precision across recall levels. "
        "The mAP at IoU 0.50 to 0.95 metric rewards tighter bounding box "
        "localisation than mAP at 0.50 alone.\n\n"
        "Per class breakdowns reveal failures that aggregate scores hide.",
        encoding="utf-8",
    )
    return docs
