"""Tests for the HTTP API.

These exercise the full request/response cycle through FastAPI's TestClient,
but with the pipeline dependency overridden to use fakes. That means we test
routing, validation, status codes, and serialisation without downloading a
single model, so the suite stays fast enough to run on every commit.

This is the payoff for the injectable design in Session 1: no monkeypatching,
no mocking internals, just a different object passed in.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from api.deps import get_pipeline
from api.main import app
from src.generator import EchoGenerator
from src.rag import RAGPipeline
from src.vectorstore import VectorStore


@pytest.fixture
def client(config, temp_dir, fake_embedder):
    """TestClient wired to a pipeline backed by fakes and a temp index."""
    store = VectorStore(config=config, persist_dir=temp_dir, embedder=fake_embedder)
    pipeline = RAGPipeline(config=config, store=store, generator=EchoGenerator())

    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as test_client:
        test_client.pipeline = pipeline  # exposed so tests can seed the index
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def seeded_client(client, sample_docs):
    """A client whose index already contains the sample documents."""
    client.pipeline.ingest_directory(sample_docs)
    return client


class TestServiceEndpoints:
    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "PaperRAG"
        assert "version" in body

    def test_health_reports_ok_when_empty(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["ready"] is False
        assert body["chunks_indexed"] == 0

    def test_health_reports_ready_once_indexed(self, seeded_client):
        body = seeded_client.get("/health").json()
        assert body["ready"] is True
        assert body["chunks_indexed"] > 0

    def test_openapi_schema_is_generated(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "/ask" in response.json()["paths"]


class TestIndexEndpoints:
    def test_stats_on_empty_index(self, client):
        body = client.get("/stats").json()
        assert body["chunks_indexed"] == 0
        assert body["document_count"] == 0
        assert body["embedding_model"]

    def test_stats_after_ingest(self, seeded_client):
        body = seeded_client.get("/stats").json()
        assert body["chunks_indexed"] > 0
        assert "detection.md" in body["documents"]

    def test_sources_lists_documents(self, seeded_client):
        body = seeded_client.get("/sources").json()
        assert body["count"] == len(body["documents"])
        assert "metrics.md" in body["documents"]

    def test_delete_index_clears_everything(self, seeded_client):
        response = seeded_client.delete("/index")
        assert response.status_code == 200
        assert response.json()["chunks_removed"] > 0
        assert seeded_client.get("/stats").json()["chunks_indexed"] == 0


class TestAskEndpoint:
    def test_returns_503_when_nothing_indexed(self, client):
        response = client.post("/ask", json={"question": "anything at all?"})
        assert response.status_code == 503
        assert "no documents indexed" in response.json()["detail"].lower()

    def test_returns_answer_with_sources(self, seeded_client):
        response = seeded_client.post("/ask", json={"question": "What is C2PSA?"})
        assert response.status_code == 200
        body = response.json()
        assert body["question"] == "What is C2PSA?"
        assert body["answer"]
        assert len(body["sources"]) > 0
        assert len(body["citations"]) > 0

    def test_source_chunks_have_expected_shape(self, seeded_client):
        body = seeded_client.post("/ask", json={"question": "What is C2PSA?"}).json()
        chunk = body["sources"][0]
        assert set(chunk) == {"source", "chunk_index", "similarity", "excerpt"}
        assert 0.0 <= chunk["similarity"] <= 1.0

    def test_top_k_is_respected(self, seeded_client):
        body = seeded_client.post(
            "/ask", json={"question": "attention modules", "top_k": 1}
        ).json()
        assert len(body["sources"]) == 1

    def test_rejects_question_that_is_too_short(self, client):
        assert client.post("/ask", json={"question": "hi"}).status_code == 422

    def test_rejects_missing_question(self, client):
        assert client.post("/ask", json={}).status_code == 422

    def test_rejects_top_k_out_of_range(self, client):
        response = client.post("/ask", json={"question": "valid question", "top_k": 99})
        assert response.status_code == 422


class TestRetrieveEndpoint:
    def test_returns_empty_on_empty_index(self, client):
        body = client.post("/retrieve", json={"query": "anything"}).json()
        assert body["count"] == 0
        assert body["results"] == []

    def test_returns_ranked_results(self, seeded_client):
        body = seeded_client.post(
            "/retrieve", json={"query": "C2PSA spatial attention", "top_k": 1}
        ).json()
        assert body["count"] == 1
        assert body["results"][0]["source"] == "detection.md"

    def test_count_matches_results_length(self, seeded_client):
        body = seeded_client.post("/retrieve", json={"query": "detection"}).json()
        assert body["count"] == len(body["results"])

    def test_rejects_empty_query(self, client):
        assert client.post("/retrieve", json={"query": ""}).status_code == 422


class TestIngestEndpoint:
    def test_uploads_and_indexes_a_markdown_file(self, client):
        content = b"# Notes\n\nRetrieval augmented generation grounds answers in sources."
        response = client.post(
            "/ingest", files={"file": ("notes.md", io.BytesIO(content), "text/markdown")}
        )
        assert response.status_code == 201
        body = response.json()
        assert body["filename"] == "notes.md"
        assert body["chunks_added"] > 0
        assert body["total_chunks_indexed"] == body["chunks_added"]

    def test_uploaded_document_becomes_searchable(self, client):
        content = b"Quantisation reduces model size by lowering numeric precision."
        client.post(
            "/ingest", files={"file": ("q.md", io.BytesIO(content), "text/markdown")}
        )
        body = client.post("/retrieve", json={"query": "quantisation"}).json()
        assert body["count"] > 0

    def test_rejects_unsupported_extension(self, client):
        response = client.post(
            "/ingest",
            files={"file": ("data.docx", io.BytesIO(b"x"), "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    def test_rejects_empty_file(self, client):
        response = client.post(
            "/ingest", files={"file": ("empty.md", io.BytesIO(b""), "text/markdown")}
        )
        assert response.status_code == 400

    def test_rejects_missing_file(self, client):
        assert client.post("/ingest").status_code == 422
