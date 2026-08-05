"""Pydantic schemas for request validation and response serialisation.

Defining these explicitly does three jobs at once:
  1. Validates incoming requests before they reach business logic
  2. Documents the API automatically (FastAPI generates OpenAPI from these)
  3. Guarantees response shape, so clients can rely on the contract

Field constraints (min_length, ge, le) mean invalid input is rejected with a
clear 422 rather than failing deeper in the stack with an unhelpful error.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- shared ----------


class SourceChunk(BaseModel):
    """One retrieved chunk, with provenance and an excerpt for display."""

    source: str = Field(..., description="Document the chunk came from")
    chunk_index: int = Field(..., description="Position within that document")
    similarity: float = Field(..., ge=0.0, le=1.0, description="0-1 relevance score")
    excerpt: str = Field(..., description="Truncated chunk text")

    model_config = {
        "json_schema_extra": {
            "example": {
                "source": "object_detection.md",
                "chunk_index": 2,
                "similarity": 0.8142,
                "excerpt": "YOLOv11 introduces C3k2 blocks and C2PSA...",
            }
        }
    }


# ---------- /ask ----------


class AskRequest(BaseModel):
    """A question to answer using the indexed documents."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to answer",
    )
    top_k: int | None = Field(
        None,
        ge=1,
        le=20,
        description="How many chunks to retrieve. Defaults to the server setting.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"question": "What is C2PSA in YOLOv11?", "top_k": 4}
        }
    }


class AskResponse(BaseModel):
    """A generated answer with the evidence it was based on."""

    question: str
    answer: str
    citations: list[str] = Field(
        default_factory=list, description="Distinct source documents used"
    )
    sources: list[SourceChunk] = Field(
        default_factory=list, description="The retrieved chunks, most relevant first"
    )


# ---------- /retrieve ----------


class RetrieveRequest(BaseModel):
    """A query for retrieval only, skipping generation."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(None, ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "example": {"query": "spatial attention modules", "top_k": 3}
        }
    }


class RetrieveResponse(BaseModel):
    """Retrieved chunks without a generated answer.

    Useful both as a fast endpoint and as a debugging tool: if an answer is
    wrong, checking retrieval first tells you whether the problem is the
    retriever or the generator.
    """

    query: str
    results: list[SourceChunk] = Field(default_factory=list)
    count: int


# ---------- /ingest ----------


class IngestResponse(BaseModel):
    """Result of indexing a document."""

    filename: str
    chunks_added: int
    total_chunks_indexed: int
    message: str


# ---------- /stats, /sources, /health ----------


class StatsResponse(BaseModel):
    """Current state of the index and the models in use."""

    chunks_indexed: int
    document_count: int
    documents: list[str]
    embedding_model: str
    generator_model: str


class SourcesResponse(BaseModel):
    """The distinct documents currently indexed."""

    documents: list[str]
    count: int


class HealthResponse(BaseModel):
    """Liveness and readiness signal.

    'ready' is false when nothing is indexed: the service is up, but it can't
    answer questions yet. Container orchestrators care about this distinction.
    """

    status: str = Field(..., description="'ok' when the service is running")
    ready: bool = Field(..., description="True when documents are indexed")
    chunks_indexed: int
    version: str


class ResetResponse(BaseModel):
    """Result of clearing the index."""

    chunks_removed: int
    message: str


# ---------- errors ----------


class ErrorResponse(BaseModel):
    """Consistent error shape across every endpoint."""

    detail: str
    error_type: str | None = None
