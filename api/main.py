"""PaperRAG HTTP API.

Exposes the RAG pipeline over HTTP with validated requests, consistent error
handling, and auto-generated interactive documentation at /docs.

Run locally:
    uvicorn api.main:app --reload

Then open http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.ingest import LOADERS
from src.rag import RAGPipeline

from .deps import get_pipeline, init_pipeline
from .schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    ResetResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourceChunk,
    SourcesResponse,
    StatsResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("paperrag.api")

# Reject uploads above this size before writing anything to disk
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown.

    The pipeline is constructed once here rather than per request. Model
    weights themselves load lazily on first use, so startup stays fast while
    still avoiding repeated construction cost.
    """
    logger.info("Starting PaperRAG API v%s", __version__)
    pipeline = init_pipeline()
    logger.info("Pipeline ready. %d chunks currently indexed.", pipeline.store.count())
    yield
    logger.info("Shutting down PaperRAG API")


app = FastAPI(
    title="PaperRAG API",
    description=(
        "Retrieval-augmented question answering over your documents. "
        "Every answer is returned with the source chunks it was based on, "
        "so claims can be verified rather than trusted blindly."
    ),
    version=__version__,
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# Permissive CORS for local development. In production this would be
# restricted to known frontend origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- error handling ----------


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Turn domain errors (e.g. unsupported file type) into clean 400s.

    Without this, a ValueError raised deep in the ingest layer would surface
    as a 500 with a stack trace, which is both unhelpful and a mild
    information leak.
    """
    logger.warning("ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "ValueError"},
    )


# ---------- service info ----------


@app.get("/", tags=["service"], summary="Service information")
async def root() -> dict:
    """Basic service metadata and where to find the docs."""
    return {
        "service": "PaperRAG",
        "version": __version__,
        "description": "Retrieval-augmented question answering with citations",
        "docs": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["service"],
    summary="Health and readiness check",
)
async def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    """Liveness and readiness.

    Deliberately separates 'running' from 'ready': the service can be healthy
    while having nothing indexed, in which case it cannot answer questions yet.
    """
    count = pipeline.store.count()
    return HealthResponse(
        status="ok",
        ready=count > 0,
        chunks_indexed=count,
        version=__version__,
    )


# ---------- index inspection ----------


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["index"],
    summary="Index statistics",
)
async def stats(pipeline: RAGPipeline = Depends(get_pipeline)) -> StatsResponse:
    """What is indexed and which models are configured."""
    data = pipeline.stats()
    return StatsResponse(
        chunks_indexed=data["chunks_indexed"],
        document_count=len(data["documents"]),
        documents=data["documents"],
        embedding_model=data["embedding_model"],
        generator_model=data["generator_model"],
    )


@app.get(
    "/sources",
    response_model=SourcesResponse,
    tags=["index"],
    summary="List indexed documents",
)
async def sources(pipeline: RAGPipeline = Depends(get_pipeline)) -> SourcesResponse:
    """The distinct documents currently searchable."""
    docs = pipeline.store.sources()
    return SourcesResponse(documents=docs, count=len(docs))


# ---------- querying ----------


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["query"],
    summary="Ask a question",
    responses={
        503: {"model": ErrorResponse, "description": "No documents indexed yet"},
    },
)
async def ask(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> AskResponse:
    """Answer a question using the indexed documents.

    Returns the generated answer plus the chunks it was based on, so the
    answer can be checked against its sources.

    Raises:
        HTTPException 503: If nothing is indexed. This is a service-state
            problem rather than a bad request, hence 503 and not 400.
    """
    if pipeline.store.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents indexed. Upload a document via POST /ingest first.",
        )

    logger.info("Question: %s", request.question[:100])
    answer = pipeline.ask(request.question, top_k=request.top_k)
    payload = answer.to_dict()

    return AskResponse(
        question=payload["question"],
        answer=payload["answer"],
        citations=payload["citations"],
        sources=[SourceChunk(**s) for s in payload["sources"]],
    )


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["query"],
    summary="Retrieve relevant chunks without generating an answer",
)
async def retrieve(
    request: RetrieveRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> RetrieveResponse:
    """Semantic search over the indexed chunks.

    Much faster than /ask because it skips generation entirely. Also the first
    thing to check when an answer looks wrong: if the right chunk was never
    retrieved, the generator was never going to produce a good answer.
    """
    chunks = pipeline.retrieve(request.query, top_k=request.top_k)
    results = [
        SourceChunk(
            source=c.source,
            chunk_index=c.chunk_index,
            similarity=c.similarity,
            excerpt=c.text[:300] + ("..." if len(c.text) > 300 else ""),
        )
        for c in chunks
    ]
    return RetrieveResponse(query=request.query, results=results, count=len(results))


# ---------- ingestion ----------


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["index"],
    summary="Upload and index a document",
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported file type"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def ingest(
    file: UploadFile = File(..., description="A .pdf, .txt, or .md file"),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """Index an uploaded document so it becomes searchable.

    Validation happens in order of cost: filename and extension are checked
    before reading the body, and size is checked before anything touches disk.

    Raises:
        HTTPException 400: Missing filename or unsupported extension.
        HTTPException 413: File exceeds the size limit.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in LOADERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(sorted(LOADERS))}"
            ),
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
        )
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Write to a temp file so the existing loaders can work with a real path,
    # then always clean up, even if indexing raises.
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        tmp_path.write_bytes(contents)
        added = pipeline.ingest_file(tmp_path)
        logger.info("Indexed %d chunks from %s", added, file.filename)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return IngestResponse(
        filename=file.filename,
        chunks_added=added,
        total_chunks_indexed=pipeline.store.count(),
        message=f"Indexed {added} chunks from {file.filename}.",
    )


@app.delete(
    "/index",
    response_model=ResetResponse,
    tags=["index"],
    summary="Clear the index",
)
async def reset_index(pipeline: RAGPipeline = Depends(get_pipeline)) -> ResetResponse:
    """Remove every indexed chunk. Irreversible."""
    removed = pipeline.store.count()
    pipeline.store.reset()
    logger.info("Index cleared (%d chunks removed)", removed)
    return ResetResponse(
        chunks_removed=removed,
        message=f"Cleared {removed} chunks from the index.",
    )
