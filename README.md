# PaperRAG

**Retrieval-Augmented Generation (RAG) over your documents — with cited answers.**

Ask questions about any PDF, Markdown, or text file. PaperRAG retrieves the most relevant passages and generates a grounded answer, telling you exactly which document and chunk it came from.

Built to demonstrate: **Hugging Face embeddings · ChromaDB vector store · RAG architecture · FastAPI REST API · Pydantic validation · pytest (61 tests) · Docker-ready design**

---

## How it works

```
Your question
    │
    ▼
Embed query ──► Search ChromaDB ──► Top-k chunks
                                         │
                                         ▼
                                  Build context
                                         │
                                         ▼
                               TinyLlama generates answer
                                         │
                                         ▼
                           Answer + cited sources returned
```

1. **Ingest** — documents are chunked (paragraph-aware, overlapping), embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB
2. **Retrieve** — the query is embedded and cosine similarity finds the closest chunks
3. **Generate** — context is assembled and passed to TinyLlama with instructions to answer only from the evidence

---

## Project structure

```
paperrag/
├── src/
│   ├── config.py        Central settings (models, chunk size, top-k)
│   ├── ingest.py        Document loading + paragraph-aware chunking
│   ├── vectorstore.py   ChromaDB wrapper with injectable embedder
│   ├── generator.py     LLM wrapper with pluggable backend
│   └── rag.py           Pipeline orchestration + citations
├── api/
│   ├── main.py          FastAPI app (8 endpoints, error handlers, CORS)
│   ├── schemas.py       Pydantic request/response models
│   └── deps.py          Dependency injection + pipeline singleton
├── tests/               61 tests (ingest, vectorstore, RAG, API)
├── data/sample_docs/    Starter corpus on ML and RAG topics
├── cli.py               Command-line interface
└── run_api.py           API entry point
```

---

## Quick start

```bash
git clone https://github.com/Kafi001/paperrag.git
cd paperrag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Run the tests (no model download needed)
```bash
PYTHONPATH=. pytest tests/ -v
# 61 passed in ~2 seconds
```

### Index sample documents
```bash
PYTHONPATH=. python cli.py ingest data/sample_docs
```

### Ask a question (CLI)
```bash
PYTHONPATH=. python cli.py retrieve "What is C2PSA in YOLOv11?"   # instant
PYTHONPATH=. python cli.py ask "What is C2PSA in YOLOv11?"        # with LLM
```

### Start the API server
```bash
PYTHONPATH=. uvicorn api.main:app --reload
# Open http://localhost:8000/docs
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service information |
| GET | `/health` | Liveness and readiness |
| GET | `/stats` | Index statistics |
| GET | `/sources` | List indexed documents |
| POST | `/ask` | Full RAG query with citations |
| POST | `/retrieve` | Semantic search only (fast) |
| POST | `/ingest` | Upload and index a document |
| DELETE | `/index` | Clear the index |

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is C2PSA in YOLOv11?", "top_k": 3}'
```

```json
{
  "question": "What is C2PSA in YOLOv11?",
  "answer": "C2PSA is a parallel spatial attention module...",
  "citations": ["object_detection.md"],
  "sources": [
    {
      "source": "object_detection.md",
      "chunk_index": 2,
      "similarity": 0.8142,
      "excerpt": "YOLOv11 introduces C3k2 blocks and C2PSA..."
    }
  ]
}
```

---

## Design decisions

**Injectable embedder and generator.** Both are Protocols with real and fake implementations. Tests inject the fake — no model download, suite runs in 2 seconds. Swapping to OpenAI or Anthropic means writing one new class.

**Pipeline as a singleton.** Models take seconds to load. The FastAPI `lifespan` context manager builds it once at startup; `Depends(get_pipeline)` shares it across all requests.

**Paragraph-aware chunking with overlap.** Splits on paragraph boundaries rather than fixed character counts, with an overlap so sentences at boundaries survive in both chunks.

**503 for empty index, not 400.** The client did nothing wrong — the service just is not ready yet.

**`/retrieve` as a separate endpoint.** Generation dominates latency. Retrieval alone is fast for interactive use, and is the first thing to check when an answer is wrong.

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (cosine similarity, HNSW index) |
| Generation | TinyLlama-1.1B-Chat (local, CPU) |
| API framework | FastAPI + Pydantic v2 |
| ASGI server | Uvicorn |
| Testing | pytest + FastAPI TestClient (61 tests) |
| PDF parsing | pypdf |

---

## Author

**Abdullah Al Kafi** — MSc Artificial Intelligence, University of Salford (2025)

[LinkedIn](https://www.linkedin.com/in/abdullah-al-kafi-173b4325b) · [GitHub](https://github.com/Kafi001)
