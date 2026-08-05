# PaperRAG API

## Running the server

```bash
PYTHONPATH=. uvicorn api.main:app --reload
```

Or:

```bash
PYTHONPATH=. python run_api.py
```

Then open **http://localhost:8000/docs** for interactive documentation.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service information |
| GET | `/health` | Liveness and readiness check |
| GET | `/stats` | Index statistics |
| GET | `/sources` | List indexed documents |
| POST | `/ask` | Ask a question, get an answer with citations |
| POST | `/retrieve` | Semantic search only (fast, no LLM) |
| POST | `/ingest` | Upload and index a document |
| DELETE | `/index` | Clear the index |

## Examples

### Health check
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "ready": true, "chunks_indexed": 10, "version": "0.1.0"}
```

### Upload a document
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample_docs/object_detection.md"
```
```json
{
  "filename": "object_detection.md",
  "chunks_added": 4,
  "total_chunks_indexed": 4,
  "message": "Indexed 4 chunks from object_detection.md."
}
```

### Ask a question
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
      "excerpt": "YOLOv11, released in 2024, introduces C3k2 blocks..."
    }
  ]
}
```

### Retrieval only (much faster)
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "spatial attention", "top_k": 2}'
```

### Clear the index
```bash
curl -X DELETE http://localhost:8000/index
```

## Status codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Document indexed |
| 400 | Bad request (unsupported file type, empty file) |
| 413 | File exceeds 25MB limit |
| 422 | Validation failed (question too short, top_k out of range) |
| 503 | Service running but no documents indexed yet |

## Design notes

**Why the pipeline is a singleton.** Models take seconds to load and hundreds of
megabytes of memory. Constructing one per request would make the API unusable.
The `lifespan` context manager builds it once at startup; `Depends(get_pipeline)`
hands the same instance to every request.

**Why `/retrieve` exists separately from `/ask`.** Generation dominates latency.
Retrieval alone is fast enough for interactive use, and it's the first thing to
check when an answer looks wrong: if the right chunk was never retrieved, no
amount of prompt tuning will fix the answer.

**Why 503 rather than 400 for an empty index.** An empty index is a service-state
problem, not a malformed request. The client did nothing wrong, so 503 with a
`Retry-After` semantic is more accurate than blaming the caller.

**Why tests don't download models.** `app.dependency_overrides` swaps in a
pipeline backed by a deterministic fake embedder and an echo generator. The full
suite runs in under two seconds, which is what makes it viable in CI.
