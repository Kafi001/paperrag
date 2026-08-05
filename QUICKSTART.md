# Quick Start — Session 1

## Setup

```bash
cd paperrag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First install pulls torch, so expect a few minutes.

## Run the tests (fast, no model download)

```bash
pytest tests/ -v
```

All 37 should pass in under a second. They use a fake embedder, which is why
they're fast — no network, no model files.

## Index the sample documents

```bash
python cli.py ingest data/sample_docs
```

First run downloads the MiniLM embedding model (~80MB).

## Check what's indexed

```bash
python cli.py stats
```

## Retrieval only (no LLM — instant)

```bash
python cli.py retrieve "What is C2PSA in YOLOv11?"
```

Run this first. It shows which chunks were retrieved and their similarity
scores, without waiting for the generator model.

## Ask a full question

```bash
python cli.py ask "What is C2PSA in YOLOv11?"
```

First run downloads flan-t5-base (~1GB) and will take a few minutes. After
that it's cached.

## Add your own documents

```bash
python cli.py ingest /path/to/your/dissertation.pdf
python cli.py ask "What were the main findings on class imbalance?"
```

Your dissertation PDF is a good test — you already know the correct answers,
so you can judge retrieval quality properly.

## Useful commands

```bash
python cli.py reset          # clear the index
python cli.py stats          # what's indexed
```

---

# Session 2 — Running the API

## Start the server

```bash
PYTHONPATH=. uvicorn api.main:app --reload
```

## Open the interactive docs

Go to **http://localhost:8000/docs** in your browser.

FastAPI generates this automatically from the Pydantic schemas. You can call
every endpoint directly from the page, which is the easiest way to explore it.

## Try it from the terminal

```bash
# Health check
curl http://localhost:8000/health

# Upload a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/sample_docs/object_detection.md"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is C2PSA in YOLOv11?"}'
```

## Run the tests

```bash
PYTHONPATH=. pytest tests/ -v
```

61 tests, all passing, in under two seconds.

See **API.md** for full endpoint documentation.
