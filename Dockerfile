# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: builder — install dependencies into a virtual env
# ============================================================
# Keeping the build tools (gcc, etc.) out of the final image is the whole
# point of a multi-stage build: they're needed to compile some Python
# packages but are dead weight at runtime, often adding hundreds of MB.
FROM python:3.12-slim AS builder

# Build tools needed to compile packages with native extensions
# (torch and its dependencies pull some in). Removed automatically since
# this whole stage is discarded, keeping the final image clean.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create an isolated venv inside the image. This gets copied whole into the
# runtime stage in one layer, rather than reinstalling packages there.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements file first. Docker caches layers by content, so
# if requirements.txt hasn't changed, this whole (slow) install step is
# skipped on rebuild rather than re-running every time source code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: runtime — slim image with only what's needed to run
# ============================================================
FROM python:3.12-slim AS runtime

# Metadata, not functional, but good practice for anyone inspecting the image
LABEL maintainer="Abdullah Al Kafi" \
      description="PaperRAG: retrieval-augmented QA with cited sources"

# Run as a non-root user. Skipping this is a common real-world security
# mistake: a container escape then runs as root on the host.
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Bring in the pre-built virtual environment from the builder stage.
# No compiler, no build cache, no apt package lists in this layer.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Application code. Copied after dependencies so that editing source files
# doesn't invalidate the (expensive) dependency-install cache layer above.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser cli.py run_api.py ./
COPY --chown=appuser:appuser data/sample_docs/ ./data/sample_docs/

# Persistent storage for the vector index. Declared as a volume mount point
# so data survives container restarts when paired with docker-compose.
RUN mkdir -p /app/data/chroma_db && chown -R appuser:appuser /app/data

# Cache directory for downloaded model weights (Hugging Face + sentence
# transformers). Without this, every container restart re-downloads ~1GB.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app/.cache

USER appuser

EXPOSE 8000

# Docker's own health check, independent of any orchestrator. Lets
# `docker ps` show container health directly and lets compose/k8s restart
# an unhealthy container automatically.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
