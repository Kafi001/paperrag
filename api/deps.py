"""Dependency injection for the API.

The pipeline is expensive to construct: it holds an embedding model and a
language model, both of which take seconds to load and hundreds of megabytes
of memory. Creating one per request would be unusable.

So we build it once and hand the same instance to every request via FastAPI's
Depends system. That also gives us a clean seam for testing: tests override
get_pipeline() to inject a pipeline backed by fakes, with no patching.
"""

from __future__ import annotations

from src.config import Config, default_config
from src.rag import RAGPipeline

# Module-level singleton, populated during app startup (see main.lifespan)
_pipeline: RAGPipeline | None = None


def init_pipeline(config: Config = default_config) -> RAGPipeline:
    """Create the shared pipeline. Called once at application startup."""
    global _pipeline
    _pipeline = RAGPipeline(config=config)
    return _pipeline


def get_pipeline() -> RAGPipeline:
    """FastAPI dependency returning the shared pipeline.

    Raises:
        RuntimeError: If called before startup completed, which would indicate
            a wiring bug rather than a user error.
    """
    if _pipeline is None:
        raise RuntimeError(
            "Pipeline not initialised. This should be done during app startup."
        )
    return _pipeline


def reset_pipeline() -> None:
    """Clear the singleton. Used by tests to avoid state leaking between them."""
    global _pipeline
    _pipeline = None
