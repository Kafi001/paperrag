"""Central configuration for PaperRAG.

Keeping settings in one place means we can change models or chunk sizes
without hunting through the codebase, and tests can override them easily.
"""

from dataclasses import dataclass
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
VECTORSTORE_DIR = DATA_DIR / "chroma_db"


@dataclass
class Config:
    """Runtime configuration.

    Attributes:
        embedding_model: Sentence-transformers model used to turn text into
            vectors. all-MiniLM-L6-v2 is small (~80MB), fast, and performs
            well on semantic similarity tasks.
        generator_model: Seq2seq model used to write the final answer.
            flan-t5-base is instruction-tuned, which matters for RAG because
            we need it to follow "answer using only this context".
        chunk_size: Target characters per chunk. Too large and retrieval
            becomes imprecise; too small and chunks lose context.
        chunk_overlap: Characters shared between neighbouring chunks, so a
            sentence spanning a boundary isn't lost.
        top_k: Number of chunks retrieved and passed to the generator.
        max_answer_tokens: Cap on generated answer length.
    """

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator_model: str = "google/flan-t5-base"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    max_answer_tokens: int = 256
    collection_name: str = "papers"


# Default instance used across the app
default_config = Config()
