"""Document loading and chunking.

RAG lives or dies on chunking. If chunks are too big, the retriever returns
loosely relevant text and the generator gets distracted. If they're too small,
individual chunks lose the context needed to answer anything.

This module keeps chunking deliberately simple and testable: split on
paragraph boundaries where possible, fall back to character windows, and
always carry an overlap so sentences spanning a boundary survive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, default_config


@dataclass
class Chunk:
    """A retrievable unit of text plus where it came from."""

    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict:
        """Chroma metadata values must be str/int/float/bool, not nested."""
        base = {"source": self.source, "chunk_index": self.chunk_index}
        base.update({k: v for k, v in self.metadata.items()
                     if isinstance(v, (str, int, float, bool))})
        return base


def clean_text(text: str) -> str:
    """Normalise whitespace without destroying paragraph structure.

    PDFs in particular produce ragged line breaks mid-sentence. We collapse
    runs of spaces and single newlines, but preserve blank lines as paragraph
    separators so the chunker has something meaningful to split on.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Preserve paragraph breaks, collapse single newlines into spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    source: str,
    config: Config = default_config,
) -> list[Chunk]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Args:
        text: Raw document text.
        source: Identifier (usually filename) recorded on every chunk so
            answers can cite where they came from.
        config: Supplies chunk_size and chunk_overlap.

    Returns:
        List of Chunk objects in document order.
    """
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # If a single paragraph exceeds chunk_size, window it directly.
        if len(para) > config.chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_window(para, config))
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= config.chunk_size:
            current = candidate
        else:
            chunks.append(current)
            # Carry overlap from the end of the previous chunk
            tail = current[-config.chunk_overlap:] if config.chunk_overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)

    return [
        Chunk(text=c, source=source, chunk_index=i)
        for i, c in enumerate(chunks)
        if c.strip()
    ]


def _window(text: str, config: Config) -> list[str]:
    """Fixed-size sliding window with overlap, for oversized paragraphs."""
    step = max(config.chunk_size - config.chunk_overlap, 1)
    return [
        text[i : i + config.chunk_size]
        for i in range(0, len(text), step)
        if text[i : i + config.chunk_size].strip()
    ]


def load_txt(path: Path) -> str:
    """Read a plain text or markdown file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    """Extract text from a PDF.

    pypdf is imported lazily so the package is only required if the user
    actually ingests PDFs, keeping the base install lighter.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PDF support requires pypdf. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


LOADERS = {".txt": load_txt, ".md": load_txt, ".pdf": load_pdf}


def load_document(path: Path, config: Config = default_config) -> list[Chunk]:
    """Load one file and return its chunks.

    Raises:
        ValueError: If the file extension has no registered loader.
    """
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(LOADERS))}"
        )
    return chunk_text(loader(path), source=path.name, config=config)


def load_directory(
    directory: Path,
    config: Config = default_config,
) -> list[Chunk]:
    """Load every supported file in a directory (non-recursive).

    Unsupported files are skipped silently rather than raising, so a stray
    .DS_Store or notes.docx doesn't abort a bulk ingest.
    """
    chunks: list[Chunk] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in LOADERS:
            chunks.extend(load_document(path, config))
    return chunks
