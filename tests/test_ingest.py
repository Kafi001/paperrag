"""Tests for document loading and chunking."""

import pytest

from src.config import Config
from src.ingest import chunk_text, clean_text, load_directory, load_document


class TestCleanText:
    def test_collapses_single_newlines_into_spaces(self):
        assert clean_text("hello\nworld") == "hello world"

    def test_preserves_paragraph_breaks(self):
        assert clean_text("para one\n\npara two") == "para one\n\npara two"

    def test_collapses_repeated_whitespace(self):
        assert clean_text("too    many     spaces") == "too many spaces"

    def test_handles_windows_line_endings(self):
        assert clean_text("a\r\n\r\nb") == "a\n\nb"

    def test_empty_input(self):
        assert clean_text("   \n\n  ") == ""


class TestChunkText:
    def test_short_text_yields_single_chunk(self):
        chunks = chunk_text("A short sentence.", source="s.md")
        assert len(chunks) == 1
        assert chunks[0].text == "A short sentence."

    def test_empty_text_yields_nothing(self):
        assert chunk_text("", source="s.md") == []

    def test_long_text_splits_into_multiple_chunks(self):
        config = Config(chunk_size=100, chunk_overlap=20)
        text = "\n\n".join(f"Paragraph number {i} with filler words." for i in range(20))
        chunks = chunk_text(text, source="s.md", config=config)
        assert len(chunks) > 1

    def test_chunks_respect_size_limit_approximately(self):
        config = Config(chunk_size=200, chunk_overlap=30)
        text = "\n\n".join(f"Sentence {i} here." for i in range(50))
        chunks = chunk_text(text, source="s.md", config=config)
        # Allow overlap headroom, but nothing wildly oversized
        assert all(len(c.text) <= config.chunk_size + config.chunk_overlap + 50
                   for c in chunks)

    def test_oversized_paragraph_is_windowed(self):
        config = Config(chunk_size=50, chunk_overlap=10)
        chunks = chunk_text("word " * 100, source="s.md", config=config)
        assert len(chunks) > 1

    def test_source_and_index_recorded(self):
        config = Config(chunk_size=60, chunk_overlap=10)
        text = "\n\n".join(f"Para {i} content here." for i in range(6))
        chunks = chunk_text(text, source="paper.pdf", config=config)
        assert all(c.source == "paper.pdf" for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chroma_metadata_is_flat_and_serialisable(self):
        chunk = chunk_text("Some text.", source="s.md")[0]
        meta = chunk.to_chroma_metadata()
        assert meta["source"] == "s.md"
        assert meta["chunk_index"] == 0
        assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())


class TestLoadDocument:
    def test_loads_markdown(self, sample_docs):
        chunks = load_document(sample_docs / "detection.md")
        assert len(chunks) >= 1
        assert "C2PSA" in " ".join(c.text for c in chunks)

    def test_rejects_unsupported_extension(self, temp_dir):
        bad = temp_dir / "notes.docx"
        bad.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(bad)

    def test_load_directory_reads_all_supported_files(self, sample_docs):
        chunks = load_directory(sample_docs)
        assert {c.source for c in chunks} == {"detection.md", "metrics.md"}

    def test_load_directory_skips_unsupported_files(self, sample_docs):
        (sample_docs / "ignore.docx").write_text("x", encoding="utf-8")
        chunks = load_directory(sample_docs)
        assert "ignore.docx" not in {c.source for c in chunks}
