"""Tests for embedding, storage, and retrieval."""

from src.ingest import chunk_text


class TestVectorStore:
    def test_starts_empty(self, store):
        assert store.count() == 0
        assert store.sources() == []

    def test_add_chunks_returns_count(self, store):
        chunks = chunk_text("Some indexed content here.", source="a.md")
        assert store.add_chunks(chunks) == len(chunks)
        assert store.count() == len(chunks)

    def test_add_empty_list_is_noop(self, store):
        assert store.add_chunks([]) == 0
        assert store.count() == 0

    def test_embeddings_have_consistent_dimension(self, store):
        vectors = store.embed(["first text", "second text"])
        assert len(vectors) == 2
        assert len(vectors[0]) == len(vectors[1])

    def test_search_on_empty_store_returns_empty(self, store):
        assert store.search("anything") == []

    def test_search_returns_results(self, store):
        store.add_chunks(chunk_text("Spatial attention modules.", source="a.md"))
        results = store.search("spatial attention", top_k=1)
        assert len(results) == 1
        assert results[0].source == "a.md"

    def test_search_ranks_relevant_document_first(self, store, sample_docs):
        from src.ingest import load_directory

        store.add_chunks(load_directory(sample_docs))
        results = store.search("C2PSA spatial attention occluded objects", top_k=1)
        assert results[0].source == "detection.md"

    def test_reingesting_same_document_does_not_duplicate(self, store):
        chunks = chunk_text("Stable content.", source="a.md")
        store.add_chunks(chunks)
        store.add_chunks(chunks)
        assert store.count() == len(chunks)

    def test_sources_lists_distinct_documents(self, store):
        store.add_chunks(chunk_text("One.", source="a.md"))
        store.add_chunks(chunk_text("Two.", source="b.md"))
        assert store.sources() == ["a.md", "b.md"]

    def test_reset_clears_index(self, store):
        store.add_chunks(chunk_text("Temporary.", source="a.md"))
        store.reset()
        assert store.count() == 0

    def test_similarity_is_bounded(self, store):
        store.add_chunks(chunk_text("Content for similarity.", source="a.md"))
        result = store.search("similarity", top_k=1)[0]
        assert 0.0 <= result.similarity <= 1.0
