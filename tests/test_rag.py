"""Tests for the end-to-end RAG pipeline."""

from src.ingest import load_directory


class TestRAGPipeline:
    def test_ask_with_empty_index_returns_guidance(self, pipeline):
        answer = pipeline.ask("anything?")
        assert answer.sources == []
        assert "no documents" in answer.answer.lower()

    def test_ingest_directory_indexes_chunks(self, pipeline, sample_docs):
        count = pipeline.ingest_directory(sample_docs)
        assert count > 0
        assert pipeline.store.count() == count

    def test_ingest_file_indexes_single_document(self, pipeline, sample_docs):
        count = pipeline.ingest_file(sample_docs / "detection.md")
        assert count > 0
        assert pipeline.store.sources() == ["detection.md"]

    def test_ask_returns_answer_with_sources(self, pipeline, sample_docs):
        pipeline.ingest_directory(sample_docs)
        answer = pipeline.ask("What is C2PSA?")
        assert answer.answer
        assert len(answer.sources) > 0

    def test_citations_are_deduplicated(self, pipeline, sample_docs):
        pipeline.ingest_directory(sample_docs)
        answer = pipeline.ask("detection and metrics", top_k=4)
        assert len(answer.citations) == len(set(answer.citations))

    def test_retrieval_finds_correct_document(self, pipeline, sample_docs):
        pipeline.ingest_directory(sample_docs)
        chunks = pipeline.retrieve("mAP IoU bounding box localisation", top_k=1)
        assert chunks[0].source == "metrics.md"

    def test_top_k_limits_results(self, pipeline, sample_docs):
        pipeline.ingest_directory(sample_docs)
        assert len(pipeline.retrieve("attention", top_k=1)) == 1

    def test_context_labels_each_chunk_with_source(self, pipeline, sample_docs):
        pipeline.ingest_directory(sample_docs)
        chunks = pipeline.retrieve("attention", top_k=2)
        context = pipeline._build_context(chunks)
        assert "[1]" in context and "from " in context

    def test_to_dict_is_json_serialisable(self, pipeline, sample_docs):
        import json

        pipeline.ingest_directory(sample_docs)
        payload = pipeline.ask("What is C2PSA?").to_dict()
        json.dumps(payload)  # raises if not serialisable
        assert set(payload) == {"question", "answer", "citations", "sources"}

    def test_stats_reports_index_state(self, pipeline, sample_docs):
        pipeline.ingest_directory(sample_docs)
        stats = pipeline.stats()
        assert stats["chunks_indexed"] > 0
        assert "detection.md" in stats["documents"]
