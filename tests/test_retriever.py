"""Tests for retriever."""

from unittest.mock import MagicMock

from gem_rag.database import Database
from gem_rag.retriever import Retriever, _merge_spans, _Span


def _seed_db(db: Database) -> None:
    """Seed a DB with a document and 5 chunks with known embeddings."""
    db.insert_document("d1", "/doc.md", "hash", 5, "model")
    for i in range(5):
        # Embeddings: chunk 2 is most similar to [1,0,0]
        if i == 2:
            emb = [1.0, 0.0, 0.0]
        elif i in (1, 3):
            emb = [0.7, 0.3, 0.0]
        else:
            emb = [0.1, 0.1, 0.8]
        db.insert_chunks([(f"c{i}", "d1", i, f"Section {i}", f"Content of chunk {i}", emb)])


def _mock_embedder(query_vector: list[float] | None = None) -> MagicMock:
    embedder = MagicMock()
    embedder.model_name = "model"
    embedder.embed_query.return_value = query_vector or [1.0, 0.0, 0.0]
    return embedder


class TestRetriever:
    def test_basic_retrieval(self) -> None:
        db = Database(":memory:")
        _seed_db(db)
        retriever = Retriever(db, _mock_embedder(), top_k=3, context_window=0)
        passages = retriever.retrieve("query")

        assert len(passages) == 3
        assert passages[0].content == "Content of chunk 2"
        assert passages[0].score > passages[1].score
        db.close()

    def test_context_expansion(self) -> None:
        db = Database(":memory:")
        _seed_db(db)
        retriever = Retriever(db, _mock_embedder(), top_k=1, context_window=1)
        passages = retriever.retrieve("query")

        assert len(passages) == 1
        # Should contain chunk 1, 2, 3 merged
        assert "chunk 1" in passages[0].content
        assert "chunk 2" in passages[0].content
        assert "chunk 3" in passages[0].content
        db.close()

    def test_empty_db(self) -> None:
        db = Database(":memory:")
        retriever = Retriever(db, _mock_embedder(), top_k=5, context_window=1)
        passages = retriever.retrieve("query")
        assert passages == []
        db.close()


class TestMergeSpans:
    def test_no_overlap(self) -> None:
        spans = [
            _Span(0, 2, 0.9, "A", "/a.md"),
            _Span(5, 7, 0.8, "B", "/a.md"),
        ]
        merged = _merge_spans(spans)
        assert len(merged) == 2

    def test_overlapping(self) -> None:
        spans = [
            _Span(0, 3, 0.8, "A", "/a.md"),
            _Span(2, 5, 0.9, "B", "/a.md"),
        ]
        merged = _merge_spans(spans)
        assert len(merged) == 1
        assert merged[0].lo == 0
        assert merged[0].hi == 5
        assert merged[0].score == 0.9  # Best score kept

    def test_adjacent(self) -> None:
        spans = [
            _Span(0, 2, 0.9, "A", "/a.md"),
            _Span(3, 5, 0.8, "B", "/a.md"),
        ]
        merged = _merge_spans(spans)
        assert len(merged) == 1
        assert merged[0].hi == 5

    def test_empty(self) -> None:
        assert _merge_spans([]) == []
