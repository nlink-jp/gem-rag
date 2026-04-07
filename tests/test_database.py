"""Tests for DuckDB database layer."""

import pytest

from gem_rag.database import Database, make_chunk_id, make_document_id


@pytest.fixture
def db() -> Database:
    d = Database(":memory:")
    yield d
    d.close()


class TestDocumentCRUD:
    def test_insert_and_find(self, db: Database) -> None:
        doc_id = make_document_id("/test.md", "hash123")
        db.insert_document(doc_id, "/test.md", "hash123", 3, "text-embedding-005")
        found = db.find_document_by_path("/test.md")
        assert found is not None
        assert found.id == doc_id
        assert found.file_hash == "hash123"
        assert found.total_chunks == 3

    def test_find_nonexistent(self, db: Database) -> None:
        assert db.find_document_by_path("/nonexistent.md") is None

    def test_list_documents(self, db: Database) -> None:
        db.insert_document("d1", "/a.md", "h1", 1, "model")
        db.insert_document("d2", "/b.md", "h2", 2, "model")
        docs = db.list_documents()
        assert len(docs) == 2

    def test_delete_document(self, db: Database) -> None:
        db.insert_document("d1", "/a.md", "h1", 1, "model")
        db.insert_chunks([("c1", "d1", 0, "", "content", [0.1, 0.2])])
        assert db.delete_document("d1") is True
        assert db.find_document_by_path("/a.md") is None
        assert db.document_chunks("d1") == []

    def test_delete_nonexistent(self, db: Database) -> None:
        assert db.delete_document("nonexistent") is False


class TestChunks:
    def test_insert_and_query(self, db: Database) -> None:
        db.insert_document("d1", "/a.md", "h1", 2, "model")
        db.insert_chunks([
            ("c1", "d1", 0, "Title", "First chunk", [0.1, 0.2, 0.3]),
            ("c2", "d1", 1, "Title", "Second chunk", [0.4, 0.5, 0.6]),
        ])
        chunks = db.document_chunks("d1")
        assert len(chunks) == 2
        assert chunks[0].content == "First chunk"
        assert chunks[1].chunk_index == 1

    def test_adjacent_chunks(self, db: Database) -> None:
        db.insert_document("d1", "/a.md", "h1", 5, "model")
        for i in range(5):
            db.insert_chunks([(f"c{i}", "d1", i, "", f"chunk {i}", [float(i)])])
        adj = db.adjacent_chunks("d1", 1, 3)
        assert len(adj) == 3
        assert adj[0].chunk_index == 1
        assert adj[2].chunk_index == 3


class TestSimilaritySearch:
    def test_cosine_similarity(self, db: Database) -> None:
        db.insert_document("d1", "/a.md", "h1", 2, "model-a")
        db.insert_chunks([
            ("c1", "d1", 0, "A", "Similar", [1.0, 0.0, 0.0]),
            ("c2", "d1", 1, "B", "Different", [0.0, 1.0, 0.0]),
        ])
        results = db.similar_chunks([1.0, 0.0, 0.0], "model-a", top_k=2)
        assert len(results) == 2
        assert results[0].content == "Similar"
        assert results[0].score > results[1].score

    def test_embedding_model_filtering(self, db: Database) -> None:
        db.insert_document("d1", "/a.md", "h1", 1, "model-a")
        db.insert_document("d2", "/b.md", "h2", 1, "model-b")
        db.insert_chunks([
            ("c1", "d1", 0, "", "chunk a", [1.0, 0.0]),
            ("c2", "d2", 0, "", "chunk b", [1.0, 0.0]),
        ])
        results = db.similar_chunks([1.0, 0.0], "model-a", top_k=10)
        assert len(results) == 1
        assert results[0].document_id == "d1"


class TestReplaceDocument:
    def test_atomic_replace(self, db: Database) -> None:
        db.insert_document("d1", "/a.md", "old_hash", 1, "model")
        db.insert_chunks([("c1", "d1", 0, "", "old", [0.1])])

        new_id = make_document_id("/a.md", "new_hash")
        db.replace_document(
            new_id, "/a.md", "new_hash", 2, "model",
            [
                (make_chunk_id(new_id, 0), new_id, 0, "", "new chunk 0", [0.2]),
                (make_chunk_id(new_id, 1), new_id, 1, "", "new chunk 1", [0.3]),
            ],
        )
        doc = db.find_document_by_path("/a.md")
        assert doc is not None
        assert doc.file_hash == "new_hash"
        chunks = db.document_chunks(new_id)
        assert len(chunks) == 2


class TestStaleDocuments:
    def test_list_stale(self, db: Database) -> None:
        db.insert_document("d1", "/a.md", "h1", 1, "old-model")
        db.insert_document("d2", "/b.md", "h2", 1, "new-model")
        stale = db.list_stale_documents("new-model")
        assert len(stale) == 1
        assert stale[0].id == "d1"

    def test_update_embeddings(self, db: Database) -> None:
        db.insert_document("d1", "/a.md", "h1", 1, "old-model")
        db.insert_chunks([("c1", "d1", 0, "", "text", [0.1])])
        db.update_chunk_embeddings("d1", [("c1", [0.9, 0.8])], "new-model")
        doc = db.find_document_by_path("/a.md")
        assert doc is not None
        assert doc.embedding_model == "new-model"
