"""Tests for indexer."""

from pathlib import Path
from unittest.mock import MagicMock

from gem_rag.chunker import Chunker
from gem_rag.database import Database
from gem_rag.indexer import Indexer


def _mock_embedder(dim: int = 3) -> MagicMock:
    embedder = MagicMock()
    embedder.model_name = "test-model"
    embedder.embed.side_effect = lambda texts, **kw: [[0.1] * dim for _ in texts]
    return embedder


class TestIndexFile:
    def test_index_new_file(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text("# Title\nContent here.", encoding="utf-8")

        db = Database(":memory:")
        indexer = Indexer(db, _mock_embedder(), Chunker())
        result = indexer.index_file(md)

        assert result is True
        docs = db.list_documents()
        assert len(docs) == 1
        assert docs[0].total_chunks >= 1
        db.close()

    def test_skip_unchanged(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text("# Title\nContent.", encoding="utf-8")

        db = Database(":memory:")
        indexer = Indexer(db, _mock_embedder(), Chunker())
        indexer.index_file(md)
        result = indexer.index_file(md)

        assert result is False  # Skipped
        db.close()

    def test_reindex_on_content_change(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text("# Title\nOriginal.", encoding="utf-8")

        db = Database(":memory:")
        indexer = Indexer(db, _mock_embedder(), Chunker())
        indexer.index_file(md)

        md.write_text("# Title\nUpdated content.", encoding="utf-8")
        result = indexer.index_file(md)

        assert result is True
        docs = db.list_documents()
        assert len(docs) == 1
        assert docs[0].file_hash != ""
        db.close()


class TestIndexDir:
    def test_index_multiple_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# A\nContent A", encoding="utf-8")
        (tmp_path / "b.md").write_text("# B\nContent B", encoding="utf-8")
        (tmp_path / "c.txt").write_text("Not markdown", encoding="utf-8")

        db = Database(":memory:")
        indexer = Indexer(db, _mock_embedder(), Chunker())
        count = indexer.index_dir(tmp_path)

        assert count == 2  # Only .md files
        db.close()

    def test_index_nested_dirs(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.md").write_text("# Root\nRoot content.", encoding="utf-8")
        (sub / "nested.md").write_text("# Nested\nNested content.", encoding="utf-8")

        db = Database(":memory:")
        indexer = Indexer(db, _mock_embedder(), Chunker())
        count = indexer.index_dir(tmp_path)

        assert count == 2
        db.close()


class TestReindex:
    def test_reindex_stale_documents(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text("# Title\nContent.", encoding="utf-8")

        db = Database(":memory:")
        old_embedder = _mock_embedder()
        old_embedder.model_name = "old-model"
        indexer = Indexer(db, old_embedder, Chunker())
        indexer.index_file(md)

        new_embedder = _mock_embedder()
        new_embedder.model_name = "new-model"
        new_indexer = Indexer(db, new_embedder, Chunker())
        count = new_indexer.reindex()

        assert count == 1
        docs = db.list_documents()
        assert docs[0].embedding_model == "new-model"
        db.close()
