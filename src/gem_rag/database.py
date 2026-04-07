"""DuckDB database layer for document and chunk storage with vector search."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import duckdb


@dataclass
class DocumentRecord:
    id: str
    file_path: str
    file_hash: str
    total_chunks: int
    indexed_at: str
    embedding_model: str


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    chunk_index: int
    heading_path: str
    content: str


@dataclass
class ScoredChunk:
    id: str
    document_id: str
    chunk_index: int
    heading_path: str
    content: str
    score: float
    file_path: str


def make_document_id(file_path: str, file_hash: str) -> str:
    return hashlib.sha256(f"{file_path}:{file_hash}".encode()).hexdigest()


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{document_id}:{chunk_index}".encode()).hexdigest()


class Database:
    """DuckDB wrapper for document and chunk storage."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = duckdb.connect(path)
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def _migrate(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id              TEXT PRIMARY KEY,
                file_path       TEXT NOT NULL,
                file_hash       TEXT NOT NULL,
                total_chunks    INTEGER NOT NULL,
                indexed_at      TEXT NOT NULL,
                embedding_model TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id           TEXT PRIMARY KEY,
                document_id  TEXT NOT NULL,
                chunk_index  INTEGER NOT NULL,
                heading_path TEXT DEFAULT '',
                content      TEXT NOT NULL,
                embedding    FLOAT[]
            )
        """)

    def find_document_by_path(self, file_path: str) -> DocumentRecord | None:
        row = self._conn.execute(
            "SELECT id, file_path, file_hash, total_chunks, indexed_at, embedding_model "
            "FROM documents WHERE file_path = ?",
            [file_path],
        ).fetchone()
        if row is None:
            return None
        return DocumentRecord(*row)

    def insert_document(
        self,
        doc_id: str,
        file_path: str,
        file_hash: str,
        total_chunks: int,
        embedding_model: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO documents (id, file_path, file_hash, total_chunks, indexed_at, embedding_model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [doc_id, file_path, file_hash, total_chunks, now, embedding_model],
        )

    def insert_chunks(
        self,
        chunks: list[tuple[str, str, int, str, str, list[float]]],
    ) -> None:
        """Insert chunks as (id, document_id, chunk_index, heading_path, content, embedding)."""
        for chunk in chunks:
            self._conn.execute(
                "INSERT INTO chunks (id, document_id, chunk_index, heading_path, content, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                list(chunk),
            )

    def replace_document(
        self,
        doc_id: str,
        file_path: str,
        file_hash: str,
        total_chunks: int,
        embedding_model: str,
        chunks: list[tuple[str, str, int, str, str, list[float]]],
    ) -> None:
        """Atomically replace a document and its chunks."""
        self._conn.execute("BEGIN TRANSACTION")
        try:
            # Delete old
            old = self.find_document_by_path(file_path)
            if old:
                self._conn.execute("DELETE FROM chunks WHERE document_id = ?", [old.id])
                self._conn.execute("DELETE FROM documents WHERE id = ?", [old.id])
            # Insert new
            self.insert_document(doc_id, file_path, file_hash, total_chunks, embedding_model)
            self.insert_chunks(chunks)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def similar_chunks(
        self,
        embedding: list[float],
        embedding_model: str,
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        rows = self._conn.execute(
            """
            SELECT c.id, c.document_id, c.chunk_index, c.heading_path, c.content,
                   list_cosine_similarity(c.embedding, ?::FLOAT[]) AS score,
                   d.file_path
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL AND len(c.embedding) > 0
              AND d.embedding_model = ?
            ORDER BY score DESC
            LIMIT ?
            """,
            [embedding, embedding_model, top_k],
        ).fetchall()
        return [ScoredChunk(*row) for row in rows]

    def adjacent_chunks(
        self,
        document_id: str,
        lo: int,
        hi: int,
    ) -> list[ChunkRecord]:
        rows = self._conn.execute(
            "SELECT id, document_id, chunk_index, heading_path, content "
            "FROM chunks WHERE document_id = ? AND chunk_index >= ? AND chunk_index <= ? "
            "ORDER BY chunk_index",
            [document_id, lo, hi],
        ).fetchall()
        return [ChunkRecord(*row) for row in rows]

    def list_documents(self) -> list[DocumentRecord]:
        rows = self._conn.execute(
            "SELECT id, file_path, file_hash, total_chunks, indexed_at, embedding_model "
            "FROM documents ORDER BY indexed_at DESC"
        ).fetchall()
        return [DocumentRecord(*row) for row in rows]

    def delete_document(self, doc_id: str) -> bool:
        self._conn.execute("DELETE FROM chunks WHERE document_id = ?", [doc_id])
        result = self._conn.execute("DELETE FROM documents WHERE id = ? RETURNING id", [doc_id]).fetchone()
        return result is not None

    def document_chunks(self, doc_id: str) -> list[ChunkRecord]:
        rows = self._conn.execute(
            "SELECT id, document_id, chunk_index, heading_path, content "
            "FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            [doc_id],
        ).fetchall()
        return [ChunkRecord(*row) for row in rows]

    def list_stale_documents(self, current_model: str) -> list[DocumentRecord]:
        rows = self._conn.execute(
            "SELECT id, file_path, file_hash, total_chunks, indexed_at, embedding_model "
            "FROM documents WHERE embedding_model != ?",
            [current_model],
        ).fetchall()
        return [DocumentRecord(*row) for row in rows]

    def update_chunk_embeddings(
        self,
        doc_id: str,
        embeddings: list[tuple[str, list[float]]],
        embedding_model: str,
    ) -> None:
        """Update embeddings for chunks and the document's embedding_model."""
        for chunk_id, embedding in embeddings:
            self._conn.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                [embedding, chunk_id],
            )
        self._conn.execute(
            "UPDATE documents SET embedding_model = ? WHERE id = ?",
            [embedding_model, doc_id],
        )
