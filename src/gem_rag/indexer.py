"""Indexer: walk, hash, chunk, embed, store with idempotency."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from gem_rag.chunker import Chunker
from gem_rag.database import Database, make_chunk_id, make_document_id
from gem_rag.llm.embedder import GeminiEmbedder
from gem_rag.normalizer import normalize, strip_markdown

logger = logging.getLogger(__name__)


class Indexer:
    """Index Markdown files into DuckDB with Gemini embeddings."""

    def __init__(
        self,
        db: Database,
        embedder: GeminiEmbedder,
        chunker: Chunker,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._chunker = chunker

    def index_dir(self, directory: Path) -> int:
        """Index all *.md files in a directory recursively. Returns count of indexed files."""
        count = 0
        for md_file in sorted(directory.rglob("*.md")):
            try:
                if self.index_file(md_file):
                    count += 1
            except Exception as e:
                logger.error("Failed to index %s: %s", md_file, e)
        return count

    def index_file(self, path: Path) -> bool:
        """Index a single file. Returns True if indexed, False if skipped (unchanged)."""
        content = path.read_text(encoding="utf-8")
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        file_path = str(path.resolve())

        # Idempotency check
        existing = self._db.find_document_by_path(file_path)
        if existing and existing.file_hash == file_hash and existing.embedding_model == self._embedder.model_name:
            logger.debug("Skipping unchanged file: %s", file_path)
            return False

        # Normalize and chunk
        normalized = normalize(content)
        chunks = self._chunker.chunk(normalized)
        if not chunks:
            logger.warning("No chunks produced for %s", file_path)
            return False

        # Embed
        texts_to_embed = [strip_markdown(c.content) for c in chunks]
        embeddings = self._embedder.embed(texts_to_embed, task_type="RETRIEVAL_DOCUMENT")

        # Build DB records
        doc_id = make_document_id(file_path, file_hash)
        chunk_records = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = make_chunk_id(doc_id, chunk.index)
            chunk_records.append((chunk_id, doc_id, chunk.index, chunk.heading_path, chunk.content, embedding))

        # Store
        self._db.replace_document(
            doc_id, file_path, file_hash, len(chunks), self._embedder.model_name, chunk_records
        )
        logger.info("Indexed %s (%d chunks)", file_path, len(chunks))
        return True

    def reindex(self) -> int:
        """Re-embed documents whose embedding model differs from current. Returns count."""
        stale = self._db.list_stale_documents(self._embedder.model_name)
        count = 0
        for doc in stale:
            chunks = self._db.document_chunks(doc.id)
            if not chunks:
                continue
            texts = [strip_markdown(c.content) for c in chunks]
            embeddings = self._embedder.embed(texts, task_type="RETRIEVAL_DOCUMENT")
            updates = [(c.id, emb) for c, emb in zip(chunks, embeddings)]
            self._db.update_chunk_embeddings(doc.id, updates, self._embedder.model_name)
            logger.info("Re-indexed %s (%d chunks)", doc.file_path, len(chunks))
            count += 1
        return count
