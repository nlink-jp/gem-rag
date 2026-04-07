"""Retriever: vector search, context expansion, span merging."""

from __future__ import annotations

from dataclasses import dataclass

from gem_rag.database import Database, ScoredChunk
from gem_rag.llm.embedder import GeminiEmbedder


@dataclass
class Passage:
    """A retrieved passage with score and source info."""

    content: str
    score: float
    heading_path: str
    file_path: str
    document_id: str


class Retriever:
    """Retrieve relevant passages from indexed documents."""

    def __init__(
        self,
        db: Database,
        embedder: GeminiEmbedder,
        *,
        top_k: int = 5,
        context_window: int = 1,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._top_k = top_k
        self._context_window = context_window

    def retrieve(self, query: str) -> list[Passage]:
        """Retrieve relevant passages for a query."""
        query_embedding = self._embedder.embed_query(query)

        hits = self._db.similar_chunks(
            query_embedding,
            self._embedder.model_name,
            top_k=self._top_k,
        )

        if not hits or self._context_window <= 0:
            return [_hit_to_passage(h) for h in hits]

        return self._expand_context(hits)

    def _expand_context(self, hits: list[ScoredChunk]) -> list[Passage]:
        """Expand hits with adjacent chunks, merge overlapping spans."""
        # Group by document
        doc_spans: dict[str, list[_Span]] = {}
        for hit in hits:
            spans = doc_spans.setdefault(hit.document_id, [])
            lo = max(0, hit.chunk_index - self._context_window)
            hi = hit.chunk_index + self._context_window
            spans.append(_Span(lo=lo, hi=hi, score=hit.score, heading_path=hit.heading_path, file_path=hit.file_path))

        # Merge overlapping spans per document
        passages: list[Passage] = []
        for doc_id, spans in doc_spans.items():
            merged = _merge_spans(spans)
            for span in merged:
                chunks = self._db.adjacent_chunks(doc_id, span.lo, span.hi)
                content = "\n".join(c.content for c in chunks)
                passages.append(Passage(
                    content=content,
                    score=span.score,
                    heading_path=span.heading_path,
                    file_path=span.file_path,
                    document_id=doc_id,
                ))

        passages.sort(key=lambda p: p.score, reverse=True)
        return passages


@dataclass
class _Span:
    lo: int
    hi: int
    score: float
    heading_path: str
    file_path: str


def _merge_spans(spans: list[_Span]) -> list[_Span]:
    """Merge overlapping/adjacent spans, keeping the best score and heading."""
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: s.lo)
    merged: list[_Span] = [sorted_spans[0]]
    for s in sorted_spans[1:]:
        last = merged[-1]
        if s.lo <= last.hi + 1:  # adjacent or overlapping
            last.hi = max(last.hi, s.hi)
            if s.score > last.score:
                last.score = s.score
                last.heading_path = s.heading_path
        else:
            merged.append(s)
    return merged


def _hit_to_passage(hit: ScoredChunk) -> Passage:
    return Passage(
        content=hit.content,
        score=hit.score,
        heading_path=hit.heading_path,
        file_path=hit.file_path,
        document_id=hit.document_id,
    )
