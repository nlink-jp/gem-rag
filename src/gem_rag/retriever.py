"""Retriever: vector search, context expansion, span merging, query rewriting."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from gem_rag.database import Database, ScoredChunk
from gem_rag.llm.embedder import GeminiEmbedder
from gem_rag.rewriter import QueryRewriter

logger = logging.getLogger(__name__)


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
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._top_k = top_k
        self._context_window = context_window
        self._rewriter = rewriter

    def retrieve(self, query: str) -> list[Passage]:
        """Retrieve relevant passages for a query."""
        hits = self._search(query)

        if not hits:
            return []

        if self._context_window <= 0:
            return [_hit_to_passage(h) for h in hits]

        return self._expand_context(hits)

    def _search(self, query: str) -> list[ScoredChunk]:
        """Search with optional query rewriting for cross-language retrieval."""
        queries = [query]
        if self._rewriter:
            variants = self._rewriter.rewrite(query)
            queries.extend(variants)

        # Parallel embedding and search for all query variants
        all_hits: list[ScoredChunk] = []
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            futures = {
                pool.submit(self._embed_and_search, q): q
                for q in queries
            }
            for future in as_completed(futures):
                try:
                    hits = future.result()
                    all_hits = _merge_hits(all_hits, hits)
                except Exception as e:
                    q = futures[future]
                    logger.warning("Search failed for variant %r: %s", q, e)

        # Sort by score and limit to top_k
        all_hits.sort(key=lambda h: h.score, reverse=True)
        return all_hits[: self._top_k]

    def _embed_and_search(self, query: str) -> list[ScoredChunk]:
        """Embed a single query and search."""
        embedding = self._embedder.embed_query(query)
        return self._db.similar_chunks(
            embedding,
            self._embedder.model_name,
            top_k=self._top_k,
        )

    def _expand_context(self, hits: list[ScoredChunk]) -> list[Passage]:
        """Expand hits with adjacent chunks, merge overlapping spans."""
        doc_spans: dict[str, list[_Span]] = {}
        for hit in hits:
            spans = doc_spans.setdefault(hit.document_id, [])
            lo = max(0, hit.chunk_index - self._context_window)
            hi = hit.chunk_index + self._context_window
            spans.append(_Span(lo=lo, hi=hi, score=hit.score, heading_path=hit.heading_path, file_path=hit.file_path))

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
        if s.lo <= last.hi + 1:
            last.hi = max(last.hi, s.hi)
            if s.score > last.score:
                last.score = s.score
                last.heading_path = s.heading_path
        else:
            merged.append(s)
    return merged


def _merge_hits(a: list[ScoredChunk], b: list[ScoredChunk]) -> list[ScoredChunk]:
    """Merge two hit lists, deduplicating by chunk ID and keeping max score."""
    if not b:
        return a
    seen: dict[str, ScoredChunk] = {}
    for h in a:
        seen[h.id] = h
    for h in b:
        if h.id not in seen or h.score > seen[h.id].score:
            seen[h.id] = h
    return list(seen.values())


def _hit_to_passage(hit: ScoredChunk) -> Passage:
    return Passage(
        content=hit.content,
        score=hit.score,
        heading_path=hit.heading_path,
        file_path=hit.file_path,
        document_id=hit.document_id,
    )
