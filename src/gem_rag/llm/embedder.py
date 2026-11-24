"""Gemini embedding client for gem-rag."""

from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types

from gem_rag.config import GemRagConfig

logger = logging.getLogger(__name__)

# Vertex AI allows at most 250 instances per embed_content request.
_EMBED_BATCH_SIZE = 250


class GeminiEmbedder:
    """Embed text using Vertex AI Gemini embedding models."""

    def __init__(self, config: GemRagConfig) -> None:
        self._config = config
        self._client = genai.Client(
            vertexai=True,
            project=config.project,
            location=config.location,
        )

    @property
    def model_name(self) -> str:
        return self._config.embedding_model

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Embed a batch of texts, splitting into API-sized batches automatically.

        Args:
            texts: Texts to embed.
            task_type: "RETRIEVAL_DOCUMENT" for indexing, "RETRIEVAL_QUERY" for search.
        """
        results: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            results.extend(self._embed_with_retry(batch, task_type=task_type))
        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query for retrieval."""
        results = self.embed([query], task_type="RETRIEVAL_QUERY")
        return results[0]

    def _embed_with_retry(
        self,
        texts: list[str],
        *,
        task_type: str,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> list[list[float]]:
        """Call embedding API with exponential backoff."""
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self._client.models.embed_content(
                    model=self._config.embedding_model,
                    contents=texts,
                    config={"task_type": task_type},
                )
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(
                    keyword in error_str for keyword in ("429", "resource_exhausted", "rate limit", "quota")
                )
                if not is_retryable or attempt == max_retries:
                    raise
                last_error = e
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Embedding API rate limited (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    e,
                )
                time.sleep(delay)
            else:
                return _vectors(response.embeddings, len(texts))

        raise last_error  # type: ignore[misc]


def _vectors(embeddings: list[types.ContentEmbedding] | None, expected: int) -> list[list[float]]:
    """Return one vector per input text, or raise if the response does not have exactly that.

    The SDK types both the list and each vector as optional. The indexer pairs
    chunks with vectors positionally, so a short or partly empty response would
    drop chunks while the document is still recorded as indexed.
    """
    if embeddings is None or len(embeddings) != expected:
        got = 0 if embeddings is None else len(embeddings)
        raise ValueError(f"embedding API returned {got} vectors for {expected} texts")
    vectors: list[list[float]] = []
    for i, e in enumerate(embeddings):
        if not e.values:
            raise ValueError(f"embedding API returned an empty vector for text {i}")
        vectors.append(e.values)
    return vectors
