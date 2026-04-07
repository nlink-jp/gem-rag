"""Gemini LLM client for gem-rag."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import types

from gem_rag.config import GemRagConfig

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper around the google-genai SDK for Vertex AI Gemini."""

    def __init__(self, config: GemRagConfig) -> None:
        self._config = config
        self._client = genai.Client(
            vertexai=True,
            project=config.project,
            location=config.location,
        )

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt and return raw text response."""
        return self._call_with_retry(system_prompt=system_prompt, user_prompt=user_prompt)

    def stream_text(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Send a prompt and yield text chunks as they arrive."""
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        )
        response = self._client.models.generate_content_stream(
            model=self._config.chat_model,
            contents=user_prompt,
            config=config,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> str:
        """Call Gemini API with exponential backoff on rate limit errors."""
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        )

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._config.chat_model,
                    contents=user_prompt,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(
                    keyword in error_str
                    for keyword in ("429", "resource_exhausted", "rate limit", "quota")
                )
                if not is_retryable or attempt == max_retries:
                    raise
                last_error = e
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Gemini API rate limited (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries + 1, delay, e,
                )
                time.sleep(delay)

        raise last_error  # type: ignore[misc]
