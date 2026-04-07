"""Configuration management for gem-rag."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class GemRagConfig(BaseSettings):
    """gem-rag configuration loaded from environment variables."""

    # Gemini
    project: str = Field(default="", description="GCP project ID")
    location: str = Field(default="us-central1", description="GCP location")
    chat_model: str = Field(default="gemini-2.5-flash", description="Chat model name")
    embedding_model: str = Field(default="text-embedding-005", description="Embedding model name")

    # Database
    db_path: str = Field(default="./gem-rag.db", description="DuckDB database file path")

    # Retrieval
    top_k: int = Field(default=5, description="Number of top chunks to retrieve")
    context_window: int = Field(default=1, description="Adjacent chunks to expand (0=disabled)")
    chunk_size: int = Field(default=512, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=64, description="Overlap tokens between chunks")

    model_config = {"env_prefix": "GEM_RAG_", "env_file": ".env", "extra": "ignore"}


def get_config(**overrides: str | int) -> GemRagConfig:
    """Load config with CLI overrides. Empty strings and None are ignored."""
    filtered = {k: v for k, v in overrides.items() if v is not None and v != ""}
    config = GemRagConfig(**filtered)
    if not config.project:
        raise ValueError("GCP project ID is required. Set GEM_RAG_PROJECT environment variable or pass --project.")
    return config
