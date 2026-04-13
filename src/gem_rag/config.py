"""Configuration management for gem-rag."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


def _load_toml(tool_name: str) -> dict[str, Any]:
    """Load TOML config from ~/.config/<tool_name>/config.toml if it exists."""
    path = Path.home() / ".config" / tool_name / "config.toml"
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    # Flatten [gcp] and [model] sections into top-level keys
    flat: dict[str, Any] = {}
    if "gcp" in data and isinstance(data["gcp"], dict):
        if "project" in data["gcp"]:
            flat["project"] = data["gcp"]["project"]
        if "location" in data["gcp"]:
            flat["location"] = data["gcp"]["location"]
    if "model" in data and isinstance(data["model"], dict):
        for k, v in data["model"].items():
            flat[k] = v
    # Pass through any top-level keys (e.g. db_path, top_k)
    for k, v in data.items():
        if k not in ("gcp", "model") and not isinstance(v, dict):
            flat[k] = v
    return flat


class GemRagConfig(BaseSettings):
    """gem-rag configuration loaded from config file and environment variables."""

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
    query_rewrite: bool = Field(default=True, description="Enable JA/EN query rewriting for cross-language search")

    model_config = {"env_prefix": "GEM_RAG_", "env_file": ".env", "extra": "ignore"}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority: init (CLI flags) > env vars > .env > config.toml > defaults."""
        from pydantic_settings import InitSettingsSource

        toml_data = _load_toml("gem-rag")
        toml_source = InitSettingsSource(settings_cls, init_kwargs=toml_data)
        return (init_settings, env_settings, dotenv_settings, toml_source, file_secret_settings)


def get_config(**overrides: str | int) -> GemRagConfig:
    """Load config with CLI overrides. Empty strings and None are ignored."""
    filtered = {k: v for k, v in overrides.items() if v is not None and v != ""}
    config = GemRagConfig(**filtered)
    if not config.project:
        raise ValueError("GCP project ID is required. Set GEM_RAG_PROJECT environment variable or pass --project.")
    return config
