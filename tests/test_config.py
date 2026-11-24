"""Tests for gem-rag configuration."""

from pathlib import Path

import pytest

from gem_rag.config import GemRagConfig, get_config


class TestGemRagConfig:
    def test_defaults(self) -> None:
        config = GemRagConfig(project="test-project")
        assert config.project == "test-project"
        assert config.location == "us-central1"
        assert config.chat_model == "gemini-2.5-flash"
        assert config.embedding_model == "text-embedding-005"
        assert config.db_path == "./gem-rag.db"
        assert config.top_k == 5
        assert config.context_window == 1
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEM_RAG_PROJECT", "env-project")
        monkeypatch.setenv("GEM_RAG_LOCATION", "asia-northeast1")
        monkeypatch.setenv("GEM_RAG_TOP_K", "10")
        config = GemRagConfig()
        assert config.project == "env-project"
        assert config.location == "asia-northeast1"
        assert config.top_k == 10

    def test_from_config_toml(self, isolate_config_sources: Path) -> None:
        cfg = isolate_config_sources / ".config" / "gem-rag" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text('[gcp]\nproject = "toml-project"\nlocation = "global"\n', encoding="utf-8")
        config = GemRagConfig()
        assert config.project == "toml-project"
        assert config.location == "global"

    def test_env_overrides_config_toml(self, isolate_config_sources: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = isolate_config_sources / ".config" / "gem-rag" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text('[gcp]\nlocation = "global"\n', encoding="utf-8")
        monkeypatch.setenv("GEM_RAG_LOCATION", "asia-northeast1")
        assert GemRagConfig().location == "asia-northeast1"


class TestGetConfig:
    def test_missing_project_raises(self) -> None:
        with pytest.raises(ValueError, match="GCP project ID is required"):
            get_config()

    def test_override_filters_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEM_RAG_PROJECT", "env-project")
        config = get_config(project="", location="")
        assert config.project == "env-project"

    def test_override_applies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEM_RAG_PROJECT", "env-project")
        config = get_config(project="cli-project")
        assert config.project == "cli-project"
