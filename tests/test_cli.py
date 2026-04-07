"""Tests for gem-rag CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from gem_rag.cli import main


class TestCLI:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "gem-rag" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "index" in result.output
        assert "ask" in result.output
        assert "docs" in result.output
        assert "reindex" in result.output


class TestIndexCommand:
    def test_requires_dir_or_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["index"])
        assert result.exit_code != 0
        assert "One of --dir or --file is required" in result.output

    def test_mutual_exclusion(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("# Test", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["index", "--dir", str(tmp_path), "--file", str(f)])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    @patch("gem_rag.cli.GeminiEmbedder")
    @patch("gem_rag.cli.get_config")
    def test_index_file(self, mock_config: MagicMock, mock_embedder_cls: MagicMock, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("# Test\nContent here.", encoding="utf-8")
        db_path = str(tmp_path / "test.db")

        mock_config.return_value = MagicMock(
            db_path=db_path, chunk_size=512, chunk_overlap=64, embedding_model="model"
        )
        mock_embedder_cls.return_value.model_name = "model"
        mock_embedder_cls.return_value.embed.return_value = [[0.1, 0.2, 0.3]]

        runner = CliRunner()
        result = runner.invoke(main, ["index", "--file", str(f)])
        assert result.exit_code == 0, result.output


class TestAskCommand:
    @patch("gem_rag.cli.get_config")
    def test_missing_project(self, mock_config: MagicMock) -> None:
        mock_config.side_effect = ValueError("GCP project ID is required")
        runner = CliRunner()
        result = runner.invoke(main, ["ask", "question"])
        assert result.exit_code != 0
        assert "GCP project ID is required" in result.output


class TestDocsCommand:
    def test_docs_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["docs", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "delete" in result.output
