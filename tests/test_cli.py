"""Tests for gem-rag CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from gem_rag.cli import _dedup_sources, main
from gem_rag.retriever import Passage


def _make_passage(file_path: str, heading_path: str, score: float) -> Passage:
    return Passage(content="x", score=score, heading_path=heading_path, file_path=file_path, document_id="d")


class TestDedupSources:
    def test_unique_files(self) -> None:
        passages = [
            _make_passage("a.md", "H1", 0.9),
            _make_passage("b.md", "H2", 0.7),
        ]
        result = _dedup_sources(passages)
        assert result == [("a.md", "H1", 0.9), ("b.md", "H2", 0.7)]

    def test_dedup_keeps_highest_score(self) -> None:
        passages = [
            _make_passage("a.md", "H1", 0.9),
            _make_passage("a.md", "H2", 0.95),
            _make_passage("b.md", "H3", 0.7),
        ]
        result = _dedup_sources(passages)
        assert len(result) == 2
        # a.md: heading from higher-score passage, score 0.95
        assert result[0] == ("a.md", "H2", 0.95)
        assert result[1] == ("b.md", "H3", 0.7)

    def test_preserves_first_seen_order(self) -> None:
        passages = [
            _make_passage("b.md", "H2", 0.7),
            _make_passage("a.md", "H1", 0.9),
            _make_passage("b.md", "H3", 0.6),
        ]
        result = _dedup_sources(passages)
        assert [fp for fp, _, _ in result] == ["b.md", "a.md"]

    def test_empty(self) -> None:
        assert _dedup_sources([]) == []


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

    @patch("gem_rag.cli.GeminiClient")
    @patch("gem_rag.cli.GeminiEmbedder")
    @patch("gem_rag.cli.Retriever")
    @patch("gem_rag.cli.Database")
    @patch("gem_rag.cli.get_config")
    def test_low_relevance_warning(
        self,
        mock_config: MagicMock,
        mock_db_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_config.return_value = MagicMock(
            db_path=str(tmp_path / "test.db"),
            top_k=5,
            context_window=1,
            query_rewrite=False,
        )
        low_score_passage = _make_passage("doc.md", "H1", 0.45)
        mock_retriever_cls.return_value.retrieve.return_value = [low_score_passage]
        mock_client_cls.return_value.stream_text.return_value = iter(["answer"])

        runner = CliRunner()
        result = runner.invoke(main, ["ask", "question"])
        assert result.exit_code == 0
        assert "Low relevance" in result.output
        assert "0.450" in result.output

    @patch("gem_rag.cli.GeminiClient")
    @patch("gem_rag.cli.GeminiEmbedder")
    @patch("gem_rag.cli.Retriever")
    @patch("gem_rag.cli.Database")
    @patch("gem_rag.cli.get_config")
    def test_no_low_relevance_warning_when_score_high(
        self,
        mock_config: MagicMock,
        mock_db_cls: MagicMock,
        mock_retriever_cls: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_config.return_value = MagicMock(
            db_path=str(tmp_path / "test.db"),
            top_k=5,
            context_window=1,
            query_rewrite=False,
        )
        high_score_passage = _make_passage("doc.md", "H1", 0.85)
        mock_retriever_cls.return_value.retrieve.return_value = [high_score_passage]
        mock_client_cls.return_value.stream_text.return_value = iter(["answer"])

        runner = CliRunner()
        result = runner.invoke(main, ["ask", "question"])
        assert result.exit_code == 0
        assert "Low relevance" not in result.output


class TestDocsCommand:
    def test_docs_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["docs", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "delete" in result.output
