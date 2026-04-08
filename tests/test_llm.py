"""Tests for Gemini LLM client and embedder."""

from unittest.mock import MagicMock, patch

import pytest

from gem_rag.config import GemRagConfig
from gem_rag.llm.client import GeminiClient
from gem_rag.llm.embedder import GeminiEmbedder


class TestGeminiClient:
    @patch("gem_rag.llm.client.genai.Client")
    def test_complete_text(self, mock_cls: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text = "answer"
        mock_cls.return_value.models.generate_content.return_value = mock_response

        config = GemRagConfig(project="test")
        client = GeminiClient(config)
        result = client.complete_text("system", "user")
        assert result == "answer"

    @patch("gem_rag.llm.client.time.sleep")
    @patch("gem_rag.llm.client.genai.Client")
    def test_retry_on_429(self, mock_cls: MagicMock, mock_sleep: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_cls.return_value.models.generate_content.side_effect = [
            Exception("429 Resource Exhausted"),
            mock_response,
        ]

        config = GemRagConfig(project="test")
        client = GeminiClient(config)
        result = client.complete_text("system", "user")
        assert result == "ok"
        mock_sleep.assert_called_once()

    @patch("gem_rag.llm.client.genai.Client")
    def test_non_retryable_raises(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.models.generate_content.side_effect = Exception("Invalid")

        config = GemRagConfig(project="test")
        client = GeminiClient(config)
        with pytest.raises(Exception, match="Invalid"):
            client.complete_text("system", "user")

    @patch("gem_rag.llm.client.genai.Client")
    def test_stream_text(self, mock_cls: MagicMock) -> None:
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "World"
        mock_cls.return_value.models.generate_content_stream.return_value = [chunk1, chunk2]

        config = GemRagConfig(project="test")
        client = GeminiClient(config)
        result = list(client.stream_text("system", "user"))
        assert result == ["Hello ", "World"]


class TestGeminiEmbedder:
    @patch("gem_rag.llm.embedder.genai.Client")
    def test_embed_batch(self, mock_cls: MagicMock) -> None:
        emb1 = MagicMock()
        emb1.values = [0.1, 0.2, 0.3]
        emb2 = MagicMock()
        emb2.values = [0.4, 0.5, 0.6]
        mock_response = MagicMock()
        mock_response.embeddings = [emb1, emb2]
        mock_cls.return_value.models.embed_content.return_value = mock_response

        config = GemRagConfig(project="test")
        embedder = GeminiEmbedder(config)
        result = embedder.embed(["text1", "text2"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @patch("gem_rag.llm.embedder.genai.Client")
    def test_embed_query(self, mock_cls: MagicMock) -> None:
        emb = MagicMock()
        emb.values = [0.1, 0.2]
        mock_response = MagicMock()
        mock_response.embeddings = [emb]
        mock_cls.return_value.models.embed_content.return_value = mock_response

        config = GemRagConfig(project="test")
        embedder = GeminiEmbedder(config)
        result = embedder.embed_query("question")
        assert result == [0.1, 0.2]

    @patch("gem_rag.llm.embedder.genai.Client")
    def test_task_type_document(self, mock_cls: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1])]
        mock_cls.return_value.models.embed_content.return_value = mock_response

        config = GemRagConfig(project="test")
        embedder = GeminiEmbedder(config)
        embedder.embed(["text"], task_type="RETRIEVAL_DOCUMENT")

        call_args = mock_cls.return_value.models.embed_content.call_args
        assert call_args.kwargs["config"]["task_type"] == "RETRIEVAL_DOCUMENT"

    @patch("gem_rag.llm.embedder.genai.Client")
    def test_model_name(self, mock_cls: MagicMock) -> None:
        config = GemRagConfig(project="test", embedding_model="custom-model")
        embedder = GeminiEmbedder(config)
        assert embedder.model_name == "custom-model"

    @patch("gem_rag.llm.embedder.genai.Client")
    def test_embed_splits_into_batches(self, mock_cls: MagicMock) -> None:
        """Inputs exceeding 250 must be split into multiple API calls."""
        def make_response(**kwargs: object) -> MagicMock:
            contents = kwargs["contents"]
            resp = MagicMock()
            resp.embeddings = [MagicMock(values=[float(i)]) for i in range(len(contents))]
            return resp

        mock_cls.return_value.models.embed_content.side_effect = make_response

        config = GemRagConfig(project="test")
        embedder = GeminiEmbedder(config)
        texts = ["t"] * 300  # exceeds 250 limit
        result = embedder.embed(texts)

        assert len(result) == 300
        assert mock_cls.return_value.models.embed_content.call_count == 2
        # First batch: 250, second batch: 50
        calls = mock_cls.return_value.models.embed_content.call_args_list
        assert len(calls[0].kwargs["contents"]) == 250
        assert len(calls[1].kwargs["contents"]) == 50
