"""Tests for query rewriter."""

from unittest.mock import MagicMock

from gem_rag.rewriter import QueryRewriter, _parse_variants


class TestParseVariants:
    def test_both_ja_en(self) -> None:
        output = "JA: 認証の仕組み\nEN: How authentication works"
        result = _parse_variants(output)
        assert result == ["認証の仕組み", "How authentication works"]

    def test_reversed_order(self) -> None:
        output = "EN: authentication mechanism\nJA: 認証メカニズム"
        result = _parse_variants(output)
        assert result == ["認証メカニズム", "authentication mechanism"]

    def test_extra_whitespace(self) -> None:
        output = "JA:  認証  \n  EN:  auth  "
        result = _parse_variants(output)
        assert result == ["認証", "auth"]

    def test_malformed_fallback(self) -> None:
        output = "Some random response"
        result = _parse_variants(output)
        assert result == ["Some random response"]

    def test_empty_output(self) -> None:
        result = _parse_variants("")
        assert result == []

    def test_only_ja(self) -> None:
        output = "JA: 認証"
        result = _parse_variants(output)
        assert result == ["JA: 認証"]  # fallback to whole output


class TestQueryRewriter:
    def test_rewrite_success(self) -> None:
        client = MagicMock()
        client.complete_text.return_value = "JA: 認証の仕組みについて\nEN: How authentication works"
        rewriter = QueryRewriter(client)
        result = rewriter.rewrite("認証はどうなってる？")
        assert len(result) == 2
        assert "認証" in result[0]
        assert "authentication" in result[1]

    def test_rewrite_failure_fallback(self) -> None:
        client = MagicMock()
        client.complete_text.side_effect = Exception("API error")
        rewriter = QueryRewriter(client)
        result = rewriter.rewrite("original query")
        assert result == ["original query"]

    def test_nonce_in_prompt(self) -> None:
        client = MagicMock()
        client.complete_text.return_value = "JA: ja\nEN: en"
        rewriter = QueryRewriter(client)
        rewriter.rewrite("test")

        call_args = client.complete_text.call_args
        system_prompt = call_args.args[0]
        user_prompt = call_args.args[1]
        assert "query-" in system_prompt
        assert "<query-" in user_prompt
        assert "</query-" in user_prompt
