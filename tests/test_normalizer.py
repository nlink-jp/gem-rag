"""Tests for text normalizer."""

from gem_rag.normalizer import estimate_tokens, normalize, strip_markdown


class TestNormalize:
    def test_nfkc_fullwidth_ascii(self) -> None:
        assert normalize("Ｈｅｌｌｏ") == "Hello"

    def test_fullwidth_space(self) -> None:
        assert normalize("hello\u3000world") == "hello world"

    def test_crlf_to_lf(self) -> None:
        assert normalize("a\r\nb\rc") == "a\nb\nc"

    def test_control_chars_removed(self) -> None:
        assert normalize("hello\x00\x01world") == "helloworld"

    def test_preserves_tab_and_lf(self) -> None:
        result = normalize("a\tb\nc")
        assert "\t" in result
        assert "\n" in result

    def test_consecutive_spaces_collapsed(self) -> None:
        assert normalize("a    b") == "a b"

    def test_empty_string(self) -> None:
        assert normalize("") == ""

    def test_japanese_text_preserved(self) -> None:
        text = "会議の議事録を作成する"
        assert normalize(text) == text


class TestStripMarkdown:
    def test_image_keeps_alt(self) -> None:
        assert strip_markdown("![alt text](image.png)") == "alt text"

    def test_link_keeps_text(self) -> None:
        assert strip_markdown("[click here](https://example.com)") == "click here"

    def test_html_tags_removed(self) -> None:
        assert strip_markdown("<b>bold</b>") == "bold"

    def test_code_fence_removed(self) -> None:
        result = strip_markdown("before\n```python\ncode\n```\nafter")
        assert "code" not in result
        assert "before" in result
        assert "after" in result

    def test_inline_code_keeps_text(self) -> None:
        assert strip_markdown("use `grep` command") == "use grep command"

    def test_heading_markers_removed(self) -> None:
        assert strip_markdown("## Section Title") == "Section Title"

    def test_plain_text_unchanged(self) -> None:
        text = "Just a plain sentence."
        assert strip_markdown(text) == text


class TestEstimateTokens:
    def test_pure_cjk(self) -> None:
        # 11 CJK chars (incl. hiragana の/を) × 2 = 22 tokens
        assert estimate_tokens("会議の議事録を作成する") == 22

    def test_pure_english(self) -> None:
        # 5 words × 1.3 ≈ 7 tokens
        result = estimate_tokens("This is a test sentence")
        assert result == 7  # int(5 * 1.3 + 0.5)

    def test_mixed_jp_en(self) -> None:
        result = estimate_tokens("会議 meeting テスト test")
        assert result > 0

    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0
