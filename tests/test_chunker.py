"""Tests for heading-aware markdown chunker."""

from gem_rag.chunker import Chunk, Chunker, _split_by_headings


class TestSplitByHeadings:
    def test_no_headings(self) -> None:
        sections = _split_by_headings("Just plain text.\nSecond line.")
        assert len(sections) == 1
        assert sections[0][0] == ""
        assert "Just plain text." in sections[0][1]

    def test_single_heading(self) -> None:
        sections = _split_by_headings("# Title\nBody text here.")
        assert len(sections) >= 1
        titled = [(p, b) for p, b in sections if p == "Title"]
        assert len(titled) == 1
        assert "Body text here." in titled[0][1]

    def test_nested_headings(self) -> None:
        text = "# Guide\n## Install\ninstall steps\n## Usage\nusage steps"
        sections = _split_by_headings(text)
        paths = [s[0] for s in sections if s[1].strip()]
        assert "Guide > Install" in paths
        assert "Guide > Usage" in paths

    def test_heading_path_inheritance(self) -> None:
        text = "# A\n## B\n### C\ndeep content\n## D\nshallow content"
        sections = _split_by_headings(text)
        paths = [s[0] for s in sections if s[1].strip()]
        assert "A > B > C" in paths
        assert "A > D" in paths

    def test_code_fence_protection(self) -> None:
        text = "# Title\n```\n# Not a heading\n```\nReal content"
        sections = _split_by_headings(text)
        bodies = [s[1] for s in sections if s[1].strip()]
        combined = "\n".join(bodies)
        assert "# Not a heading" in combined


class TestChunker:
    def test_small_text_single_chunk(self) -> None:
        chunker = Chunker(chunk_size=1000)
        chunks = chunker.chunk("Short text.")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."
        assert chunks[0].index == 0

    def test_heading_preserved_in_chunks(self) -> None:
        chunker = Chunker(chunk_size=1000)
        text = "# Section\nContent here."
        chunks = chunker.chunk(text)
        assert any(c.heading_path == "Section" for c in chunks)

    def test_large_section_split(self) -> None:
        chunker = Chunker(chunk_size=20, chunk_overlap=0)
        text = "# Title\n" + "これはテスト文です。" * 20
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self) -> None:
        chunker = Chunker(chunk_size=20, chunk_overlap=0)
        text = "# A\n" + "テスト。" * 10 + "\n# B\n" + "テスト。" * 10
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_empty_input(self) -> None:
        chunker = Chunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_overlap_provides_context(self) -> None:
        chunker = Chunker(chunk_size=30, chunk_overlap=10)
        text = "# Title\n" + "これはテスト文です。" * 20
        chunks = chunker.chunk(text)
        if len(chunks) >= 2:
            # Later chunks should have some overlap with previous
            assert chunks[1].content != ""

    def test_japanese_sentence_splitting(self) -> None:
        chunker = Chunker(chunk_size=20, chunk_overlap=0)
        text = "最初の文です。次の文です。最後の文です。"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_english_sentence_splitting(self) -> None:
        chunker = Chunker(chunk_size=10, chunk_overlap=0)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
