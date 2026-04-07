"""Heading-aware Markdown chunker with JP/EN sentence boundary detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from gem_rag.normalizer import estimate_tokens


@dataclass
class Chunk:
    """A text chunk with heading context."""

    content: str
    heading_path: str
    index: int


class Chunker:
    """Split Markdown text into chunks by headings and sentence boundaries."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into heading-aware chunks."""
        sections = _split_by_headings(text)
        chunks: list[Chunk] = []
        idx = 0
        for heading_path, body in sections:
            if not body.strip():
                continue
            for piece in self._pack_section(body):
                chunks.append(Chunk(content=piece, heading_path=heading_path, index=idx))
                idx += 1
        return chunks

    def _pack_section(self, text: str) -> list[str]:
        """Pack a section into chunks respecting size limits."""
        if estimate_tokens(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        blocks = _atomize(text)
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for block in blocks:
            block_tokens = estimate_tokens(block)
            if current and current_tokens + block_tokens > self.chunk_size:
                result.append("\n".join(current).strip())
                # Retain overlap from tail
                current, current_tokens = self._trim_to_overlap(current)
            current.append(block)
            current_tokens += block_tokens

        if current:
            text_out = "\n".join(current).strip()
            if text_out:
                result.append(text_out)

        return result

    def _trim_to_overlap(self, blocks: list[str]) -> tuple[list[str], int]:
        """Retain blocks from the tail that fit within chunk_overlap tokens."""
        kept: list[str] = []
        total = 0
        for block in reversed(blocks):
            t = estimate_tokens(block)
            if total + t > self.chunk_overlap:
                break
            kept.insert(0, block)
            total += t
        return kept, total


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split text by Markdown headings, tracking hierarchical heading path."""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_body: list[str] = []
    in_fence = False

    for line in lines:
        stripped = line.strip()

        # Track code fences
        if stripped.startswith("```"):
            in_fence = not in_fence

        if in_fence:
            current_body.append(line)
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            # Flush previous section
            if current_body or heading_stack:
                path = _build_heading_path(heading_stack)
                body = "\n".join(current_body)
                sections.append((path, body))
                current_body = []

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # Pop headings at same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            current_body.append(line)

    # Flush final section
    path = _build_heading_path(heading_stack)
    body = "\n".join(current_body)
    sections.append((path, body))

    return sections


def _build_heading_path(stack: list[tuple[int, str]]) -> str:
    """Build a hierarchical heading path string."""
    if not stack:
        return ""
    return " > ".join(title for _, title in stack)


def _atomize(text: str) -> list[str]:
    """Split text into atomic blocks by paragraphs, then sentences."""
    paragraphs = _split_paragraphs(text)
    blocks: list[str] = []
    for para in paragraphs:
        if not para.strip():
            continue
        blocks.extend(_split_sentences(para))
    return blocks


def _split_paragraphs(text: str) -> list[str]:
    """Split by blank lines."""
    return re.split(r"\n\s*\n", text)


def _split_sentences(text: str) -> list[str]:
    """Split by JP and EN sentence boundaries."""
    parts = _SENTENCE_END_RE.split(text)
    sentences: list[str] = []
    current = ""
    for part in parts:
        current += part
        if _SENTENCE_END_RE.match(part):
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences if sentences else [text]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_SENTENCE_END_RE = re.compile(r"([。．！？]+|[.!?]+(?:\s+|$))")
