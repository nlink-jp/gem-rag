"""Text normalization and token estimation for indexing and embedding."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Normalize text for consistent indexing.

    1. Unicode NFKC (fullwidth→halfwidth, compatibility decomposition)
    2. Fullwidth space → ASCII space
    3. Line endings → LF
    4. Control characters removed (except LF, TAB)
    5. Consecutive whitespace collapsed
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _CONSECUTIVE_SPACES.sub(" ", text)
    return text.strip()


def strip_markdown(text: str) -> str:
    """Strip Markdown formatting, keeping readable text for embedding.

    Removes: images, links (keeps text), HTML tags, code fences, inline code markers.
    """
    text = _IMAGE_PATTERN.sub(r"\1", text)
    text = _LINK_PATTERN.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = _CODE_FENCE.sub("", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _HEADING_MARKER.sub("", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Estimate token count for mixed JP/EN text.

    CJK characters: 1 char ≈ 2 tokens
    ASCII/Latin words: 1 word ≈ 1.3 tokens
    """
    cjk_count = 0
    ascii_chars: list[str] = []

    for ch in text:
        if _is_cjk(ch):
            cjk_count += 1
        else:
            ascii_chars.append(ch)

    ascii_text = "".join(ascii_chars)
    word_count = len(ascii_text.split())

    return cjk_count * 2 + int(word_count * 1.3 + 0.5)


def _is_cjk(ch: str) -> bool:
    """Check if a character is CJK (Han, Hiragana, Katakana)."""
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)        # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)     # CJK Extension A
        or (0x3040 <= cp <= 0x309F)     # Hiragana
        or (0x30A0 <= cp <= 0x30FF)     # Katakana
        or (0xF900 <= cp <= 0xFAFF)     # CJK Compatibility Ideographs
        or (0x20000 <= cp <= 0x2A6DF)   # CJK Extension B
    )


# Pre-compiled patterns
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_CONSECUTIVE_SPACES = re.compile(r"[ \t]{2,}")
_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_CODE_FENCE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_HEADING_MARKER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
