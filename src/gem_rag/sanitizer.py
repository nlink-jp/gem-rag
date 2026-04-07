"""Prompt injection defense with collision-avoidant nonce-tagged XML wrapping.

Generates a cryptographically random nonce that does NOT appear as a
substring in any provided text, preventing tag-escape attacks.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(?:(?:previous|all|above|prior)\s+)*instructions?", "Instruction override attempt"),
    (r"forget\s+(everything|all|previous|prior)", "Memory wipe attempt"),
    (r"you\s+are\s+now\s+", "Persona reassignment attempt"),
    (r"new\s+instructions?\s*:", "New instruction injection"),
    (r"system\s*:\s*", "System prompt injection marker"),
    (r"<\s*/?system\s*>", "XML system tag injection"),
    (r"<\s*/?instructions?\s*>", "XML instructions tag injection"),
    (r"\[INST\]", "Llama instruction marker"),
    (r"###\s*instruction", "Markdown instruction header injection"),
    (r"act\s+as\s+", "Role-play directive"),
    (r"roleplay\s+as", "Role-play directive"),
    (r"pretend\s+(you\s+are|to\s+be)", "Persona pretend directive"),
    (r"disregard\s+(previous|all|above|prior)", "Instruction disregard attempt"),
    (r"override\s+(previous|system|all)\s+(prompt|instructions?)?", "System override attempt"),
]

_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), description)
    for pattern, description in _INJECTION_PATTERNS
]


@dataclass
class SanitizationResult:
    text: str
    nonce: str
    has_risk: bool
    warnings: list[str] = field(default_factory=list)


def generate_nonce() -> str:
    """Generate a 32-character hex nonce (128 bits)."""
    return secrets.token_hex(16)


def generate_nonce_not_in(*texts: str) -> str:
    """Generate a nonce that does not appear as a substring in any provided text.

    This prevents an attacker from crafting input that contains the closing tag.
    """
    combined = "".join(texts)
    for _ in range(100):
        nonce = generate_nonce()
        if nonce not in combined:
            return nonce
    # Astronomically unlikely to reach here with 128-bit nonces
    raise RuntimeError("Failed to generate collision-free nonce after 100 attempts")


def detect_injection(text: str) -> list[str]:
    """Detect potential prompt injection patterns in text."""
    warnings = []
    for compiled_pattern, description in _COMPILED_PATTERNS:
        match = compiled_pattern.search(text)
        if match:
            warnings.append(f"{description}: matched '{match.group(0)}' at position {match.start()}")
    return warnings


def sanitize_for_llm(text: str, *, tag_name: str = "user_data", nonce: str | None = None) -> SanitizationResult:
    """Sanitize text with nonce-tagged XML wrapping.

    Wraps text in <{tag_name}-{nonce}>...</{tag_name}-{nonce}> tags.
    """
    if nonce is None:
        nonce = generate_nonce_not_in(text)

    warnings = detect_injection(text)
    safe_text = f"<{tag_name}-{nonce}>\n{text}\n</{tag_name}-{nonce}>"

    return SanitizationResult(
        text=safe_text,
        nonce=nonce,
        has_risk=len(warnings) > 0,
        warnings=warnings,
    )


def build_tag(tag_name: str, nonce: str) -> str:
    """Return the opening tag."""
    return f"<{tag_name}-{nonce}>"


def build_tag_close(tag_name: str, nonce: str) -> str:
    """Return the closing tag."""
    return f"</{tag_name}-{nonce}>"
