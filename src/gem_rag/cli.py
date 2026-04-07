"""CLI entry point for gem-rag."""

from __future__ import annotations

import click

from gem_rag import __version__


@click.group()
@click.version_option(version=__version__, prog_name="gem-rag")
def main() -> None:
    """Gemini-powered RAG CLI for Markdown documents."""
    pass
