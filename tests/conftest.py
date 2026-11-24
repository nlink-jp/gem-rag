"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_config_sources(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep the operator's own configuration out of every test.

    GemRagConfig reads ~/.config/gem-rag/config.toml, GEM_RAG_* environment
    variables and a .env file in the current directory. Without this fixture
    the results depended on the machine the tests ran on. Returns the fake
    home directory so a test can place a config.toml there.
    """
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    for name in list(os.environ):
        if name.startswith("GEM_RAG_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    return home
