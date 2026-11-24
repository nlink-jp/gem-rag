# gem-rag

Gemini-powered RAG CLI for Markdown documents.

- **Language**: Python 3.11+ / uv
- **LLM**: Vertex AI Gemini (google-genai SDK, ADC auth)
- **Embedding**: Gemini text-embedding-005 (task_type: RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)
- **Storage**: DuckDB 2.0 (`duckdb>=2.0.1`; list_cosine_similarity for vector search).
  A database created by DuckDB 2.0 does not open in DuckDB 1.x (gem-rag ≤ 0.4.x)
- **Tests**: `tests/conftest.py` isolates HOME, `GEM_RAG_*` env vars and the cwd
  for every test — config is read from all three
- **Series**: util-series
- **CLI**: `gem-rag index --dir <dir>`, `gem-rag ask <question>`, `gem-rag docs list/show/delete`, `gem-rag reindex`
- **Build**: `uv build --out-dir dist/` via `make build`
- **Test**: `uv run --extra dev pytest tests/ -v` via `make test`; lint with `make lint` (ruff check + format check + pyright); `make vuln` (uv audit, run before tagging)
- **Module path**: `src/gem_rag/`
- **Replaces**: lite-rag (Go + local LLM + CGO/DuckDB) — solves memory and Windows issues
