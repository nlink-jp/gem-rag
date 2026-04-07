# gem-rag

Gemini-powered RAG CLI for Markdown documents.

- **Language**: Python 3.11+ / uv
- **LLM**: Vertex AI Gemini (google-genai SDK, ADC auth)
- **Embedding**: Gemini text-embedding-005 (task_type: RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)
- **Storage**: DuckDB (list_cosine_similarity for vector search)
- **Series**: util-series
- **CLI**: `gem-rag index --dir <dir>`, `gem-rag ask <question>`, `gem-rag docs list/show/delete`, `gem-rag reindex`
- **Build**: `uv build --out-dir dist/` via `make build`
- **Test**: `uv run pytest tests/ -v` via `make test`
- **Module path**: `src/gem_rag/`
- **Replaces**: lite-rag (Go + local LLM + CGO/DuckDB) — solves memory and Windows issues
