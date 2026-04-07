# gem-rag Project Rules

## Purpose

Gemini-powered RAG CLI for Markdown documents. Indexes Markdown files into
DuckDB with Vertex AI Gemini embeddings, then answers questions using
vector search + context expansion + Gemini chat.

Replaces lite-rag (Go + local LLM) with Python + Vertex AI Gemini to solve:
- Local LLM memory requirements (now cloud-based)
- Windows cross-compile issues (Python + uv = cross-platform)

## Architecture

### Commands
```
gem-rag index --dir <directory>     Index *.md files
gem-rag index --file <file>         Index a single file
gem-rag ask <question>              Answer with streamed output
gem-rag ask --json <question>       Answer with JSON output
gem-rag docs list [--json]          List indexed documents
gem-rag docs show <id>              Show document content
gem-rag docs delete <id>            Delete document
gem-rag reindex                     Re-embed with current model
```

### Module Structure
```
src/gem_rag/
  cli.py             Click CLI
  config.py          GemRagConfig (pydantic-settings, GEM_RAG_*)
  normalizer.py      NFKC, markdown strip, token estimation
  chunker.py         Heading-aware chunking, JP/EN sentences
  sanitizer.py       Nonce-tagged XML wrapping
  database.py        DuckDB (documents + chunks + vectors)
  indexer.py          Walk, hash, chunk, embed, store
  retriever.py       Vector search, context expansion
  llm/
    client.py        GeminiClient (stream, retry)
    embedder.py      GeminiEmbedder (task_type)
```

## Security Rules

1. **No external transmission**: Only Vertex AI Gemini endpoint receives data.
2. **Prompt injection defense**: ALL document content and user queries MUST be
   wrapped in nonce-tagged XML blocks before LLM prompts. Nonces are
   collision-avoidant (checked against all text in the prompt).
3. **No secret logging**: API keys, tokens, and credentials must never appear
   in logs or output.
4. **Embedding model isolation**: Vector comparisons only within same model
   (enforced by `embedding_model` column in DB).

## Development Rules

- Small, focused modules
- Tests alongside code in `tests/`
- Type hints required
- CHANGELOG.md updated per feature
- No hardcoded credentials (ADC only)
- Python with uv (`uv sync` to install)
- Run tests: `uv run pytest tests/ -v`

## Configuration

```bash
GEM_RAG_PROJECT=your-gcp-project-id       # Required
GEM_RAG_LOCATION=us-central1              # Default
GEM_RAG_CHAT_MODEL=gemini-2.5-flash       # Default
GEM_RAG_EMBEDDING_MODEL=text-embedding-005 # Default
GEM_RAG_DB_PATH=./gem-rag.db              # Default
```

Authentication: Application Default Credentials (ADC).

## Communication Language

All communication between contributors and Claude Code is conducted in **Japanese**.
