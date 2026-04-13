# gem-rag

Gemini-powered RAG CLI for Markdown documents — index, search, and answer
questions using Vertex AI embeddings and DuckDB vector storage.

## Features

- **Markdown indexing**: Heading-aware chunking with JP/EN sentence boundary detection
- **Semantic search**: Gemini text-embedding-005 with DuckDB cosine similarity
- **Context-aware answers**: Vector search + adjacent chunk expansion + Gemini chat
- **Cross-platform**: Python + uv, runs on macOS, Linux, and Windows
- **Prompt injection defense**: Nonce-tagged XML wrapping for all user content

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Cloud project with Vertex AI API enabled
- Application Default Credentials:
  ```bash
  gcloud auth application-default login
  ```

## Installation

```bash
# From source
git clone https://github.com/nlink-jp/gem-rag.git
cd gem-rag
uv sync

# Or install as a tool
uv tool install gem-rag
```

## Configuration

Configuration is resolved in the following priority order:

1. **CLI flags** (highest priority)
2. **Environment variables** (`GEM_RAG_*`)
3. **`.env` file** (in current directory)
4. **Config file** (`~/.config/gem-rag/config.toml`)
5. **Defaults** (lowest priority)

### Config file setup

Create `~/.config/gem-rag/config.toml`:

```toml
project = "your-gcp-project-id"
location = "us-central1"
chat_model = "gemini-2.5-flash"
embedding_model = "text-embedding-005"
db_path = "./gem-rag.db"
```

See [`config.example.toml`](config.example.toml) for a full example.

### Environment variables

Set environment variables (or create a `.env` file):

```bash
GEM_RAG_PROJECT=your-gcp-project-id       # Required
GEM_RAG_LOCATION=us-central1              # Default
GEM_RAG_CHAT_MODEL=gemini-2.5-flash       # Default
GEM_RAG_EMBEDDING_MODEL=text-embedding-005 # Default
GEM_RAG_DB_PATH=./gem-rag.db              # Default
```

## Usage

```bash
# Index Markdown files
gem-rag index --dir ./docs

# Ask a question
gem-rag ask "How does authentication work?"

# JSON output with sources
gem-rag ask --json "What are the API endpoints?"

# Manage documents
gem-rag docs list
gem-rag docs delete <id>

# Re-embed after changing embedding model
gem-rag reindex
```

## Building

```bash
make build    # Build package to dist/
make test     # Run tests
make lint     # Run linter
```

## Documentation

- [Database Schema](docs/en/database-schema.md) — DuckDB tables, vector search, idempotency
- [Chunking Architecture](docs/en/chunking-architecture.md) — heading-aware splitting, token estimation, JP/EN boundaries
- [Cross-Language Search](docs/en/cross-language-search.md) — JA/EN query rewriting, parallel search, result merging
- [日本語ドキュメント](README.ja.md)

## License

MIT
