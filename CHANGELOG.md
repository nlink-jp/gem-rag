# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-04-08

### Fixed

- Embedding API error when a file produces more than 250 chunks: `embed()` now
  automatically splits large batches into requests of ≤250 texts (Vertex AI limit)

## [0.3.0] - 2026-04-08

### Added

- Low relevance warning: when the top search score is below 0.60, `ask` now
  prints a warning before generating the answer (mirrors lite-rag behaviour)
- Source deduplication: `ask` deduplicates retrieved passages by file path,
  keeping the highest-scoring passage per file in both text and JSON output

## [0.2.0] - 2026-04-08

### Added

- Cross-language query rewriting: JA/EN parallel search via Gemini-powered
  query reformulation (enabled by default, `GEM_RAG_QUERY_REWRITE`)
- Documentation (en/ja): database schema, chunking architecture, cross-language search

## [0.1.0] - 2026-04-08

### Added

- `index` command: index Markdown files with heading-aware chunking and Gemini embeddings
  - Recursive directory walking for *.md files
  - SHA-256 file hash idempotency (unchanged files skipped)
  - JP/EN sentence boundary detection for chunk splitting
  - Hierarchical heading path preservation
- `ask` command: question answering with vector search and streamed Gemini responses
  - DuckDB cosine similarity search with adjacent chunk context expansion
  - Nonce-tagged XML wrapping for prompt injection defense (collision-avoidant 128-bit nonces)
  - `--json` flag for structured output with source attribution
- `docs` command group: list, show, delete indexed documents
- `reindex` command: re-embed documents after embedding model change
- Text normalization: NFKC, fullwidth→halfwidth, markdown stripping, JP/EN token estimation
- DuckDB storage with native `list_cosine_similarity()` vector search
- Gemini LLM client with exponential backoff retry on rate limits
- Gemini embedder with task_type support (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)
- Configuration via `GEM_RAG_*` environment variables with `.env` file support

### Security

- Nonce-tagged XML wrapping for all document content and user queries in LLM prompts
- Collision-avoidant nonce generation (verified not present in input text)
- 14-pattern prompt injection detection with warning logs
- Embedding model isolation in vector search (prevents cross-model contamination)
