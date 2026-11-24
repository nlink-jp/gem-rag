# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.1] - 2026-11-24

### Security

- The locked dependency versions are upgraded; `uv audit` reported 23 known
  advisories in the previous lock, all in transitive packages (urllib3,
  cryptography, pyasn1, anyio, certifi, idna, pydantic-settings, and pytest
  in the dev extra). The new lock has none. This affects `uv sync` from the
  repository; `uv tool install` already resolved current versions.

### Fixed

- The embedder now rejects a response that does not carry exactly one
  non-empty vector per input text. Before, a short response would have
  dropped the unpaired chunks while the document was still recorded as
  indexed, so a later `index` would not have repaired it.

### Changed

- `make test` and `make lint` run with the `dev` extra, so they work on a
  fresh `uv sync` without installing pytest or ruff separately. The existing
  lint and format violations are cleared; `make lint` now passes.
- `make lint` also runs pyright (configured in `pyproject.toml` but never
  run before); its existing errors are cleared.
- `make vuln` runs `uv audit` on the lock and fails on a known advisory.

## [0.5.0] - 2026-11-24

### Security

- **CVE-2026-111497 is fixed** (the out-of-bounds write in the Brotli decoder
  that DuckDB's parquet extension bundles). The `duckdb` dependency moves from
  `>=1.0,<2.0` (1.5.1 in the lock file) to `>=2.0.1,<3.0`: DuckDB v2.0.1 carries
  the upstream decoder fix (duckdb/duckdb#27791), and the 1.5 line has not
  received it. gem-rag runs only its own fixed SQL and never reads Parquet, so
  the decoder was linked but not reached.

### Changed

- **An index created by this release does not open in older versions.** DuckDB
  2.0 writes a storage format DuckDB 1.x cannot read, so a new index database
  does not open in gem-rag 0.4.x. Indexes created earlier keep working here and
  stay readable by 0.4.x after this release has written to them (measured with
  DuckDB 1.5.1 and 2.0.1).

### Fixed

- **The tests no longer read the operator's own configuration.** They picked up
  `~/.config/gem-rag/config.toml`, `GEM_RAG_*` environment variables and a
  `.env` in the current directory, so two configuration tests failed on a
  machine with a config file. A shared fixture now gives every test an empty
  home and working directory; new tests cover reading `config.toml` and an
  environment variable overriding it.

## [0.4.1] - 2026-04-14

### Added

- TOML config file support (`~/.config/gem-rag/config.toml`)
- Configuration priority: CLI flags > env vars > .env > config.toml > defaults

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
