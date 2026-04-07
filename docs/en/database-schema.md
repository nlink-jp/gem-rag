# Database Schema

gem-rag uses DuckDB as a single-file embedded database for document metadata,
chunk content, and embedding vectors.

## Tables

### `documents`

Stores metadata about each indexed file.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | SHA-256(`file_path` + `:` + `file_hash`) |
| `file_path` | TEXT | Absolute path to the source file |
| `file_hash` | TEXT | SHA-256 of file content (change detection) |
| `total_chunks` | INTEGER | Number of chunks produced |
| `indexed_at` | TEXT | ISO 8601 timestamp (UTC) |
| `embedding_model` | TEXT | Model name used for embedding (e.g. `text-embedding-005`) |

### `chunks`

Stores chunk content and embedding vectors.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | SHA-256(`document_id` + `:` + `chunk_index`) |
| `document_id` | TEXT | References `documents.id` |
| `chunk_index` | INTEGER | 0-based position within the document |
| `heading_path` | TEXT | Hierarchical heading context (e.g. `Guide > Install > Linux`) |
| `content` | TEXT | Normalized chunk text |
| `embedding` | FLOAT[] | DuckDB native array type for vector storage |

## Key Design Decisions

### Embedding Model Isolation

The `embedding_model` column ensures vector comparisons only occur between
embeddings from the same model. The similarity search query includes:

```sql
WHERE d.embedding_model = ?
```

This prevents cross-model contamination when switching embedding models.
The `reindex` command updates stale documents by checking this column.

### Deterministic IDs

Both document and chunk IDs are deterministic SHA-256 hashes:
- Document ID = `SHA-256(file_path + ":" + file_hash)` — same file + same content = same ID
- Chunk ID = `SHA-256(document_id + ":" + chunk_index)` — same document + same position = same ID

This enables idempotent indexing without UUID collisions.

### Idempotency Check

On `index`, the indexer compares:
1. `file_hash` — has the file content changed?
2. `embedding_model` — was it embedded with the current model?

If both match, the file is skipped. If either differs, the document is
atomically replaced (old chunks deleted, new chunks inserted in a transaction).

## Vector Search

Similarity search uses DuckDB's built-in `list_cosine_similarity()`:

```sql
SELECT c.id, c.document_id, c.chunk_index, c.heading_path, c.content,
       list_cosine_similarity(c.embedding, ?::FLOAT[]) AS score,
       d.file_path
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.embedding IS NOT NULL AND len(c.embedding) > 0
  AND d.embedding_model = ?
ORDER BY score DESC
LIMIT ?
```

## Adjacent Chunk Queries

Context expansion fetches ±N chunks around each hit:

```sql
SELECT id, document_id, chunk_index, heading_path, content
FROM chunks
WHERE document_id = ? AND chunk_index >= ? AND chunk_index <= ?
ORDER BY chunk_index
```

Overlapping spans from multiple hits are merged before fetching.
