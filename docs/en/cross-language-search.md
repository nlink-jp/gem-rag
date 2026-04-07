# Cross-Language Search

gem-rag supports cross-language retrieval via LLM-powered query rewriting.
A Japanese question can find English documents, and vice versa.

## How It Works

```
User Query ("認証の仕組みは？")
    │
    ├─ Original query ──→ embed ──→ vector search ──┐
    │                                                │
    ├─ Gemini rewrite ──→ "JA: 認証の仕組みと方式"  │
    │                  ──→ "EN: Authentication mechanism and methods"
    │                      │                         │
    │                      ├─ embed JA ──→ search ──┤
    │                      └─ embed EN ──→ search ──┤
    │                                                │
    └───────────────── merge (max score per chunk) ──┘
                              │
                        Top-K results
```

## Query Rewriting

When `query_rewrite` is enabled (default: `true`), the user's query is sent to
Gemini with a system prompt that requests two reformulations:

```
JA: <rewritten query in Japanese>
EN: <rewritten query in English>
```

The rewriter:
1. Expands abbreviations and adds relevant synonyms
2. Converts interrogative form to declarative (better embedding match)
3. Produces both JA and EN variants regardless of input language

### Example

| Input | JA Variant | EN Variant |
|-------|-----------|-----------|
| "認証はどうなってる？" | "認証の仕組みと認証方式" | "Authentication mechanism and methods" |
| "How does indexing work?" | "インデックスの仕組みとチャンク処理" | "Indexing mechanism and chunking process" |

## Parallel Execution

All query variants (original + JA + EN) are embedded and searched **in parallel**
using `ThreadPoolExecutor`. This minimizes latency:

```
Time ──→

Original:  [embed]───[search]
JA:        [embed]───[search]
EN:        [embed]───[search]
                              ├── merge ── result
```

## Result Merging

Results from all searches are merged with deduplication:

```python
# For each chunk ID, keep the highest score
if chunk_id not in seen or new_score > seen[chunk_id].score:
    seen[chunk_id] = hit
```

This means a chunk that matches both the JA and EN variant will keep its
best score, not be counted twice.

## Security

The rewriter prompt uses **collision-avoidant nonce-tagged XML**:

```
<query-{nonce}>user question</query-{nonce}>
```

The nonce is a 128-bit hex string verified to not appear in the query text.
The system prompt explicitly instructs the LLM to treat tag content as data only.

## Configuration

| Parameter | Default | Environment Variable |
|-----------|---------|---------------------|
| `query_rewrite` | `true` | `GEM_RAG_QUERY_REWRITE` |

To disable (for lower latency or offline scenarios):

```bash
GEM_RAG_QUERY_REWRITE=false gem-rag ask "question"
```

## Why Cross-Language Search Matters

In multilingual codebases and documentation:
- README.md may be in English, README.ja.md in Japanese
- Code comments often mix languages
- Meeting notes (from meeting-note) may be in Japanese while referenced docs are in English

Without query rewriting, a Japanese question would only match Japanese chunks
(and vice versa), missing relevant content in the other language.
With query rewriting, both language spaces are searched simultaneously.
