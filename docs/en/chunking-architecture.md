# Chunking Architecture

gem-rag uses a heading-aware, multi-level chunking strategy optimized for
Markdown documents with mixed Japanese and English content.

## Pipeline

```
Raw Markdown
    │
    ├─ normalize()          Unicode NFKC, fullwidth→halfwidth, control chars
    │
    ├─ split_by_headings()  Split at # headings, track hierarchy
    │   │
    │   └─ For each section:
    │       │
    │       ├─ If tokens ≤ chunk_size → single chunk
    │       │
    │       └─ If tokens > chunk_size:
    │           │
    │           ├─ atomize()
    │           │   ├─ split_paragraphs()    Blank lines
    │           │   └─ split_sentences()     JP: 。．！？  EN: .!?
    │           │
    │           └─ pack with overlap
    │
    ├─ strip_markdown()     Remove formatting before embedding
    │
    └─ embed()              Gemini text-embedding-005
```

## Text Normalization

Applied before chunking to ensure consistent splitting:

| Step | Operation | Example |
|------|-----------|---------|
| 1 | Unicode NFKC | `Ｈｅｌｌｏ` → `Hello` |
| 2 | Fullwidth space | `\u3000` → ASCII space |
| 3 | Line endings | CRLF/CR → LF |
| 4 | Control chars | Remove C0/C1 (keep LF, TAB) |
| 5 | Consecutive spaces | `a    b` → `a b` |

## Heading-Aware Splitting

Documents are first split at Markdown heading boundaries (`#` through `######`).

### Hierarchical Heading Path

Each chunk inherits a **heading path** from its position in the document:

```markdown
# Guide
## Installation
### Linux
apt-get install ...     ← heading_path: "Guide > Installation > Linux"
### macOS
brew install ...        ← heading_path: "Guide > Installation > macOS"
## Usage
gem-rag ask "question"  ← heading_path: "Guide > Usage"
```

When a heading at level N is encountered, all headings at level ≥ N are
popped from the stack, and the new heading is pushed.

### Code Fence Protection

Headings inside fenced code blocks are not treated as section boundaries:

```markdown
# Real Heading
\`\`\`
# This is NOT a heading — it's code
\`\`\`
```

## Token Estimation

Mixed JP/EN token estimation (conservative, avoids truncation):

| Character type | Estimation | Rationale |
|---------------|------------|-----------|
| CJK (Han, Hiragana, Katakana) | 1 char = 2 tokens | CJK chars typically tokenize to 2+ subwords |
| ASCII/Latin words | 1 word = 1.3 tokens | English words average ~1.3 subword tokens |

Formula: `(cjk_chars × 2) + int(ascii_words × 1.3 + 0.5)`

## Sentence Boundary Detection

When a section exceeds `chunk_size`, it's atomized:

1. **Paragraph breaks** (blank lines) — primary split unit
2. **Japanese sentence endings**: `。` `．` `！` `？`
3. **English sentence endings**: `.` `!` `?` followed by whitespace

Pattern: `[。．！？]+|[.!?]+(?:\s+|$)`

## Overlap

After emitting a chunk, blocks from the tail are retained as seed for the
next chunk (up to `chunk_overlap` tokens). This ensures context continuity
across chunk boundaries.

## Markdown Stripping (Before Embedding)

Formatting is stripped before embedding to reduce noise in vector space:

| Pattern | Replacement |
|---------|-------------|
| `![alt](url)` | `alt` |
| `[text](url)` | `text` |
| `<html>` tags | removed |
| `` ```code``` `` fences | removed |
| `` `inline` `` code | `inline` |
| `## Heading` markers | removed |

The raw Markdown is stored in `chunks.content` for display.
Stripping is applied only at embedding time.

## Default Parameters

| Parameter | Default | Environment Variable |
|-----------|---------|---------------------|
| `chunk_size` | 512 tokens | `GEM_RAG_CHUNK_SIZE` |
| `chunk_overlap` | 64 tokens | `GEM_RAG_CHUNK_OVERLAP` |
