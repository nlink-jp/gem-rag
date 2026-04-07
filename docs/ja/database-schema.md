# データベーススキーマ

gem-rag は DuckDB を単一ファイルの組み込みデータベースとして使用し、
ドキュメントメタデータ、チャンクコンテンツ、embedding ベクトルを格納します。

## テーブル

### `documents`

インデックス済みファイルのメタデータを格納。

| カラム | 型 | 説明 |
|--------|-----|------|
| `id` | TEXT PK | SHA-256(`file_path` + `:` + `file_hash`) |
| `file_path` | TEXT | ソースファイルの絶対パス |
| `file_hash` | TEXT | ファイル内容の SHA-256（変更検出用） |
| `total_chunks` | INTEGER | 生成されたチャンク数 |
| `indexed_at` | TEXT | ISO 8601 タイムスタンプ（UTC） |
| `embedding_model` | TEXT | embedding に使用したモデル名（例: `text-embedding-005`） |

### `chunks`

チャンクのコンテンツと embedding ベクトルを格納。

| カラム | 型 | 説明 |
|--------|-----|------|
| `id` | TEXT PK | SHA-256(`document_id` + `:` + `chunk_index`) |
| `document_id` | TEXT | `documents.id` を参照 |
| `chunk_index` | INTEGER | ドキュメント内の 0 起点の位置 |
| `heading_path` | TEXT | 階層的な見出しコンテキスト（例: `Guide > Install > Linux`） |
| `content` | TEXT | 正規化済みチャンクテキスト |
| `embedding` | FLOAT[] | DuckDB ネイティブ配列型によるベクトル格納 |

## 主要な設計判断

### embedding モデル分離

`embedding_model` カラムにより、同一モデルの embedding 間のみでベクトル比較が行われます。
類似検索クエリには以下が含まれます:

```sql
WHERE d.embedding_model = ?
```

モデル切り替え時のクロスモデル汚染を防止します。
`reindex` コマンドはこのカラムをチェックして古いドキュメントを更新します。

### 決定論的 ID

ドキュメントとチャンクの ID は決定論的な SHA-256 ハッシュです:
- ドキュメント ID = `SHA-256(file_path + ":" + file_hash)` — 同じファイル + 同じ内容 = 同じ ID
- チャンク ID = `SHA-256(document_id + ":" + chunk_index)` — 同じドキュメント + 同じ位置 = 同じ ID

UUID 衝突なしの冪等なインデックスを実現します。

### 冪等性チェック

`index` 実行時に以下を比較:
1. `file_hash` — ファイル内容が変更されたか？
2. `embedding_model` — 現在のモデルで embedding されたか？

両方一致すればスキップ。いずれかが異なれば、トランザクション内で
原子的に置換（旧チャンク削除 → 新チャンク挿入）します。

## ベクトル検索

DuckDB 組み込みの `list_cosine_similarity()` を使用:

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

## 隣接チャンククエリ

コンテキスト展開で各ヒットの前後 ±N チャンクを取得:

```sql
SELECT id, document_id, chunk_index, heading_path, content
FROM chunks
WHERE document_id = ? AND chunk_index >= ? AND chunk_index <= ?
ORDER BY chunk_index
```

複数ヒットからの重複スパンは取得前にマージされます。
