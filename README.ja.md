# gem-rag

Gemini を活用した Markdown ドキュメント向け RAG CLI — Vertex AI embedding と
DuckDB ベクトルストレージによるインデックス・検索・質問応答。

## 特徴

- **Markdown インデックス**: 見出し認識チャンキング、日英文境界検出
- **セマンティック検索**: Gemini text-embedding-005 + DuckDB コサイン類似度
- **コンテキスト認識回答**: ベクトル検索 + 隣接チャンク展開 + Gemini チャット
- **クロスプラットフォーム**: Python + uv、macOS / Linux / Windows 対応
- **プロンプトインジェクション防御**: ノンスタグ XML ラッピング

## 前提条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ
- Vertex AI API が有効な Google Cloud プロジェクト
- Application Default Credentials:
  ```bash
  gcloud auth application-default login
  ```

## インストール

```bash
# ソースから
git clone https://github.com/nlink-jp/gem-rag.git
cd gem-rag
uv sync

# ツールとしてインストール
uv tool install gem-rag
```

## 設定

設定は以下の優先順位で解決されます:

1. **CLI フラグ**（最優先）
2. **環境変数** (`GEM_RAG_*`)
3. **`.env` ファイル**（カレントディレクトリ）
4. **設定ファイル** (`~/.config/gem-rag/config.toml`)
5. **デフォルト値**（最低優先）

### 設定ファイル

`~/.config/gem-rag/config.toml` を作成:

```toml
project = "your-gcp-project-id"
location = "us-central1"
chat_model = "gemini-2.5-flash"
embedding_model = "text-embedding-005"
db_path = "./gem-rag.db"
```

完全な例は [`config.example.toml`](config.example.toml) を参照してください。

### 環境変数

環境変数を設定（または `.env` ファイルを作成）:

```bash
GEM_RAG_PROJECT=your-gcp-project-id       # 必須
GEM_RAG_LOCATION=us-central1              # デフォルト
GEM_RAG_CHAT_MODEL=gemini-2.5-flash       # デフォルト
GEM_RAG_EMBEDDING_MODEL=text-embedding-005 # デフォルト
GEM_RAG_DB_PATH=./gem-rag.db              # デフォルト
```

## 使い方

```bash
# Markdown ファイルをインデックス
gem-rag index --dir ./docs

# 質問する
gem-rag ask "認証の仕組みは？"

# JSON 出力（ソース付き）
gem-rag ask --json "API エンドポイントは？"

# ドキュメント管理
gem-rag docs list
gem-rag docs delete <id>

# embedding モデル変更後に再インデックス
gem-rag reindex
```

## ビルド

```bash
make build    # パッケージを dist/ にビルド
make test     # テスト実行
make lint     # リンター実行
```

## ドキュメント

- [データベーススキーマ](docs/ja/database-schema.ja.md) — DuckDB テーブル、ベクトル検索、冪等性
- [チャンキングアーキテクチャ](docs/ja/chunking-architecture.ja.md) — 見出し認識分割、トークン推定、日英文境界
- [クロス言語検索](docs/ja/cross-language-search.ja.md) — JA/EN クエリリライト、並行検索、結果マージ
- [English documentation](README.md)

## ライセンス

MIT
