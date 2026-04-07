# 新入社員オンボーディングガイド

## 初日の手続き

1. 人事部で入社手続きを完了する
2. IDカードを受け取る
3. PCとアカウントのセットアップを行う
4. セキュリティ研修を受講する

## 開発環境のセットアップ

### 必要なツール

- Git: `brew install git`
- Python 3.11+: `brew install python@3.11`
- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Google Cloud SDK: `brew install google-cloud-sdk`

### ADC認証の設定

```bash
gcloud auth application-default login
```

これにより、ローカル開発環境からVertex AI APIにアクセスできるようになります。

## チーム構成

| チーム | 責任範囲 | リーダー |
|--------|---------|---------|
| Platform | インフラ・CI/CD | 山田 |
| Security | セキュリティ監視・IR | 田中 |
| Product | 機能開発 | 佐藤 |

## 質問がある場合

- Slackの `#general` チャンネルで質問してください
- 緊急のセキュリティ問題は `#security-alerts` チャンネルに報告してください
