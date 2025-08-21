# Smart Docs Chat - Phase 01 Local

社内ドキュメントを活用したRAG（Retrieval-Augmented Generation）チャットボットシステムです。
NotionとGoogle Driveから文書を収集し、高精度な検索と回答生成を実現します。

## 🎯 プロジェクト概要

### 主な機能
- **データ収集**: NotionとGoogle Driveからの自動文書収集
- **インデックス構築**: ChromaDBを使用したベクトル検索システム
- **RAGチャット**: OpenAI GPT-4を使用した高精度な質問応答
- **CLIインターフェース**: 対話的なコマンドラインチャット

### 達成した成果
- ✅ 検索精度: 0.69-0.88（目標0.5を大幅に超過）
- ✅ Notionページ: 182ページを正常に取得
- ✅ Google Driveファイル: 24ファイルを処理
- ✅ 設定管理: 秘密情報とシステム設定の分離

## 📁 ディレクトリ構造

```
phase01-local/
├── config/                 # 設定ファイル
│   ├── __init__.py        # 設定管理クラス
│   └── settings.yaml      # システム設定
├── credentials/           # 認証情報（.gitignore対象）
│   └── google_service_account.json
├── data/                  # データ格納
│   ├── chromadb/         # ベクトルDB
│   ├── documents/        # 収集した文書
│   │   ├── notion/       # Notionからの文書
│   │   └── google/       # Google Driveからの文書
│   └── chat_logs/        # チャット履歴
├── docs/                  # ドキュメント
│   ├── 01_成果報告書.md
│   ├── 02_使用ガイド.md
│   ├── 03_システム構成.md
│   ├── 04_フロー図_notion.md
│   ├── 05_フロー図_google_drive.md
│   └── 06_デバッグ設定.md
├── scripts/               # ユーティリティスクリプト
│   └── run.sh            # 実行ヘルパー
├── src/                   # ソースコード
│   ├── cli_chat.py       # CLIチャットインターフェース
│   ├── data_loader_notion.py  # Notionデータローダー
│   ├── data_loader_google.py  # Google Driveデータローダー
│   ├── indexer.py        # ベクトルDB構築
│   ├── rag_chain.py      # RAGチェーン実装
│   └── debug_helper.py   # デバッグユーティリティ
├── tests/                 # テストコード
│   └── test_connections.py
├── .env                   # 環境変数（秘密情報）
├── .env.example          # 環境変数テンプレート
├── main.py               # メインエントリーポイント
└── requirements.txt      # Pythonパッケージ
```

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリのクローン
git clone <repository-url>
cd phase01-local

# Python仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# パッケージのインストール
pip install -r requirements.txt
```

### 2. 設定ファイルの管理

本プロジェクトでは、**秘密情報**と**システム設定**を分離して管理しています。

#### 📁 設定ファイル構成
```
phase01-local/
├── .env                    # 秘密情報（APIキー等）※gitignore対象
├── .env.example           # .envのテンプレート
├── config/
│   ├── __init__.py       # 設定管理クラス
│   └── settings.yaml     # システム設定
```

#### 🔐 .env（環境変数 - 秘密情報）

`.env.example`を`.env`にコピーして設定：

```bash
cp .env.example .env
```

**必須項目：**
```bash
# OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx

# Notion
NOTION_TOKEN=secret_xxxxxxxxxxxxxxxxxxxx
NOTION_ROOT_PAGE_ID=xxxxxxxxxxxxxxxxxxxx

# Google Drive（オプション）
GOOGLE_DRIVE_FOLDER_ID=xxxxxxxxxxxxxxxxxxxx
GOOGLE_DRIVE_CREDENTIALS_PATH=./credentials/google_service_account.json
```

**特徴：**
- ✅ Gitにコミットしない（.gitignore対象）
- ✅ 環境ごとに異なる値
- ✅ 秘密情報のみを管理

#### ⚙️ config/settings.yaml（システム設定）

動作パラメータを管理（Gitコミット可能）：

```yaml
# ChromaDB設定
chromadb:
  persist_directory: "./data/chromadb"
  collection_name: "phase01_documents"

# 埋め込みモデル設定
embedding:
  model: "text-embedding-3-small"  # 高精度モデル
  # model: "text-embedding-ada-002"  # 低コストモデル

# チャンク分割設定
chunking:
  chunk_size: 500         # 文字数
  chunk_overlap: 100      # オーバーラップ

# 検索設定
retriever:
  k: 10                   # 取得する文書数
  score_threshold: 0.2    # 最小スコア閾値

# LLM設定
llm:
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1000
```

**特徴：**
- ✅ Gitにコミット可能
- ✅ チーム共有可能
- ✅ コメント付きで理解しやすい
- ✅ 環境変数でオーバーライド可能

#### 🔄 設定の優先順位

1. **環境変数**（最優先）
2. **settings.yaml**
3. **デフォルト値**

例：`EMBEDDING_MODEL=text-embedding-ada-002`を設定すると、settings.yamlの値を上書き

### 3. データ収集とインデックス構築

```bash
# 初期セットアップ（データ収集→インデックス構築）
python main.py setup

# または個別実行
python src/data_loader_notion.py   # Notionデータ収集
python src/data_loader_google.py   # Google Driveデータ収集
python src/indexer.py               # インデックス構築
```

### 4. チャットの開始

```bash
# CLIチャットインターフェース
python src/cli_chat.py

# またはメインから
python main.py chat
```

## 💬 使用方法

### CLIチャットコマンド

```
/help      - ヘルプを表示
/history   - 会話履歴を表示
/clear     - 履歴をクリア
/search    - ドキュメント検索のみ
/sources   - 参照ソース詳細
/save      - 会話履歴を保存
/stats     - セッション統計
/exit      - 終了
```

### RAGチェーンの直接使用

```python
from src.rag_chain import RAGChain

# 初期化
rag = RAGChain()

# 質問応答
result = rag.ask("就業規則について教えてください")
print(result["answer"])
print(f"スコア: {result['sources'][0]['score']}")
```

## 🔧 トラブルシューティング

### Notion APIエラー
- トークンの有効性を確認
- ルートページIDが正しいか確認
- APIの権限設定を確認

### Google Drive APIエラー
- サービスアカウントのJSONファイルが正しい場所にあるか確認
- Google DriveフォルダIDが正しいか確認
- サービスアカウントにフォルダへのアクセス権限があるか確認

### 検索精度が低い場合
- `config/settings.yaml`でチャンクサイズを調整
- 埋め込みモデルを変更（ada-002 → text-embedding-3-small）
- スコア閾値を調整

## 📊 パフォーマンス

- **Notionデータ収集**: 約30分（182ページ）
- **Google Driveデータ収集**: 約76秒（24ファイル）
- **インデックス構築**: 約2分（830チャンク）
- **質問応答**: 8-12秒/質問

## 🔄 更新とメンテナンス

```bash
# データとインデックスの更新
python main.py update

# テストの実行
python main.py test
```

## 📝 今後の拡張予定

- [ ] Webインターフェース（Streamlit/Gradio）
- [ ] Slack Bot統合
- [ ] リランカーによる精度向上
- [ ] メタデータフィルタリング
- [ ] ユーザー認証と権限管理

## 📄 ライセンス

社内利用限定

## 👥 コントリビューター

- 開発チーム

---

最終更新: 2025年1月20日