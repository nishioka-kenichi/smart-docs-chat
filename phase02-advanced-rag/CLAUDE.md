# Phase 2 Advanced RAG - Claude Code設定

## 重要な設定情報

### 仮想環境
**仮想環境のディレクトリ名: `.venv`**
- `venv`ではなく`.venv`を使用しています
- アクティベート: `source .venv/bin/activate`

### ディレクトリ構成
```
phase02-advanced-rag/
├── .venv/           # 仮想環境（.venvに注意）
├── src/             # ソースコード
├── docs/            # ドキュメント
├── config/          # 設定ファイル
├── .env             # 環境変数
└── requirements.txt # 依存パッケージ
```

### テスト実行
```bash
# 仮想環境のアクティベート
source .venv/bin/activate

# セットアップテスト
python test_setup.py

# Advanced RAGチェーンのテスト
cd src
python advanced_rag_chain.py
```

### 環境変数
- OPENAI_API_KEY: 必須
- COHERE_API_KEY: オプション（Reranker用）
- CHROMADB_PATH: ../phase01-local/data/chromadb

### 注意事項
- Phase 1のChromaDBを直接参照しているため、Phase 1の環境が必要
- オプションパッケージ（cohere, sentence-transformers）は未インストール