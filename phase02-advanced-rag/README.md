# Phase 2: Advanced RAG Implementation

## 📌 概要

Phase 1の基本RAGシステム（検索精度69-88%）を、Advanced RAG技術により90-95%まで向上させる実装フェーズです。

## 🎯 本日の目標（Day 1）

### 実装する3つの技術

1. **HyDE (Hypothetical Document Embeddings)**
   - ユーザーの質問から仮想回答を生成
   - 仮想回答を使って類似文書を検索
   - 曖昧な質問でも適切な文書を発見

2. **RAG-Fusion**
   - 元の質問から複数の関連クエリを生成
   - 並列検索と結果の統合（Reciprocal Rank Fusion）
   - より包括的な情報収集

3. **Reranker**
   - Cohere Rerank APIまたはCross-Encoderモデル
   - 検索結果の再順位付け
   - 最も関連性の高い文書を上位に

## 📊 期待される成果

| 指標 | Phase 1 | Phase 2目標 | 改善率 |
|------|---------|-------------|--------|
| 検索精度 | 69-88% | 90-95% | +15% |
| 応答時間 | 8-12秒 | 5-8秒 | -40% |
| 検索ノイズ | 中 | 低 | - |

## 🗂️ ディレクトリ構造

```
phase02-advanced-rag/
├── README.md              # 本ファイル
├── requirements.txt       # 依存パッケージ
├── .env.example          # 環境変数テンプレート
├── config/               # 設定ファイル
│   └── settings.yaml     # システム設定
├── src/                  # ソースコード
│   ├── __init__.py
│   ├── hyde.py          # HyDE実装
│   ├── rag_fusion.py    # RAG-Fusion実装
│   ├── reranker.py      # Reranker実装
│   ├── advanced_rag_chain.py  # 統合チェーン
│   └── utils.py         # ユーティリティ
├── tests/               # テストコード
│   ├── test_hyde.py
│   ├── test_fusion.py
│   └── test_reranker.py
├── docs/                # ドキュメント
│   └── implementation_notes.md
└── data/                # データ格納
    └── .gitkeep
```

## 🚀 セットアップ

### 1. 環境構築

```bash
# Phase 2ディレクトリに移動
cd phase02-advanced-rag

# Python仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 環境変数の設定

```bash
# .env.exampleをコピー
cp .env.example .env

# 必要なAPIキーを設定
# - OPENAI_API_KEY
# - COHERE_API_KEY (Rerankerで使用)
```

### 3. 実行

```bash
# HyDEのテスト
python src/hyde.py

# RAG-Fusionのテスト
python src/rag_fusion.py

# 統合システムの実行
python src/advanced_rag_chain.py
```

## 💻 実装の詳細

### HyDE (Hypothetical Document Embeddings)

```python
# 基本的な流れ
User Query → LLM → Hypothetical Answer → Embedding → Vector Search
```

**利点**:
- 短い質問でも豊富なコンテキストで検索
- ドメイン知識を活用した検索
- Zero-shotでの精度向上

### RAG-Fusion

```python
# 基本的な流れ
User Query → Query Generation → Parallel Search → RRF → Unified Results
```

**利点**:
- 複数視点からの情報収集
- 検索漏れの削減
- より網羅的な回答生成

### Reranker

```python
# 基本的な流れ
Search Results → Cross-Encoder/Cohere → Reranked Results → LLM
```

**利点**:
- より精密な関連性評価
- ノイズの除去
- 上位結果の品質向上

## 📈 パフォーマンス指標

### 評価基準
- **Precision@K**: 上位K件の精度
- **Recall@K**: 上位K件の再現率
- **MRR (Mean Reciprocal Rank)**: 最初の正解順位の逆数の平均
- **Response Time**: エンドツーエンドの応答時間

### ベンチマーク方法

```python
# テストクエリセットでの評価
python tests/benchmark.py

# 結果は results/benchmark_report.json に出力
```

## 🔧 トラブルシューティング

### Cohere APIエラー
- APIキーの確認
- レート制限の確認（1分あたり100リクエストまで）
- 代替: sentence-transformersのCross-Encoderを使用

### メモリ不足
- バッチサイズを調整（config/settings.yaml）
- より軽量な埋め込みモデルを使用

### 検索精度が低い
- チャンクサイズとオーバーラップを調整
- Rerankerのtop_kパラメータを増やす

## 📚 参考資料

- [HyDE論文](https://arxiv.org/abs/2212.10496)
- [RAG-Fusion解説](https://github.com/Raudaschl/RAG-Fusion)
- [Cohere Rerank API](https://docs.cohere.com/docs/reranking)
- 書籍「LangChainとLangGraphによる RAG・AIエージェント［実践］入門」第6章

## ✅ 実装チェックリスト

- [ ] HyDE基本実装
- [ ] RAG-Fusion基本実装
- [ ] Cohere Reranker統合
- [ ] 3つの技術を統合したチェーン構築
- [ ] Phase 1からのデータ移行
- [ ] パフォーマンステスト
- [ ] ドキュメント整備

## 🎓 学習ポイント

1. **埋め込みの質**: 検索精度の根幹
2. **クエリ拡張**: 情報の網羅性向上
3. **順位付けアルゴリズム**: 最終的な品質決定要因

---

*開始日: 2025年1月21日*  
*目標完了: Day 1終了時*