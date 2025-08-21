# Smart Docs Chat

社内ドキュメントを活用したAI検索・質問応答システムから、高度なAIエージェントシステムへと進化するプロジェクトです。

## 📌 プロジェクト概要

NotionやGoogle Driveに散在するドキュメントを統合的に検索し、AIが適切な回答を生成するRAGシステムから始まり、最終的には自律的に動作するAIエージェントシステムへと発展させます。

### プロジェクトの進化
- 🎯 **Phase 1**: 基本的なRAGシステム
- 🚀 **Phase 2**: Advanced RAGによる検索精度の大幅向上
- 🤖 **Phase 3**: LangGraphによる自律型AIエージェント
- 🌟 **Phase 4**: マルチエージェントシステム

---

## 🗂️ プロジェクト構成

```
smart-docs-chat/
├── phase01-local/            # 基本RAGシステム
├── phase02-advanced-rag/     # Advanced RAG実装
├── phase03-langgraph/        # LangGraphエージェント
├── phase04-multi-agent/      # マルチエージェントシステム
├── docs/                     # プロジェクト共通ドキュメント
│   ├── 01.プロジェクト概要.md
│   ├── 02.技術アーキテクチャ.md
│   ├── 03.開発ガイドライン.md
│   ├── 04.Phase2.Advanced.RAG.md
│   ├── 05.Phase3.LangGraph.md
│   ├── 06.Phase4.Multi.Agent.md
│   └── 07.API仕様.md
└── README.md                # 本ファイル
```

---

## 🚀 開発フェーズ

### Phase 1: 基本RAGシステム構築
**期間**: 2025年1月

**成果**:
- CLIベースのRAGシステム完成
- 検索精度: 0.69-0.88達成（目標0.5を大幅に超過）
- Notionから182ページ、Google Driveから24ファイルの取得成功
- ChromaDBによる高速ベクトル検索実装

**[→ Phase 1の詳細はこちら](./phase01-local/README.md)**

---

### Phase 2: Advanced RAG実装
**期間**: Day 1（1日間）

**目標**:
Phase 1のRAGシステムを大幅に強化し、検索精度90%以上を達成する

**実装内容**:
- **HyDE (Hypothetical Document Embeddings)**
  - ユーザーの質問から仮想的な回答を生成し、それを使って検索
  - 曖昧な質問でも適切な文書を発見可能に
  
- **RAG-Fusion**
  - 複数の検索クエリを生成して並列検索
  - 結果を統合して包括的な回答を生成
  
- **Reranker実装**
  - Cohere RerankerまたはCross-Encoderモデルの統合
  - 検索結果の再順位付けによる精度向上

**期待される成果**:
- 検索精度: 85-95%（Phase 1の69-88%から向上）
- より自然な回答生成
- 専門用語への対応力向上

**技術スタック**:
- 書籍「LangChainとLangGraphによる RAG・AIエージェント［実践］入門」第6章ベース
- Cohere Rerank API / sentence-transformers
- LangChain v0.3.x

---

### Phase 3: LangGraphによるAIエージェント
**期間**: Day 2（1日間）

**目標**:
静的な質問応答システムから、自律的に動作するAIエージェントへの進化

**実装内容**:
- **グラフベースのワークフロー**
  - State、Node、Edgeによる処理フローの定義
  - 条件分岐とループ処理の実装
  
- **ReActパターン実装**
  - Reasoning（推論）とActing（行動）の組み合わせ
  - 外部ツールの呼び出しと結果の活用
  
- **チェックポイント機能**
  - 処理の中断・再開が可能
  - エラー回復機能の実装

**期待される成果**:
- 複雑な質問への段階的な回答生成
- 自律的な情報収集と統合
- エラー耐性の向上

**技術スタック**:
- LangGraph v0.2.x
- 書籍第9章のアーキテクチャをベース
- StateGraph、MessageGraph

---

### Phase 4: 高度なエージェント機能
**期間**: Day 3-4（2日間）

**Day 3 - エージェントの知能化**:
- **セルフリフレクション機能**
  - 生成した回答の自己評価
  - 不十分な場合の自動改善
  
- **要件定義エージェント**
  - ペルソナ生成とインタビュー実施
  - 書籍第10章の簡易版実装
  
- **Plan-and-Executeパターン**
  - タスクの計画と実行の分離
  - より構造化されたアプローチ

**Day 4 - マルチエージェントシステム**:
- **Supervisor型アーキテクチャ**
  - 複数エージェントの協調動作
  - タスクの分配と統合
  
- **統合デモアプリケーション**
  - Streamlitによる対話型UI
  - 全機能を統合したショーケース

**期待される成果**:
- 実用的な要件定義自動生成ツール
- 複数の専門エージェントによる高度な問題解決
- プロダクションレディなシステム

**技術スタック**:
- crewAI / AutoGen の概念を参考
- Streamlit for UI
- 書籍第10章のアーキテクチャ

---

## 🛠️ 技術スタック

### コア技術
- **Python 3.11+**
- **LangChain v0.3.x** - LLMアプリケーション開発フレームワーク
- **LangGraph v0.2.x** - グラフベースのエージェントフレームワーク
- **OpenAI GPT-4o-mini** - 大規模言語モデル
- **ChromaDB** - ベクトルデータベース

### 追加ライブラリ
- **Cohere** - Rerankerモデル
- **sentence-transformers** - Cross-Encoder
- **Streamlit** - Web UI
- **pdfplumber** - PDF処理
- **notion-client** - Notion API

---

## 📖 参考書籍

本プロジェクトは以下の書籍を教科書として進めています：

**「LangChainとLangGraphによる RAG・AIエージェント［実践］入門」**
- 著者: 西見 公宏, 吉田 真吾, 大嶋 勇樹
- 発行: 2024年11月
- 活用章:
  - 第6章: Advanced RAG → Phase 2
  - 第9章: LangGraph → Phase 3  
  - 第10章: 要件定義エージェント → Phase 4

---

## 🎯 プロジェクトの目標

### 短期目標（4日間）
1. **Day 1**: Advanced RAGで検索精度90%超を達成
2. **Day 2**: LangGraphで自律型エージェントを実装
3. **Day 3**: 高度なエージェント機能の実装
4. **Day 4**: マルチエージェントシステムの構築

### 長期的な価値
- **実務への即応用**: AIエージェント開発の実践経験
- **技術力の証明**: GitHubポートフォリオとして公開
- **案件対応力**: 「LangGraphでエージェント開発経験あり」と言える

---

## 📊 成果指標

| フェーズ | 検索精度 | 応答時間 | 自律性 |
|---------|---------|---------|--------|
| Phase 1 | 69-88% | 8-12秒 | なし |
| Phase 2 | 90-95% | 5-8秒 | なし |
| Phase 3 | 90-95% | 3-5秒 | 中 |
| Phase 4 | 95%+ | 2-3秒 | 高 |

---

## 🚀 クイックスタート

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/smart-docs-chat.git
cd smart-docs-chat

# Phase 2の開発を開始
cd phase02-advanced-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 実行
python main.py
```

---

## 📝 開発ルール

### コーディング規約
- **コメント**: 日本語で詳細に記述
- **型ヒント**: 全ての関数に型アノテーション
- **ドキュメント**: 各モジュールにdocstring

### Git運用
- **ブランチ**: phase-X-feature-name
- **コミット**: 日本語で簡潔に
- **タグ**: phase-X-complete

---

## 🔐 セキュリティ

- APIキーは`.env`ファイルで管理（.gitignore対象）
- 社内データは`data/`ディレクトリに隔離
- 本番環境では追加のセキュリティ対策を実施

---

## 📄 ライセンス

本プロジェクトは学習目的のため、MITライセンスで公開予定

---

## 🤝 コントリビューション

現在は個人プロジェクトですが、将来的にはオープンソース化を検討

---

## 📧 連絡先

質問や提案がある場合は、GitHubのIssueでお願いします。

---

*Last Updated: 2025年1月21日*
*Next Milestone: Phase 2完成（Day 1終了時）*