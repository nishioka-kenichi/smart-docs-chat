# Phase 3: LangGraphによるAIエージェント

## 📌 概要

Phase 2のAdvanced RAGシステムを、LangGraphを使用した自律型AIエージェントへと進化させる実装フェーズです。静的な質問応答から、動的な推論と行動を組み合わせたシステムへの移行を実現します。

## 🎯 本日の目標（Day 2）

### 実装する3つの核心機能

1. **グラフベースのワークフロー**
   - State、Node、Edgeによる処理フローの定義
   - 条件分岐とループ処理の実装
   - 動的な処理経路の決定

2. **ReActパターン実装**
   - Reasoning（推論）とActing（行動）の組み合わせ
   - 外部ツールの呼び出しと結果の活用
   - 思考の連鎖による問題解決

3. **チェックポイント機能**
   - 処理の中断・再開が可能
   - エラー回復機能の実装
   - 状態の永続化と復元

## 📊 期待される成果

| 指標 | Phase 2 | Phase 3目標 | 改善 |
|------|---------|-------------|------|
| 検索精度 | 90-95% | 90-95% | 維持 |
| 応答時間 | 5-8秒 | 3-5秒 | -40% |
| 自律性 | なし | 中 | ↑↑ |
| エラー耐性 | 低 | 高 | ↑↑↑ |

## 🗂️ ディレクトリ構造

```
phase03-langgraph/
├── README.md              # 本ファイル
├── requirements.txt       # 依存パッケージ
├── .env.example          # 環境変数テンプレート
├── config/               # 設定ファイル
│   └── settings.yaml     # システム設定
├── src/                  # ソースコード
│   ├── __init__.py
│   ├── agent_state.py    # State管理
│   ├── tools.py         # ツール定義
│   ├── react_agent.py   # ReActエージェント
│   ├── graph_builder.py # グラフ構築
│   ├── checkpointer.py  # チェックポイント管理
│   ├── main.py          # メインアプリケーション
│   └── utils.py         # ユーティリティ
├── tests/               # テストコード
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_graph.py
├── docs/                # ドキュメント
│   └── architecture.md  # アーキテクチャ説明
├── notebooks/           # 実験用ノートブック
│   └── demo.ipynb
└── data/                # データ格納
    ├── checkpoints/     # チェックポイント保存
    └── logs/           # 実行ログ
```

## 🚀 セットアップ

### 1. 環境構築

```bash
# Phase 3ディレクトリに移動
cd phase03-langgraph

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
# - TAVILY_API_KEY (Web検索用、オプション)
```

### 3. Phase 1/2のデータベース接続

```bash
# Phase 1のChromaDBへのパスを設定
export CHROMADB_PATH="../phase01-local/data/chromadb"
```

### 4. 実行

```bash
# エージェントのテスト実行
python src/main.py

# 対話モード
python src/main.py --interactive

# チェックポイントから再開
python src/main.py --resume checkpoint_id
```

## 💻 実装の詳細

### State管理

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next_step: str
    context: dict
    tools_output: list
    final_answer: Optional[str]
```

### ReActパターン

```python
# 基本的な流れ
User Query → Reasoning → Tool Selection → Action → Observation → Loop/Answer
```

**特徴**:
- 思考プロセスの可視化
- 段階的な問題解決
- ツール使用の最適化

### グラフ構造

```python
# ノード定義
- start: 初期処理
- reason: 推論ノード
- act: 行動ノード
- observe: 観察ノード
- answer: 回答生成

# エッジ定義
- 条件付き分岐
- ループバック
- 終了条件
```

### チェックポイント

```python
# 自動保存ポイント
- 各ノード実行後
- エラー発生時
- 明示的な保存指示

# 復元機能
- 任意のポイントから再開
- 状態の完全復元
- エラー後の継続実行
```

## 🛠️ ツール実装

### 基本ツール

1. **RAG検索ツール**
   - Phase 1/2のChromaDBを活用
   - Advanced RAG技術の統合

2. **Web検索ツール**（オプション）
   - Tavily APIを使用
   - 最新情報の取得

3. **計算ツール**
   - 数値計算と分析
   - データ処理

4. **ファイル操作ツール**
   - ドキュメント読み込み
   - 結果の保存

## 📈 パフォーマンス最適化

### 並列処理
- ツール呼び出しの並列実行
- 非同期処理によるレスポンス向上

### キャッシング
- LLM応答のキャッシュ
- ツール結果の再利用

### ストリーミング
- 段階的な応答生成
- リアルタイムフィードバック

## 🧪 テスト

```bash
# 単体テスト
pytest tests/test_agent.py

# 統合テスト
pytest tests/test_graph.py

# パフォーマンステスト
python tests/benchmark.py
```

## 📊 評価指標

### エージェント性能
- **Task Completion Rate**: タスク完了率
- **Average Steps**: 平均ステップ数
- **Tool Usage Efficiency**: ツール使用効率
- **Error Recovery Rate**: エラー回復率

### システム性能
- **Response Time**: 応答時間
- **Memory Usage**: メモリ使用量
- **Checkpoint Size**: チェックポイントサイズ

## 🔧 トラブルシューティング

### LangGraph関連エラー
- グラフの循環参照を確認
- ノードの入出力型を確認
- エッジ条件の妥当性を確認

### メモリリーク
- チェックポイントの定期的なクリーンアップ
- 大きなStateオブジェクトの最適化

### パフォーマンス問題
- ノードの並列化を検討
- キャッシュの活用
- 不要なLLM呼び出しの削減

## 📚 参考資料

- [LangGraph公式ドキュメント](https://github.com/langchain-ai/langgraph)
- [ReActパターン論文](https://arxiv.org/abs/2210.03629)
- 書籍「LangChainとLangGraphによる RAG・AIエージェント［実践］入門」第9章

## ✅ 実装チェックリスト

- [x] ディレクトリ構造作成
- [ ] State管理の実装
- [ ] 基本ツールの実装
- [ ] ReActエージェントの実装
- [ ] グラフビルダーの構築
- [ ] チェックポイント機能
- [ ] Phase 1/2との統合
- [ ] テストケースの作成
- [ ] パフォーマンス最適化
- [ ] ドキュメント整備

## 🎓 学習ポイント

1. **グラフ思考**: 処理フローのグラフ表現
2. **状態管理**: 複雑な状態の効率的管理
3. **エージェント設計**: 自律的な問題解決の実装

## 🚀 次のステップ（Phase 4予告）

- セルフリフレクション機能
- Plan-and-Executeパターン
- マルチエージェントシステム

---

*開始日: 2025年1月22日*  
*目標完了: Day 2終了時*