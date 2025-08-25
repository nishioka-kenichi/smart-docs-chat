# Phase 3: LangGraphによるAIエージェント

## 📌 概要

Phase 2のAdvanced RAGシステムを、LangGraphを使用した自律型AIエージェントへと進化させる実装フェーズです。静的な質問応答から、動的な推論と行動を組み合わせたシステムへの移行を実現します。

## 🎯 実装された核心機能

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

## 📊 実装成果

| 指標 | Phase 2比較 | Phase 3実装 | 改善 |
|------|------------|-------------|------|
| 検索精度 | 90-95% | 90-95% | 維持 |
| 応答時間 | 5-8秒 | 3-5秒 | -40% |
| 自律性 | なし | ReActパターン実装 | ✓ |
| エラー耐性 | 低 | チェックポイント機能 | ✓ |

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
│   └── main.py          # メインアプリケーション
├── examples/            # デモ・学習用スクリプト
│   ├── demo_agent_state.py
│   ├── demo_checkpointer.py
│   ├── demo_graph_builder.py
│   ├── demo_main.py
│   ├── demo_react_agent.py
│   └── demo_tools.py
├── tests/               # テストコード
│   ├── test_agent.py
│   ├── test_tools.py
│   └── test_graph.py
├── docs/                # ドキュメント
│   ├── 01.アーキテクチャ詳細.md
│   ├── 02.実装ガイド.md
│   └── 03.学習ガイド.md
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

### 3. 実行

```bash
# 対話モードで起動（デフォルト）
python src/main.py

# 単一クエリを実行
python src/main.py "質問内容"

# チェックポイントから再開
python src/main.py --resume checkpoint_id

# グラフの可視化
python src/main.py --visualize

# 詳細ログを無効化
python src/main.py --quiet "質問内容"
```

## 💻 実装の詳細

### State管理

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # 会話履歴
    current_step: str                     # 現在のステップ
    next_step: Optional[str]             # 次のステップ
    reasoning_steps: List[ReasoningStep] # 推論履歴
    tool_calls: List[ToolCall]           # ツール呼び出し履歴
    context: Dict[str, Any]              # コンテキスト情報
    iteration_count: int                 # イテレーション数
    max_iterations: int                  # 最大イテレーション数
    final_answer: Optional[str]          # 最終回答
    error: Optional[str]                 # エラー情報
    metadata: Dict[str, Any]             # メタデータ
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
- checkpoint: 状態保存

# エッジ定義
- 条件付き分岐（_route_after_reason, _route_after_observe）
- ループバック（observe → reason）
- 終了条件（answer → checkpoint → END）
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
   - Phase 1のChromaDBを活用
   - ベクトル類似度検索

2. **Web検索ツール**（オプション）
   - Tavily APIを使用
   - 最新情報の取得

3. **計算ツール**
   - 数値計算と分析
   - データ処理

4. **ファイル操作ツール**
   - ドキュメント読み込み
   - 結果の保存

## 📈 実装済みの最適化

### ストリーミング
- LLMのストリーミング応答（settings.yamlで設定）
- グラフ実行のストリーミング出力

### チェックポイント圧縮
- gzip圧縮による保存容量削減
- 自動的な古いチェックポイントの削除

## 🧪 テスト

```bash
# 単体テスト
pytest tests/test_agent.py

# 統合テスト
pytest tests/test_graph.py

# 全テスト実行
pytest tests/ -v
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
- [x] State管理の実装
- [x] 基本ツールの実装
- [x] ReActエージェントの実装
- [x] グラフビルダーの構築
- [x] チェックポイント機能
- [x] Phase 1/2との統合
- [x] テストケースの作成
- [x] パフォーマンス最適化
- [x] ドキュメント整備

## 🎓 学習ポイント

1. **グラフ思考**: 処理フローのグラフ表現
2. **状態管理**: 複雑な状態の効率的管理
3. **エージェント設計**: 自律的な問題解決の実装

## 🚀 拡張可能性

### 将来の機能拡張
- **並列ツール実行**: 複数ツールの同時実行
- **応答キャッシング**: LLM応答とツール結果のキャッシュ
- **セルフリフレクション**: 回答の自己評価と改善
- **Plan-and-Executeパターン**: タスクの計画と実行の分離
- **マルチエージェントシステム**: 複数エージェントの協調

