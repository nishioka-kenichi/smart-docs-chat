"""
LangGraphグラフ構築モジュール

LangGraphを使用してエージェントの処理フローをグラフとして定義します。
ノード（処理ステップ）とエッジ（遷移条件）を組み合わせて、
ReActパターンのワークフローを構築します。

主な機能:
1. StateGraphを使用したワークフロー定義
2. 条件付きエッジによる動的なルーティング
3. チェックポイント機能による中断・再開
4. グラフの可視化
"""

# 標準ライブラリのインポート
from typing import Dict, Any, Optional, Literal  # 型ヒント用

# LangGraph関連のインポート
from langgraph.graph import StateGraph, END  # グラフ構築と終了ノード
from langgraph.checkpoint.memory import MemorySaver  # チェックポイント機能
# from langgraph.prebuilt import ToolExecutor  # ツール実行ユーティリティ（未使用）

# LangChain関連のインポート
from langchain_openai import ChatOpenAI  # OpenAIのチャットモデル

# ローカルモジュールのインポート
from agent_state import AgentState, should_continue  # エージェント状態と継続判定
from react_agent import ReActAgent  # ReActエージェント
from tools import ToolsManager  # ツール管理
from checkpointer import CheckpointManager  # チェックポイント管理


class GraphBuilder:
    """
    LangGraphを使用したエージェントグラフ構築クラス
    
    ReActエージェントのワークフローをグラフとして定義し、
    各ノード間の遷移条件を管理します。
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        tools_manager: Optional[ToolsManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        config_path: str = "./config/settings.yaml",
        verbose: bool = True
    ):
        """
        グラフビルダーの初期化
        
        ReActエージェントを初期化し、StateGraphを構築します。
        
        Args:
            llm: 使用するLLMインスタンス（Noneの場合はデフォルト設定）
            tools_manager: ツール管理インスタンス（Noneの場合は新規作成）
            checkpoint_manager: チェックポイント管理インスタンス（Noneの場合は新規作成）
            config_path: 設定ファイルパス
            verbose: 詳細ログ出力の有効/無効
        """
        self.verbose = verbose
        
        # ReActエージェントの初期化
        self.agent = ReActAgent(
            llm=llm,
            tools_manager=tools_manager,
            config_path=config_path,
            verbose=verbose
        )
        
        # チェックポイント管理
        self.checkpoint_manager = checkpoint_manager
        if checkpoint_manager is None:
            from checkpointer import CheckpointManager
            self.checkpoint_manager = CheckpointManager()
        
        # グラフの構築
        self.graph = self._build_graph()
        
        # メモリセーバー（LangGraphのチェックポイント機能）
        self.memory = MemorySaver()
        
        # コンパイル済みグラフ
        self.app = None
    
    def _build_graph(self) -> StateGraph:
        """
        エージェントグラフを構築
        
        ReActパターンのワークフローをStateGraphとして定義します。
        ノード： start → reason → act → observe → answer → checkpoint
        
        Returns:
            構築されたStateGraphオブジェクト
        """
        if self.verbose:
            print("📊 Building agent graph...")
        
        # StateGraphの初期化
        workflow = StateGraph(AgentState)
        
        # ノードの追加
        workflow.add_node("start", self._start_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("answer", self._answer_node)
        workflow.add_node("checkpoint", self._checkpoint_node)
        
        # エッジの追加（条件付き遷移）
        workflow.set_entry_point("start")
        
        # startノードからの遷移
        workflow.add_edge("start", "reason")
        
        # reasonノードからの条件付き遷移
        workflow.add_conditional_edges(
            "reason",
            self._route_after_reason,
            {
                "act": "act",
                "answer": "answer",
                "checkpoint": "checkpoint",
                "end": END
            }
        )
        
        # actノードからの遷移
        workflow.add_edge("act", "observe")
        
        # observeノードからの条件付き遷移
        workflow.add_conditional_edges(
            "observe",
            self._route_after_observe,
            {
                "reason": "reason",
                "answer": "answer",
                "checkpoint": "checkpoint",
                "end": END
            }
        )
        
        # answerノードからの遷移
        workflow.add_edge("answer", "checkpoint")
        
        # checkpointノードからの遷移
        workflow.add_edge("checkpoint", END)
        
        return workflow
    
    def _start_node(self, state: AgentState) -> AgentState:
        """
        開始ノード
        
        ワークフローの開始点。メタデータの初期化や
        チェックポイントからの復元処理を行います。
        
        Args:
            state: 現在のエージェント状態
            
        Returns:
            初期化された状態
        """
        if self.verbose:
            print("\n🎯 Start Node")
        
        # メタデータの初期化
        if "start_time" not in state.get("metadata", {}):
            from datetime import datetime
            state["metadata"]["start_time"] = datetime.now().isoformat()
        
        # チェックポイントから復元する場合の処理
        if state.get("metadata", {}).get("resumed_from_checkpoint"):
            if self.verbose:
                print("  📂 Resumed from checkpoint")
        
        return state
    
    def _reason_node(self, state: AgentState) -> AgentState:
        """
        推論ノード
        
        ReActエージェントの推論ステップを実行します。
        
        Args:
            state: 現在のエージェント状態
            
        Returns:
            推論結果で更新された状態
        """
        return self.agent.reason(state)  # ReActエージェントの推論メソッドを呼び出し
    
    def _act_node(self, state: AgentState) -> AgentState:
        """
        アクションノード
        
        ReActエージェントのアクション（ツール実行）ステップを実行します。
        
        Args:
            state: 現在のエージェント状態
            
        Returns:
            ツール実行結果で更新された状態
        """
        return self.agent.act(state)  # ReActエージェントのアクションメソッドを呼び出し
    
    def _observe_node(self, state: AgentState) -> AgentState:
        """
        観察ノード
        
        ReActエージェントの観察ステップを実行します。
        
        Args:
            state: 現在のエージェント状態
            
        Returns:
            次のステップが決定された状態
        """
        return self.agent.observe(state)  # ReActエージェントの観察メソッドを呼び出し
    
    def _answer_node(self, state: AgentState) -> AgentState:
        """
        回答生成ノード
        
        ReActエージェントの最終回答生成ステップを実行します。
        
        Args:
            state: 現在のエージェント状態
            
        Returns:
            最終回答が設定された状態
        """
        return self.agent.answer(state)  # ReActエージェントの回答生成メソッドを呼び出し
    
    def _checkpoint_node(self, state: AgentState) -> AgentState:
        """
        チェックポイント保存ノード
        
        現在の状態をチェックポイントとして保存し、
        後で再開できるようにします。
        
        Args:
            state: 現在のエージェント状態
            
        Returns:
            チェックポイントIDが記録された状態
        """
        if self.verbose:
            print("\n💾 Checkpoint Node")
        
        try:
            # チェックポイントを保存
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                state=state,
                step_name=state.get("current_step", "unknown"),
                iteration=state.get("iteration_count", 0)
            )
            
            if self.verbose:
                print(f"  ✅ Checkpoint saved: {checkpoint_id}")
            
            state["metadata"]["last_checkpoint_id"] = checkpoint_id
            
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Checkpoint save failed: {e}")
        
        return state
    
    def _route_after_reason(self, state: AgentState) -> Literal["act", "answer", "checkpoint", "end"]:
        """
        推論後のルーティング
        
        推論ステップの結果に基づいて、
        次に実行すべきノードを決定します。
        
        Args:
            state: 現在のエージェント状態
            
        Returns:
            次のノード名（"act", "answer", "checkpoint", "end"のいずれか）
        """
        # エラーがある場合
        if state.get("error"):
            return "answer"
        
        # 次のステップが指定されている場合
        next_step = state.get("next_step")
        if next_step == "act":
            return "act"
        elif next_step == "answer":
            return "answer"
        
        # 継続判定
        if not should_continue(state):
            return "answer"
        
        # デフォルトはチェックポイント
        return "checkpoint"
    
    def _route_after_observe(self, state: AgentState) -> Literal["reason", "answer", "checkpoint", "end"]:
        """
        観察後のルーティング
        
        観察ステップの結果に基づいて、
        次に実行すべきノードを決定します。
        
        Args:
            state: 現在のエージェント状態
            
        Returns:
            次のノード名（"reason", "answer", "checkpoint", "end"のいずれか）
        """
        # エラーがある場合
        if state.get("error"):
            return "answer"
        
        # 次のステップが指定されている場合
        next_step = state.get("next_step")
        if next_step == "reason":
            return "reason"
        elif next_step == "answer":
            return "answer"
        
        # 継続判定
        if not should_continue(state):
            return "answer"
        
        # 定期的なチェックポイント
        if state.get("iteration_count", 0) % 5 == 0:
            return "checkpoint"
        
        return "reason"
    
    def compile(self) -> Any:
        """
        グラフをコンパイル
        
        StateGraphを実行可能な形式にコンパイルし、
        チェックポイント機能を有効化します。
        
        Returns:
            コンパイル済みグラフアプリケーション
        """
        if self.verbose:
            print("🔨 Compiling graph...")
        
        self.app = self.graph.compile(checkpointer=self.memory)
        
        if self.verbose:
            print("✅ Graph compiled successfully")
        
        return self.app
    
    def run(
        self,
        query: str,
        max_iterations: int = 10,
        thread_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        グラフを実行
        
        ユーザーのクエリを受け取り、LangGraphのワークフローを
        実行して回答を生成します。チェックポイントからの再開も可能です。
        
        Args:
            query: ユーザーの質問文字列
            max_iterations: 最大イテレーション数（無限ループ防止）
            thread_id: スレッドID（会話の継続用）
            checkpoint_id: 再開するチェックポイントID
            
        Returns:
            実行結果を含む辞書（回答、ステップ数、ツール呼び出し回数等）
        """
        if not self.app:
            self.compile()
        
        # 初期状態の作成または復元
        if checkpoint_id:
            # チェックポイントから復元
            if self.verbose:
                print(f"📂 Resuming from checkpoint: {checkpoint_id}")
            
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_id)
            if checkpoint_data:
                initial_state = checkpoint_data.state
                initial_state["metadata"]["resumed_from_checkpoint"] = checkpoint_id
            else:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
        else:
            # 新規作成
            from agent_state import create_initial_state
            initial_state = create_initial_state(query, max_iterations)
        
        # 設定
        config = {
            "configurable": {
                "thread_id": thread_id or "default"
            }
        }
        
        if self.verbose:
            print(f"\n🚀 Running graph with query: {query}")
            print(f"   Thread ID: {config['configurable']['thread_id']}")
            print(f"   Max iterations: {max_iterations}")
        
        try:
            # グラフを実行
            final_state = None
            all_outputs = []
            for output in self.app.stream(initial_state, config):
                all_outputs.append(output)
                
                # ストリーミング出力
                if self.verbose:
                    for key, value in output.items():
                        if key != "__end__":
                            print(f"   📍 Node: {key}")
            
            # 最終状態を抽出 - 最後の出力から取得
            if all_outputs:
                # 最後の出力を確認
                last_output = all_outputs[-1]
                
                # "__end__"キーがあればそれを使用、なければ最後のノードの出力を使用
                if "__end__" in last_output:
                    result_state = last_output["__end__"]
                else:
                    # 最後のノード（通常はcheckpoint）の出力を使用
                    node_name = list(last_output.keys())[0] if last_output else None
                    result_state = last_output.get(node_name, initial_state) if node_name else initial_state
            else:
                result_state = initial_state
            
            # 結果を返す
            from agent_state import extract_final_state
            return extract_final_state(result_state)
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Graph execution error: {e}")
            
            return {
                "answer": f"エラーが発生しました: {str(e)}",
                "error": str(e),
                "reasoning_steps": 0,
                "tool_calls": 0,
                "iterations": 0
            }
    
    def visualize(self, output_path: str = "./data/graph.png"):
        """
        グラフを可視化
        
        LangGraphのワークフローをMermaid形式で
        画像ファイルとして出力します。
        
        Args:
            output_path: 出力ファイルパス（PNG形式）
        """
        try:
            if not self.app:
                self.compile()
            
            # グラフを画像として保存
            from IPython.display import Image
            import matplotlib.pyplot as plt
            
            graph_image = self.app.get_graph().draw_mermaid_png()
            
            with open(output_path, "wb") as f:
                f.write(graph_image)
            
            if self.verbose:
                print(f"📊 Graph visualization saved to {output_path}")
            
            return graph_image
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Visualization error: {e}")
            return None
