"""
エージェント状態管理モジュール

LangGraphのStateを定義し、エージェントの実行状態を管理します。
ReActパターン（Reasoning + Acting）の実装において、
エージェントの思考・行動・観察のサイクルを追跡します。

主な機能:
1. エージェントの状態（AgentState）の定義
2. ReActパターンの推論ステップの記録
3. ツール呼び出し履歴の管理
4. チェックポイントデータの保存と復元
"""

# 標準ライブラリのインポート
from typing import TypedDict, Annotated, Optional, Any, List, Dict  # 型ヒント用
from datetime import datetime  # タイムスタンプ記録用
import operator  # Annotatedで累積演算を指定するため

# LangChain関連のインポート
from langchain_core.messages import (
    BaseMessage,  # メッセージの基底クラス
    HumanMessage,  # ユーザーからのメッセージ
    AIMessage,  # AIからのメッセージ
    SystemMessage,  # システムメッセージ
    ToolMessage,  # ツール実行結果のメッセージ
)

# Pydanticモデル（データ検証用）
from pydantic import BaseModel, Field  # 型安全なデータモデル定義


class ToolCall(BaseModel):
    """
    ツール呼び出しの記録
    
    エージェントが実行したツール（検索、計算など）の
    呼び出し情報と結果を記録します。
    """
    tool_name: str  # 使用したツールの名前
    arguments: Dict[str, Any]  # ツールに渡した引数
    result: Optional[Any] = None  # ツールの実行結果
    error: Optional[str] = None  # エラーが発生した場合のメッセージ
    timestamp: datetime = Field(default_factory=datetime.now)  # 実行時刻


class ReasoningStep(BaseModel):
    """
    推論ステップの記録
    
    ReActパターンにおける各ステップ（思考→行動→観察）の
    詳細を記録します。
    """
    step_number: int  # ステップ番号（1から開始）
    thought: str  # 思考内容（次に何をすべきかの分析）
    action: Optional[str] = None  # 実行したアクション（ツール名など）
    observation: Optional[str] = None  # アクションの結果から得た観察
    timestamp: datetime = Field(default_factory=datetime.now)  # 記録時刻


class AgentState(TypedDict):
    """
    LangGraphエージェントの状態定義
    
    エージェントの実行中に必要なすべての情報を保持する状態クラス。
    LangGraphのグラフ実行において、ノード間で共有される状態を定義します。
    
    Attributes:
        messages: 会話履歴（メッセージの累積）
        current_step: 現在のステップ名
        next_step: 次のステップ名
        reasoning_steps: ReActパターンの推論履歴
        tool_calls: ツール呼び出し履歴
        context: RAG検索結果などのコンテキスト情報
        iteration_count: 現在のイテレーション数
        max_iterations: 最大イテレーション数
        final_answer: 最終的な回答
        error: エラー情報
        metadata: その他のメタデータ
    """
    # メッセージ履歴（Annotatedで累積を指定）
    messages: Annotated[List[BaseMessage], operator.add]
    
    # ステップ管理
    current_step: str  # 現在実行中のステップ（"start", "reason", "act", "observe", "respond"）
    next_step: Optional[str]  # 次に実行すべきステップ
    
    # ReActパターン用
    reasoning_steps: List[ReasoningStep]  # 思考・行動・観察の履歴
    tool_calls: List[ToolCall]  # 実行したツールの記録
    
    # コンテキスト情報
    context: Dict[str, Any]  # RAG検索結果やその他の参照情報
    
    # 実行制御
    iteration_count: int  # 現在のイテレーション回数
    max_iterations: int  # 最大イテレーション数（無限ループ防止）
    
    # 結果
    final_answer: Optional[str]  # 最終的な回答文
    error: Optional[str]  # エラーが発生した場合のメッセージ
    
    # メタデータ
    metadata: Dict[str, Any]  # 開始時刻、クエリなどの追加情報


class CheckpointData(BaseModel):
    """
    チェックポイントデータ
    
    エージェントの実行状態を保存・復元するためのデータモデル。
    中断した処理の再開や、デバッグのために使用されます。
    """
    checkpoint_id: str  # 一意のチェックポイントID
    state: Dict[str, Any]  # 保存する状態の辞書表現
    timestamp: datetime = Field(default_factory=datetime.now)  # 保存時刻
    step_name: str  # チェックポイントを作成したステップ名
    iteration: int  # イテレーション番号
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 追加のメタデータ


def create_initial_state(query: str, max_iterations: int = 10) -> AgentState:
    """
    初期状態を作成
    
    Args:
        query: ユーザーのクエリ
        max_iterations: 最大イテレーション数
        
    Returns:
        初期化されたAgentState
    """
    return AgentState(
        messages=[HumanMessage(content=query)],
        current_step="start",
        next_step="reason",
        reasoning_steps=[],
        tool_calls=[],
        context={},
        iteration_count=0,
        max_iterations=max_iterations,
        final_answer=None,
        error=None,
        metadata={
            "start_time": datetime.now().isoformat(),
            "query": query
        }
    )


def add_reasoning_step(
    state: AgentState,
    thought: str,
    action: Optional[str] = None,
    observation: Optional[str] = None
) -> AgentState:
    """
    推論ステップを追加
    
    Args:
        state: 現在の状態
        thought: 思考内容
        action: 実行したアクション
        observation: 観察結果
        
    Returns:
        更新された状態
    """
    step = ReasoningStep(
        step_number=len(state["reasoning_steps"]) + 1,
        thought=thought,
        action=action,
        observation=observation
    )
    state["reasoning_steps"].append(step)
    return state


def add_tool_call(
    state: AgentState,
    tool_name: str,
    arguments: Dict[str, Any],
    result: Optional[Any] = None,
    error: Optional[str] = None
) -> AgentState:
    """
    ツール呼び出しを記録
    
    Args:
        state: 現在の状態
        tool_name: ツール名
        arguments: 引数
        result: 実行結果
        error: エラー情報
        
    Returns:
        更新された状態
    """
    call = ToolCall(
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        error=error
    )
    state["tool_calls"].append(call)
    return state


def should_continue(state: AgentState) -> bool:
    """
    処理を継続すべきか判定
    
    Args:
        state: 現在の状態
        
    Returns:
        継続する場合True
    """
    # エラーが発生している場合は停止
    if state.get("error"):
        return False
    
    # 最終回答が生成されている場合は停止
    if state.get("final_answer"):
        return False
    
    # イテレーション数が上限に達している場合は停止
    if state["iteration_count"] >= state["max_iterations"]:
        return False
    
    return True


def format_reasoning_history(state: AgentState) -> str:
    """
    推論履歴をフォーマット
    
    Args:
        state: 現在の状態
        
    Returns:
        フォーマットされた推論履歴
    """
    history = []
    for step in state["reasoning_steps"]:
        history.append(f"Step {step.step_number}:")
        history.append(f"  Thought: {step.thought}")
        if step.action:
            history.append(f"  Action: {step.action}")
        if step.observation:
            history.append(f"  Observation: {step.observation}")
        history.append("")
    
    return "\n".join(history)


def extract_final_state(state: AgentState) -> Dict[str, Any]:
    """
    最終状態から必要な情報を抽出
    
    Args:
        state: 最終状態
        
    Returns:
        抽出された情報
    """
    return {
        "answer": state.get("final_answer", "No answer generated"),
        "reasoning_steps": len(state.get("reasoning_steps", [])),
        "tool_calls": len(state.get("tool_calls", [])),
        "iterations": state.get("iteration_count", 0),
        "error": state.get("error"),
        "context": state.get("context", {}),
        "metadata": state.get("metadata", {})
    }
