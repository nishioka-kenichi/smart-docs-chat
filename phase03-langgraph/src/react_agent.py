"""
ReActパターンエージェント実装モジュール

Reasoning（推論）とActing（行動）を組み合わせたReActパターンを実装します。
エージェントは以下のサイクルを繰り返します：
1. Thought（思考）: 現在の状況を分析し、次のアクションを決定
2. Action（行動）: 必要なツールを実行して情報を収集
3. Observation（観察）: ツールの結果を観察し、次のステップを決定

主な機能:
1. 自律的な意思決定とツール選択
2. 段階的な問題解決
3. エラー処理とリトライ
4. 詳細な推論履歴の記録
"""

# 標準ライブラリのインポート
import json  # JSON形式のデータ処理
from typing import Dict, Any, List, Optional, Union  # 型ヒント用
from datetime import datetime  # タイムスタンプ記録用
from textwrap import dedent  # 複数行文字列のインデント調整

# LangChain関連のインポート
from langchain_core.messages import (
    HumanMessage,  # ユーザーからのメッセージ
    AIMessage,  # AIからのメッセージ
    SystemMessage,  # システムメッセージ
    ToolMessage,  # ツール実行結果のメッセージ
)
from langchain_openai import ChatOpenAI  # OpenAIのチャットモデル
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # プロンプトテンプレート
from langchain_core.output_parsers import JsonOutputParser  # JSON出力パーサー

# Pydanticモデル（データ検証用）
from pydantic import BaseModel, Field  # 型安全なデータモデル定義

# ローカルモジュールのインポート
from agent_state import (
    AgentState,  # エージェント状態クラス
    add_reasoning_step,  # 推論ステップ追加関数
    add_tool_call,  # ツール呼び出し記録関数
    format_reasoning_history,  # 推論履歴フォーマット関数
)
from tools import ToolsManager  # ツール管理クラス


class ReActThought(BaseModel):
    """
    ReActの思考プロセスモデル

    LLMが生成する思考プロセスを構造化し、
    次のアクションを決定するためのデータモデルです。
    """

    reasoning: str = Field(description="現在の状況と次に何をすべきかの推論")
    action_needed: bool = Field(description="ツールを使用する必要があるか")
    action: Optional[str] = Field(description="実行するアクション（ツール名）", default=None)
    action_input: Optional[Dict[str, Any]] = Field(description="アクションの入力パラメータ", default=None)
    is_final_answer: bool = Field(description="最終回答を生成する準備ができているか", default=False)
    final_answer: Optional[str] = Field(description="最終回答の内容", default=None)


class ReActAgent:
    """
    ReActパターンを実装したエージェント

    思考→行動→観察のサイクルを繰り返し、
    自律的に問題を解決するエージェントです。
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        tools_manager: Optional[ToolsManager] = None,
        config_path: str = "./config/settings.yaml",
        verbose: bool = True,
    ):
        """
        ReActエージェントの初期化

        LLM、ツールマネージャー、プロンプトテンプレートを設定します。

        Args:
            llm: 使用するLLMインスタンス（Noneの場合はデフォルト設定）
            tools_manager: ツール管理インスタンス（Noneの場合は新規作成）
            config_path: 設定ファイルパス
            verbose: 詳細ログ出力の有効/無効
        """
        self.verbose = verbose

        # LLMの初期化
        if llm is None:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)
        else:
            self.llm = llm

        # ツールマネージャーの初期化
        if tools_manager is None:
            self.tools_manager = ToolsManager(config_path)
        else:
            self.tools_manager = tools_manager

        # プロンプトテンプレートの設定
        self._setup_prompts()

        # 出力パーサーの設定
        self.output_parser = JsonOutputParser(pydantic_object=ReActThought)

    def _setup_prompts(self):
        """
        プロンプトテンプレートを設定

        ReActパターン用のシステムプロンプトと
        各ステップ用のテンプレートを定義します。
        """

        # システムプロンプト（ReActパターンの説明）
        self.system_prompt = dedent(
            """
            あなたはReActパターンで動作する高度なAIアシスタントです。
            ユーザーの質問に答えるために、以下のプロセスに従ってください：

            1. **Reasoning（推論）**: 現在の状況を分析し、次に何をすべきか考える
            2. **Acting（行動）**: 必要に応じてツールを使用して情報を収集する
            3. **Observation（観察）**: ツールの結果を観察し、次のステップを決定する
            4. このプロセスを繰り返し、十分な情報が集まったら最終回答を生成する

            利用可能なツール:
            {tools_description}

            重要な注意点:
            - 一度に1つのツールのみ使用する
            - ツールの結果を観察してから次のアクションを決定する
            - 最終回答は収集した情報に基づいて包括的に作成する
            - 推論プロセスを明確に説明する
            - **必ず指定されたJSON形式で応答すること**

            出力は必ず以下のJSON形式で返してください:
            {format_instructions}
            
            JSON以外の形式での応答は許可されません。
        """
        ).strip()

        # ReActステップのプロンプト
        self.react_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "これまでの推論履歴:\n{reasoning_history}\n\n次のステップを決定してください。"),
            ]
        )

        # 最終回答生成のプロンプト
        self.final_answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは親切で知識豊富なAIアシスタントです。収集した情報を基に、ユーザーの質問に対する包括的な回答を生成してください。",
                ),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "human",
                    "推論履歴:\n{reasoning_history}\n\n収集した情報を基に、最終的な回答を生成してください。",
                ),
            ]
        )

    def reason(self, state: AgentState) -> AgentState:
        """
        推論ステップを実行

        現在の状態を分析し、次に実行すべきアクションを決定します。
        LLMによる思考プロセスを実行し、結果を構造化して返します。

        Args:
            state: 現在のエージェント状態

        Returns:
            推論結果で更新された状態
        """
        if self.verbose:
            print(f"\n🧠 Reasoning Step {state['iteration_count'] + 1}")

        # ツールの説明を生成
        tools_description = self._format_tools_description()

        # 推論履歴をフォーマット
        reasoning_history = format_reasoning_history(state)

        # プロンプトを構築
        prompt = self.react_prompt.format_messages(
            tools_description=tools_description,
            format_instructions=self.output_parser.get_format_instructions(),
            messages=state["messages"],
            reasoning_history=reasoning_history,
        )

        try:
            # LLMに推論を依頼
            response = self.llm.invoke(prompt)

            # レスポンスをパース
            try:
                thought = self.output_parser.parse(response.content)
            except Exception as parse_error:
                # JSON形式でない場合のフォールバック処理
                if self.verbose:
                    print(f"  ⚠️ JSON解析エラー、フォールバック処理を実行: {parse_error}")
                
                # レスポンス内容から最終回答を生成すべきか判定
                content_lower = response.content.lower()
                if "最終回答" in response.content or "final answer" in content_lower:
                    # 最終回答として処理
                    thought = ReActThought(
                        reasoning="収集した情報から最終回答を生成します。",
                        action_needed=False,
                        is_final_answer=True,
                        final_answer=None  # answerステップで生成
                    )
                else:
                    # エラーとして処理
                    raise parse_error
            
            # thoughtが辞書の場合、ReActThoughtオブジェクトに変換
            if isinstance(thought, dict):
                # action_inputが文字列の場合、辞書に変換
                if "action_input" in thought and isinstance(thought["action_input"], str):
                    action_name = thought.get("action", "")
                    input_str = thought["action_input"]
                    
                    # 各ツールに応じた適切なキーで辞書化
                    if action_name == "calculator":
                        thought["action_input"] = {"expression": input_str}
                    elif action_name == "rag_search":
                        thought["action_input"] = {"query": input_str}
                    elif action_name == "web_search":
                        thought["action_input"] = {"query": input_str}
                    elif action_name == "read_file":
                        thought["action_input"] = {"file_path": input_str}
                    elif action_name == "write_file":
                        # write_fileは複雑なので、文字列の場合はエラーにする
                        thought["action_input"] = {"file_path": "", "content": input_str}
                    else:
                        # その他のツールは汎用的に処理
                        thought["action_input"] = {"input": input_str}
                
                # final_answerが辞書の場合、文字列に変換
                if "final_answer" in thought and isinstance(thought["final_answer"], dict):
                    import json
                    thought["final_answer"] = json.dumps(thought["final_answer"], ensure_ascii=False)
                
                thought = ReActThought(**thought)

            if self.verbose:
                print(f"  思考: {thought.reasoning}")
                if thought.action_needed:
                    print(f"  アクション: {thought.action} with {thought.action_input}")

            # 状態を更新
            state = add_reasoning_step(
                state, thought=thought.reasoning, action=thought.action if thought.action_needed else None
            )

            # 次のステップを決定
            if thought.is_final_answer:
                state["next_step"] = "answer"
                state["final_answer"] = thought.final_answer
            elif thought.action_needed:
                state["next_step"] = "act"
                state["metadata"]["pending_action"] = {"tool": thought.action, "input": thought.action_input}
            else:
                state["next_step"] = "observe"

            state["iteration_count"] += 1

        except Exception as e:
            if self.verbose:
                print(f"  ❌ 推論エラー: {e}")
            state["error"] = f"推論エラー: {str(e)}"
            state["next_step"] = None

        return state

    def act(self, state: AgentState) -> AgentState:
        """
        アクション（ツール実行）ステップ

        推論ステップで決定されたツールを実行し、
        結果を状態に記録します。

        Args:
            state: 現在のエージェント状態

        Returns:
            ツール実行結果で更新された状態
        """
        if self.verbose:
            print(f"\n🔧 Action Step")

        # 実行するアクションを取得
        pending_action = state["metadata"].get("pending_action")
        if not pending_action:
            state["error"] = "実行するアクションが指定されていません"
            state["next_step"] = None
            return state

        tool_name = pending_action["tool"]
        tool_input = pending_action["input"] or {}

        try:
            # ツールを取得
            tool = None
            for t in self.tools_manager.get_tools():
                if t.name == tool_name:
                    tool = t
                    break

            if not tool:
                raise ValueError(f"ツール '{tool_name}' が見つかりません")

            # ツールを実行
            if self.verbose:
                print(f"  実行中: {tool_name}")

            result = tool.invoke(tool_input)

            if self.verbose:
                print(f"  結果: {result[:200]}..." if len(str(result)) > 200 else f"  結果: {result}")

            # 結果を記録
            state = add_tool_call(state, tool_name=tool_name, arguments=tool_input, result=result)

            # 観察結果を推論履歴に追加
            if state["reasoning_steps"]:
                state["reasoning_steps"][-1].observation = str(result)

            # AIメッセージとツールメッセージを適切に追加
            # まずツール呼び出しを含むAIメッセージを追加
            from langchain_core.messages import AIMessage
            state["messages"].append(
                AIMessage(
                    content=f"ツール {tool_name} を使用します。",
                    tool_calls=[
                        {
                            "id": f"{tool_name}_{state['iteration_count']}",
                            "name": tool_name,
                            "args": tool_input
                        }
                    ]
                )
            )
            
            # それに対応するツールメッセージを追加
            state["messages"].append(
                ToolMessage(
                    content=str(result), 
                    tool_call_id=f"{tool_name}_{state['iteration_count']}"
                )
            )

            # アクションをクリア
            state["metadata"].pop("pending_action", None)
            state["next_step"] = "observe"

        except Exception as e:
            if self.verbose:
                print(f"  ❌ アクションエラー: {e}")

            state = add_tool_call(state, tool_name=tool_name, arguments=tool_input, error=str(e))

            state["error"] = f"ツール実行エラー: {str(e)}"
            state["next_step"] = None

        return state

    def observe(self, state: AgentState) -> AgentState:
        """
        観察ステップ（次のアクションを決定）

        ツール実行結果を観察し、次のステップを決定します。
        最大イテレーション数のチェックも行います。

        Args:
            state: 現在のエージェント状態

        Returns:
            次のステップが設定された状態
        """
        if self.verbose:
            print(f"\n👁️ Observation Step")

        # 最大イテレーションチェック
        if state["iteration_count"] >= state["max_iterations"]:
            if self.verbose:
                print("  ⚠️ 最大イテレーション数に到達")
            state["next_step"] = "answer"
            return state

        # エラーチェック
        if state.get("error"):
            state["next_step"] = "answer"
            return state

        # 次は推論ステップへ
        state["next_step"] = "reason"
        return state

    def answer(self, state: AgentState) -> AgentState:
        """
        最終回答を生成

        収集した情報と推論履歴を基に、
        ユーザーの質問に対する包括的な回答を生成します。

        Args:
            state: 現在のエージェント状態

        Returns:
            最終回答が設定された状態
        """
        if self.verbose:
            print(f"\n✨ Generating Final Answer")

        # すでに最終回答がある場合はそれを使用
        if state.get("final_answer"):
            if self.verbose:
                print(f"  回答: {state['final_answer']}")
            return state

        # 推論履歴をフォーマット
        reasoning_history = format_reasoning_history(state)

        # エラーがある場合
        if state.get("error"):
            state["final_answer"] = f"申し訳ございません。処理中にエラーが発生しました: {state['error']}"
            return state

        try:
            # 最終回答を生成
            prompt = self.final_answer_prompt.format_messages(
                messages=state["messages"], reasoning_history=reasoning_history
            )

            response = self.llm.invoke(prompt)
            state["final_answer"] = response.content

            if self.verbose:
                print(f"  回答: {state['final_answer'][:500]}...")

            # 回答をメッセージに追加
            state["messages"].append(AIMessage(content=state["final_answer"]))
            state["next_step"] = None  # 終了

        except Exception as e:
            if self.verbose:
                print(f"  ❌ 回答生成エラー: {e}")
            state["error"] = f"回答生成エラー: {str(e)}"
            state["final_answer"] = "申し訳ございません。回答の生成に失敗しました。"
            state["next_step"] = None

        return state

    def _format_tools_description(self) -> str:
        """
        ツールの説明をフォーマット

        利用可能なツールのリストをLLMが理解しやすい形式にフォーマットします。

        Returns:
            フォーマットされたツール説明文字列
        """
        descriptions = []
        for tool in self.tools_manager.get_tools():
            descriptions.append(f"- {tool.name}: {tool.description}")

        return "\n".join(descriptions)

    def run(self, query: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        エージェントを実行

        ユーザーのクエリを受け取り、ReActパターンの
        サイクルを繰り返して回答を生成します。

        Args:
            query: ユーザーの質問文字列
            max_iterations: 最大イテレーション数（無限ループ防止）

        Returns:
            実行結果を含む辞書（回答、ステップ数、ツール呼び出し回数等）
        """
        if self.verbose:
            print(f"\n🚀 ReAct Agent Started")
            print(f"Query: {query}")
            print(f"Max iterations: {max_iterations}")

        # 初期状態を作成
        from agent_state import create_initial_state

        state = create_initial_state(query, max_iterations)

        # エージェントループ
        while state.get("next_step"):
            current_step = state["next_step"]

            if current_step == "reason":
                state = self.reason(state)
            elif current_step == "act":
                state = self.act(state)
            elif current_step == "observe":
                state = self.observe(state)
            elif current_step == "answer":
                state = self.answer(state)
            else:
                break

            # エラーチェック
            if state.get("error") and not state.get("final_answer"):
                state = self.answer(state)
                break

        if self.verbose:
            print(f"\n✅ ReAct Agent Completed")
            print(f"Total iterations: {state['iteration_count']}")
            print(f"Tool calls: {len(state['tool_calls'])}")

        # 結果を返す
        from agent_state import extract_final_state

        return extract_final_state(state)
