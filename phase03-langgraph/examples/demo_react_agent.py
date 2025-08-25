#!/usr/bin/env python
"""
ReActエージェントのデモンストレーション

このスクリプトは、ReActパターンを使用したエージェントの動作を
実際に確認するためのデモンストレーションです。

実行方法:
    python examples/demo_react_agent.py

デバッグ方法:
    1. VSCodeでこのファイルを開く
    2. ブレークポイントを設定（例：reason関数内）
    3. F5キーでデバッグ実行
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_state import create_initial_state, add_reasoning_step
from react_agent import ReActAgent, ReActThought


def main():
    """メイン処理"""
    # .envファイルを読み込み
    load_dotenv()
    
    print("=" * 60)
    print("ReActエージェント - 学習モード")
    print("=" * 60)

    # APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 注意: OPENAI_API_KEYが設定されていません")
        print("  .envファイルにAPIキーを設定してください")

    # 1. ReActAgentの初期化
    print("\n1. ReActエージェントの初期化")
    print("-" * 40)

    try:
        agent = ReActAgent(verbose=True)
        print("✅ ReActAgent初期化成功")
        print(f"  LLMモデル: {agent.llm.model_name}")
        print(f"  Temperature: {agent.llm.temperature}")
        print(f"  ツール数: {len(agent.tools_manager.get_tools())}")
    except Exception as e:
        print(f"⚠️ 初期化時の警告: {e}")
        agent = None

    # 2. プロンプトテンプレートの確認
    print("\n2. プロンプトテンプレート")
    print("-" * 40)

    if agent:
        print("システムプロンプト（一部）:")
        print(agent.system_prompt[:300] + "...")
        print("\n利用可能なツール:")
        print(agent._format_tools_description())

    # 3. ReActThoughtのデモ
    print("\n3. ReActThought（思考プロセス）のデモ")
    print("-" * 40)

    # 思考プロセスの例を作成
    thoughts = [
        ReActThought(
            reasoning="ユーザーは天気について聞いている。現在の天気情報を取得する必要がある。",
            action_needed=True,
            action="web_search",
            action_input={"query": "今日の東京の天気"},
            is_final_answer=False,
        ),
        ReActThought(
            reasoning="天気情報を取得できた。ユーザーに分かりやすく回答する。",
            action_needed=False,
            is_final_answer=True,
            final_answer="今日の東京は晴れで、最高気温は25度の予報です。",
        ),
    ]

    for i, thought in enumerate(thoughts, 1):
        print(f"\n思考ステップ {i}:")
        print(f"  推論: {thought.reasoning}")
        print(f"  アクション必要: {thought.action_needed}")
        if thought.action_needed:
            print(f"  アクション: {thought.action}")
            print(f"  入力: {thought.action_input}")
        if thought.is_final_answer:
            print(f"  最終回答: {thought.final_answer}")

    # 4. 実際のツール実行テスト
    print("\n4. ツール実行テスト")
    print("-" * 40)

    if agent:
        # 4-1. Calculatorツールのテスト
        print("\n[Test 1] Calculator ツールテスト")
        print("-" * 30)
        
        calculator_tool = None
        for tool in agent.tools_manager.get_tools():
            if tool.name == "calculator":
                calculator_tool = tool
                break
        
        if calculator_tool:
            test_cases = [
                {"expression": "2 + 2", "expected": 4},
                {"expression": "10 * 5", "expected": 50},
                {"expression": "sqrt(16)", "expected": 4.0},
                {"expression": "max(1, 5, 3)", "expected": 5},
            ]
            
            for test in test_cases:
                try:
                    result = calculator_tool.invoke({"expression": test["expression"]})
                    # 結果から数値を抽出
                    import re
                    match = re.search(r"= ([\d\.]+)", result)
                    if match:
                        actual = float(match.group(1))
                        passed = abs(actual - test["expected"]) < 0.0001
                        status = "✅ PASS" if passed else "❌ FAIL"
                        print(f"  {test['expression']}: {actual} (期待値: {test['expected']}) {status}")
                    else:
                        print(f"  {test['expression']}: 結果のパースに失敗")
                except Exception as e:
                    print(f"  {test['expression']}: エラー - {e}")
        else:
            print("  ⚠️ Calculator ツールが見つかりません")
        
        # 4-2. ReActループの個別メソッドテスト
        print("\n[Test 2] ReActループメソッドテスト")
        print("-" * 30)
        
        # テスト用の状態を作成
        test_query = "What is 15 + 25?"
        state = create_initial_state(test_query, max_iterations=3)
        
        # reason（推論）のテスト
        print("\n  推論ステップテスト:")
        initial_iteration = state["iteration_count"]
        
        # APIキーが無効な場合はスキップ
        if not os.getenv("OPENAI_API_KEY"):
            print("    ⚠️ APIキーが設定されていないため、推論テストをスキップ")
        else:
            state = agent.reason(state)
            
            # 推論が実行されたことを確認
            assert state["iteration_count"] == initial_iteration + 1, "イテレーションカウントが増加していない"
            assert state.get("next_step") is not None, "次のステップが設定されていない"
            print(f"    ✅ 推論実行成功: 次のステップ = {state['next_step']}")
        
        # ペンディングアクションがある場合、actのテスト
        if state["next_step"] == "act" and state["metadata"].get("pending_action"):
            print("\n  アクションステップテスト:")
            pending_action = state["metadata"]["pending_action"]
            print(f"    実行予定: {pending_action['tool']}({pending_action['input']})")
            
            state = agent.act(state)
            
            # アクションが実行されたことを確認
            assert len(state["tool_calls"]) > 0, "ツール呼び出しが記録されていない"
            assert state["next_step"] == "observe", "次のステップがobserveになっていない"
            print(f"    ✅ アクション実行成功: ツール呼び出し数 = {len(state['tool_calls'])}")
        
        # observe（観察）のテスト  
        if state["next_step"] == "observe":
            print("\n  観察ステップテスト:")
            state = agent.observe(state)
            
            # 観察が実行されたことを確認
            assert state["next_step"] in ["reason", "answer"], "次のステップが適切に設定されていない"
            print(f"    ✅ 観察実行成功: 次のステップ = {state['next_step']}")

    # 5. 完全な実行フローテスト
    print("\n5. 統合テスト - 完全な実行フロー")
    print("-" * 40)

    if agent:
        # APIキーが設定されている場合のみ実行
        if os.getenv("OPENAI_API_KEY"):
            print("\n[Test 3] エンドツーエンドテスト")
            print("-" * 30)
            
            test_queries = [
                {"query": "What is 100 divided by 4?", "check_keyword": "25"},
                {"query": "Calculate the square root of 144", "check_keyword": "12"},
            ]
            
            for test in test_queries:
                print(f"\n  テストクエリ: {test['query']}")
                try:
                    result = agent.run(test['query'], max_iterations=5)
                    
                    # 結果の検証
                    assert "answer" in result, "回答が含まれていない"
                    assert result["answer"], "回答が空"
                    
                    # キーワードチェック
                    if test['check_keyword'] in str(result["answer"]):
                        print(f"    ✅ PASS: 期待値 '{test['check_keyword']}' が回答に含まれている")
                    else:
                        print(f"    ⚠️ WARNING: 期待値 '{test['check_keyword']}' が回答に含まれていない")
                    
                    print(f"    ステップ数: {result.get('total_steps', 'N/A')}")
                    print(f"    ツール呼び出し: {result.get('tool_calls_count', 'N/A')}")
                    
                except Exception as e:
                    print(f"    ❌ FAIL: {e}")
        else:
            print("\n  ⚠️ 有効なOPENAI_API_KEYが設定されていないため、エンドツーエンドテストはスキップ")
            print("  💡 テスト用のモック実行:")
            
            # モック実行のデモ
            from unittest.mock import Mock, MagicMock, patch
            
            print("  💡 モックを使った単体テスト:")
            
            # Calculatorツールの直接テスト
            calc_tool = None
            for tool in agent.tools_manager.get_tools():
                if tool.name == "calculator":
                    calc_tool = tool
                    break
            
            if calc_tool:
                # 正常な計算のテスト
                result = calc_tool.invoke({"expression": "100 / 4"})
                assert "25" in result, "計算結果が正しくない"
                print("    ✅ モック計算テスト: 100 / 4 = 25.0")
                
                # 状態管理のテスト
                test_state = create_initial_state("test query", max_iterations=2)
                test_state = add_reasoning_step(test_state, thought="テスト思考", action="calculator")
                assert len(test_state["reasoning_steps"]) == 1, "推論ステップが追加されていない"
                assert test_state["reasoning_steps"][0].thought == "テスト思考", "思考内容が正しくない"
                print("    ✅ 状態管理テスト: 推論ステップの追加成功")

    # 6. エラーハンドリングテスト
    print("\n6. エラーハンドリングテスト")
    print("-" * 40)
    
    if agent:
        print("\n[Test 4] エラーケーステスト")
        print("-" * 30)
        
        # 無効な式のテスト
        calculator_tool = None
        for tool in agent.tools_manager.get_tools():
            if tool.name == "calculator":
                calculator_tool = tool
                break
        
        if calculator_tool:
            error_cases = [
                {"expression": "1 / 0", "error_type": "ZeroDivisionError"},
                {"expression": "invalid_expr", "error_type": "NameError"},
                {"expression": "import os", "error_type": "SyntaxError"},
            ]
            
            for test in error_cases:
                try:
                    result = calculator_tool.invoke({"expression": test["expression"]})
                    if "エラー" in result or "error" in result.lower():
                        print(f"  ✅ {test['expression']}: エラーが適切に処理された")
                    else:
                        print(f"  ❌ {test['expression']}: エラーが検出されなかった")
                except Exception as e:
                    print(f"  ✅ {test['expression']}: 例外が適切にキャッチされた ({type(e).__name__})")
        
        # 存在しないツールのテスト
        print("\n  存在しないツールのテスト:")
        state = create_initial_state("test", max_iterations=1)
        state["metadata"]["pending_action"] = {"tool": "non_existent_tool", "input": {}}
        state["next_step"] = "act"
        
        state = agent.act(state)
        assert state.get("error") is not None, "エラーが記録されていない"
        print(f"    ✅ 存在しないツールのエラー処理成功")
    
    print("\n" + "=" * 60)
    print("✅ テスト完了")
    print("\n📊 テストサマリー:")
    print("  1. ツール実行テスト: 完了")
    print("  2. ReActループメソッドテスト: 完了")
    print("  3. 統合テスト: 完了/スキップ（APIキー依存）")
    print("  4. エラーハンドリングテスト: 完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
