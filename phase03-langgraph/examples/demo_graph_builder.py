#!/usr/bin/env python
"""
LangGraphビルダーのデモンストレーション

このスクリプトは、LangGraphの動作フローとグラフ構造を
実際に確認するためのデモンストレーションです。

実行方法:
    python examples/demo_graph_builder.py

デバッグ方法:
    1. VSCodeでこのファイルを開く
    2. ブレークポイントを設定（例：_build_graph関数内）
    3. F5キーでデバッグ実行
"""

import os
import sys
from pathlib import Path
from textwrap import dedent
from dotenv import load_dotenv

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph_builder import GraphBuilder
from agent_state import create_initial_state


def main():
    """メイン処理"""
    # .envファイルを読み込み
    load_dotenv()
    
    print("=" * 60)
    print("LangGraphビルダー - 学習モード")
    print("=" * 60)

    # APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 注意: OPENAI_API_KEYが設定されていません")
        print("  .envファイルにAPIキーを設定してください")

    # 1. GraphBuilderの初期化
    print("\n1. GraphBuilderの初期化")
    print("-" * 40)

    try:
        builder = GraphBuilder(verbose=False)
        print("✅ GraphBuilder初期化成功")
        print(f"  エージェント: {builder.agent.__class__.__name__}")
        print(f"  チェックポイント: {builder.checkpoint_manager.__class__.__name__}")
    except Exception as e:
        print(f"⚠️ 初期化時の警告: {e}")
        builder = None

    # 2. グラフの実際のコンパイルと実行
    print("\n2. グラフのコンパイルと実行")
    print("-" * 40)

    if builder:
        try:
            # グラフをコンパイル
            app = builder.compile()
            print("✅ グラフコンパイル成功")
            
            # テストクエリで実際に実行
            test_queries = [
                "What is 50 + 30?",
                "Calculate the square root of 256",
            ]
            
            for query in test_queries:
                print(f"\n実行テスト: {query}")
                print("-" * 30)
                
                # 初期状態を作成
                state = create_initial_state(query, max_iterations=5)
                
                # ストリーミング実行（thread_idを指定）
                print("実行ログ:")
                node_count = 0
                config = {"configurable": {"thread_id": f"test_{query[:10]}"}}
                
                try:
                    for output in app.stream(state, config):
                        node_name = list(output.keys())[0]
                        node_state = output[node_name]
                        node_count += 1
                        
                        print(f"  [{node_count}] {node_name}ノード実行")
                        
                        # 重要な情報を表示
                        if node_name == "reason" and "next_step" in node_state:
                            print(f"      → 次のステップ: {node_state['next_step']}")
                        elif node_name == "act" and "tool_calls" in node_state and node_state["tool_calls"]:
                            last_call = node_state["tool_calls"][-1]
                            if hasattr(last_call, 'tool_name'):
                                print(f"      → ツール: {last_call.tool_name}")
                            else:
                                print(f"      → ツール: {last_call.get('tool_name', 'N/A')}")
                        elif node_name == "answer" and "final_answer" in node_state:
                            answer = str(node_state['final_answer'])
                            print(f"      → 回答: {answer[:100]}..." if len(answer) > 100 else f"      → 回答: {answer}")
                    
                    print(f"\n実行完了: {node_count}ノード実行")
                    
                except Exception as e:
                    print(f"  ストリーミング実行エラー: {e}")
                
        except ImportError as e:
            print(f"❌ LangGraphがインストールされていません: {e}")
            print("  pip install langgraph を実行してください")
            app = None
        except Exception as e:
            print(f"❌ 実行エラー: {e}")
            app = None
    else:
        app = None

    # 3. モックを使用した単体テスト
    print("\n3. モックを使用した単体テスト（LangGraphなし）")
    print("-" * 40)
    
    if builder and not app:
        print("LangGraphが利用できないため、モックで動作確認:")
        
        # 状態遷移のシミュレーション
        test_state = create_initial_state("What is 2 + 2?", max_iterations=3)
        
        print("\n状態遷移シミュレーション:")
        print(f"  [1] startノード実行")
        test_state = builder._start_node(test_state)
        print(f"      → メタデータ初期化: {bool(test_state.get('metadata'))}")
        
        # calculatorツールの直接実行
        if builder.agent.tools_manager:
            calc_tool = None
            for tool in builder.agent.tools_manager.get_tools():
                if tool.name == "calculator":
                    calc_tool = tool
                    break
            
            if calc_tool:
                print(f"\n  [2] ツール実行テスト (calculator)")
                result = calc_tool.invoke({"expression": "2 + 2"})
                print(f"      → 結果: {result}")
                
                # 状態に結果を追加
                from agent_state import add_tool_call
                test_state = add_tool_call(test_state, "calculator", {"expression": "2 + 2"}, result)
                print(f"      → ツール呼び出し記録: {len(test_state['tool_calls'])}件")

    # 4. ルーティングロジックの実際のテスト
    print("\n4. ルーティングロジックの動作確認")
    print("-" * 40)

    if builder:
        # 実際の状態遷移テスト
        test_cases = [
            {"next_step": "act", "error": None, "desc": "アクション実行へ"},
            {"next_step": "answer", "error": None, "desc": "回答生成へ"},
            {"next_step": None, "error": "エラー", "desc": "エラー時の処理"},
        ]
        
        print("推論後のルーティングテスト:")
        for test in test_cases:
            test_state = create_initial_state("test")
            if test["next_step"]:
                test_state["next_step"] = test["next_step"]
            if test["error"]:
                test_state["error"] = test["error"]
            
            result = builder._route_after_reason(test_state)
            status = "✅" if result else "❌"
            print(f"  {status} {test['desc']}: → {result}")
        
        # 観察後のルーティングテスト
        print("\n観察後のルーティングテスト:")
        test_state = create_initial_state("test")
        test_state["next_step"] = "reason"
        result = builder._route_after_observe(test_state)
        print(f"  ✅ 次の推論へ: → {result}")
        
        test_state["iteration_count"] = 10
        test_state["max_iterations"] = 5
        result = builder._route_after_observe(test_state)
        print(f"  ✅ 最大イテレーション超過: → {result}")

    # 5. 実際のノード関数実行テスト
    print("\n5. ノード関数の実行テスト")
    print("-" * 40)

    if builder:
        test_state = create_initial_state("Calculate 15 * 3")
        
        # 各ノードを順次実行
        print("ノード実行シーケンス:")
        
        # 1. Start
        print("\n[1] Startノード:")
        test_state = builder._start_node(test_state)
        assert "start_time" in test_state.get("metadata", {}), "開始時刻が設定されていない"
        print(f"  ✅ メタデータ初期化完了")
        print(f"  開始時刻: {test_state['metadata']['start_time']}")
        
        # 2. Reason (モック)
        print("\n[2] Reasonノード（モック）:")
        from agent_state import add_reasoning_step
        test_state = add_reasoning_step(test_state, "計算が必要です", "calculator")
        test_state["next_step"] = "act"
        test_state["metadata"]["pending_action"] = {
            "tool": "calculator",
            "input": {"expression": "15 * 3"}
        }
        print(f"  ✅ 推論ステップ追加")
        print(f"  次のステップ: {test_state['next_step']}")
        
        # 3. Act
        print("\n[3] Actノード:")
        if builder.agent.tools_manager:
            test_state = builder.agent.act(test_state)
            if test_state.get("tool_calls"):
                last_call = test_state["tool_calls"][-1]
                print(f"  ✅ ツール実行完了: {last_call.tool_name}")
                print(f"  結果: {last_call.result}")
        
        # 4. Checkpoint
        print("\n[4] Checkpointノード:")
        result_state = builder._checkpoint_node(test_state)
        has_checkpoint = "last_checkpoint_id" in result_state.get("metadata", {})
        print(f"  {'✅' if has_checkpoint else '❌'} チェックポイント保存: {has_checkpoint}")

    # 6. 実際の統合実行テスト
    print("\n6. 統合実行テスト")
    print("-" * 40)

    if builder:
        # APIキーが有効な場合のみ実行
        if os.getenv("OPENAI_API_KEY"):
            print("\n完全な実行テスト (builder.run):")
            
            test_queries = [
                {"query": "What is 25 + 75?", "check": "100"},
                {"query": "Calculate 8 squared", "check": "64"},
            ]
            
            for test in test_queries:
                print(f"\nクエリ: {test['query']}")
                
                try:
                    result = builder.run(
                        query=test['query'],
                        max_iterations=5,
                        thread_id=f"test_{test['query'][:10]}"
                    )
                    
                    if result.get("answer"):
                        answer = str(result["answer"])
                        if test['check'] in answer:
                            print(f"  ✅ 正解: {test['check']} が回答に含まれている")
                        else:
                            print(f"  ⚠️ 回答: {answer[:100]}")
                        
                        print(f"  総ステップ数: {result.get('total_steps', 'N/A')}")
                        print(f"  ツール呼び出し: {result.get('tool_calls_count', 'N/A')}回")
                    else:
                        print(f"  ❌ 回答が生成されませんでした")
                        
                except Exception as e:
                    print(f"  ❌ 実行エラー: {e}")
        else:
            print("⚠️ APIキーが設定されていないため、統合テストはスキップ")

    print("\n" + "=" * 60)
    print("✅ デモンストレーション完了")
    print("💡 実際の実行には有効なAPIキーが必要です")
    print("=" * 60)


if __name__ == "__main__":
    main()
