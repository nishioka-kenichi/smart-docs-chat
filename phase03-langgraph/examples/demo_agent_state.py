#!/usr/bin/env python
"""
Agent State管理のデモンストレーション

このスクリプトは、LangGraphのAgent State管理機能を実際に動かして
学習するためのデモンストレーションです。

実行方法:
    python examples/demo_agent_state.py

デバッグ方法:
    1. VSCodeでこのファイルを開く
    2. ブレークポイントを設定
    3. F5キーでデバッグ実行
"""

import sys
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_state import (
    create_initial_state,
    add_reasoning_step,
    add_tool_call,
    should_continue,
    format_reasoning_history,
    extract_final_state,
    CheckpointData
)


def main():
    """メイン処理"""
    print("="*60)
    print("Agent State管理モジュール - 学習モード")
    print("="*60)
    
    # 1. 初期状態の作成テスト
    print("\n1. 初期状態の作成")
    print("-"*40)
    test_query = "Pythonのリスト内包表記について教えてください"
    state = create_initial_state(test_query, max_iterations=5)
    
    print(f"クエリ: {test_query}")
    print(f"現在のステップ: {state['current_step']}")
    print(f"次のステップ: {state['next_step']}")
    print(f"最大イテレーション: {state['max_iterations']}")
    print(f"メッセージ数: {len(state['messages'])}")
    
    # 2. 推論ステップの追加テスト
    print("\n2. 推論ステップの追加")
    print("-"*40)
    state = add_reasoning_step(
        state,
        thought="ユーザーはPythonのリスト内包表記について知りたがっている。基本的な構文と例を提供する必要がある。",
        action="search_documentation",
        observation="Pythonドキュメントから関連情報を取得"
    )
    
    print(f"推論ステップ数: {len(state['reasoning_steps'])}")
    for i, step in enumerate(state['reasoning_steps'], 1):
        print(f"\nステップ {i}:")
        print(f"  思考: {step.thought[:50]}...")
        print(f"  アクション: {step.action}")
        print(f"  観察: {step.observation}")
    
    # 3. ツール呼び出しの記録テスト
    print("\n3. ツール呼び出しの記録")
    print("-"*40)
    state = add_tool_call(
        state,
        tool_name="rag_search",
        arguments={"query": "Python list comprehension", "top_k": 5},
        result="[x**2 for x in range(10)] creates a list of squares"
    )
    
    print(f"ツール呼び出し数: {len(state['tool_calls'])}")
    for call in state['tool_calls']:
        print(f"\nツール: {call.tool_name}")
        print(f"  引数: {call.arguments}")
        print(f"  結果: {call.result[:50]}..." if call.result else "  結果: なし")
    
    # 4. 継続判定のテスト
    print("\n4. 継続判定のテスト")
    print("-"*40)
    
    # 正常な状態での判定
    print(f"正常な状態での継続判定: {should_continue(state)}")
    
    # エラー状態での判定
    error_state = dict(state)
    error_state["error"] = "エラーが発生しました"
    print(f"エラー状態での継続判定: {should_continue(error_state)}")
    
    # 最大イテレーション到達
    max_iter_state = dict(state)
    max_iter_state["iteration_count"] = 5
    print(f"最大イテレーション到達時の継続判定: {should_continue(max_iter_state)}")
    
    # 5. 推論履歴のフォーマットテスト
    print("\n5. 推論履歴のフォーマット")
    print("-"*40)
    
    # もう一つ推論ステップを追加
    state = add_reasoning_step(
        state,
        thought="基本的な情報を取得できた。次は実用的な例を提供する。",
        action="generate_examples",
        observation="複数の実用例を生成"
    )
    
    history = format_reasoning_history(state)
    print("フォーマット済み履歴:")
    print(history)
    
    # 6. 最終状態の抽出テスト
    print("\n6. 最終状態の抽出")
    print("-"*40)
    
    # 最終回答を設定
    state["final_answer"] = "リスト内包表記は [式 for 変数 in イテラブル] の形式で書きます。"
    state["iteration_count"] = 3
    
    final = extract_final_state(state)
    print("抽出された最終状態:")
    for key, value in final.items():
        if key == "answer":
            print(f"  {key}: {value[:50]}...")
        elif key not in ["context", "metadata"]:
            print(f"  {key}: {value}")
    
    # 7. CheckpointDataのテスト
    print("\n7. CheckpointDataの作成")
    print("-"*40)
    
    checkpoint = CheckpointData(
        checkpoint_id="test_checkpoint_001",
        state=dict(state),
        step_name="reasoning",
        iteration=3,
        metadata={"test_mode": True}
    )
    
    print(f"チェックポイントID: {checkpoint.checkpoint_id}")
    print(f"ステップ名: {checkpoint.step_name}")
    print(f"イテレーション: {checkpoint.iteration}")
    print(f"タイムスタンプ: {checkpoint.timestamp}")
    
    print("\n" + "="*60)
    print("✅ すべてのテストが完了しました")
    print("="*60)


if __name__ == "__main__":
    main()