#!/usr/bin/env python
"""
ツール管理のデモンストレーション

このスクリプトは、LangGraphで使用する各種ツールの動作を
実際に確認するためのデモンストレーションです。

実行方法:
    python examples/demo_tools.py

デバッグ方法:
    1. VSCodeでこのファイルを開く
    2. ブレークポイントを設定（例：_calculate関数内）
    3. F5キーでデバッグ実行
"""

import os
import sys
import tempfile
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools import ToolsManager
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


def main():
    """メイン処理"""
    print("=" * 60)
    print("ツール管理モジュール - 学習モード")
    print("=" * 60)

    # # 環境変数の設定（テスト用）
    # if not os.getenv("OPENAI_API_KEY"):
    #     os.environ["OPENAI_API_KEY"] = "sk-test-key-for-demo"

    # 1. ToolsManagerの初期化
    print("\n1. ToolsManagerの初期化")
    print("-" * 40)

    try:
        manager = ToolsManager()
        print("✅ ToolsManager初期化成功")
    except Exception as e:
        print(f"⚠️ 初期化時の警告: {e}")
        # 設定ファイルがない場合のフォールバック
        print("デフォルト設定で続行します...")
        manager = None

    # 2. 利用可能なツールの確認
    print("\n2. 利用可能なツール")
    print("-" * 40)

    if manager:
        tools = manager.get_tools()
        tool_names = manager.get_tool_names()

        print(f"登録ツール数: {len(tools)}")
        for name in tool_names:
            print(f"  - {name}")

        print("\nツールの説明:")
        for name, desc in manager.get_tool_descriptions().items():
            print(f"  {name}: {desc}")

    # 3. 計算ツールのテスト
    print("\n3. 計算ツールのテスト")
    print("-" * 40)

    if manager:
        calc_tool = None
        for tool in manager.get_tools():
            if tool.name == "calculator":
                calc_tool = tool
                break

        if calc_tool:
            test_expressions = ["2 + 2", "10 * 5", "100 / 4", "2 ** 8", "sqrt(16)", "sum([1, 2, 3, 4, 5])"]

            for expr in test_expressions:
                try:
                    result = calc_tool.invoke({"expression": expr})
                    print(f"  {expr} → {result}")
                except Exception as e:
                    print(f"  {expr} → エラー: {e}")
        else:
            print("  計算ツールが見つかりません")

    # 4. ファイル操作ツールのテスト
    print("\n4. ファイル操作ツールのテスト")
    print("-" * 40)

    if manager:
        write_tool = None
        read_tool = None

        for tool in manager.get_tools():
            if tool.name == "write_file":
                write_tool = tool
            elif tool.name == "read_file":
                read_tool = tool

        if write_tool and read_tool:
            # テンポラリファイルで安全にテスト
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
                test_file = tf.name

            try:
                # 書き込みテスト
                test_content = "これはツールテストです。\n複数行のテキストを\n書き込みます。"
                write_result = write_tool.invoke(
                    {"file_path": test_file, "content": test_content, "mode": "w"}
                )
                print(f"  書き込み: {write_result}")

                # 読み込みテスト
                read_result = read_tool.invoke({"file_path": test_file, "encoding": "utf-8"})
                print(f"  読み込み: 成功（{len(read_result)}文字）")

            finally:
                # クリーンアップ
                Path(test_file).unlink(missing_ok=True)
                print("  テストファイルを削除しました")
        else:
            print("  ファイル操作ツールが見つかりません")

    # 5. RAG検索ツールの状態確認
    print("\n5. RAG検索ツールの状態")
    print("-" * 40)

    if manager:
        rag_tool = None
        for tool in manager.get_tools():
            if tool.name == "rag_search":
                rag_tool = tool
                break

        if rag_tool:
            print("  ✅ RAG検索ツールが登録されています")
            print("  注: 実際の検索にはPhase 1のChromaDBが必要です")

            # モック検索のデモ
            try:
                result = rag_tool.invoke({"query": "テストクエリ", "top_k": 3, "score_threshold": 0.5})
                print(f"  検索結果: {result[:100]}...")
            except Exception as e:
                print(f"  検索実行時の注意: {e}")
        else:
            print("  RAG検索ツールは無効または未設定です")

    # 6. カスタムツールの追加デモ
    print("\n6. カスタムツールの追加（デモ）")
    print("-" * 40)

    class GreetingInput(BaseModel):
        """挨拶ツールの入力"""

        name: str = Field(description="挨拶する相手の名前")
        language: str = Field(default="ja", description="言語（ja/en）")

    def greeting_function(name: str, language: str = "ja") -> str:
        """挨拶を返す関数"""
        if language == "ja":
            return f"こんにちは、{name}さん！"
        else:
            return f"Hello, {name}!"

    greeting_tool = StructuredTool.from_function(
        func=greeting_function,
        name="greeting",
        description="名前を指定して挨拶する",
        args_schema=GreetingInput,
    )

    # カスタムツールのテスト
    result = greeting_tool.invoke({"name": "学習者", "language": "ja"})
    print(f"  カスタムツール実行: {result}")

    result = greeting_tool.invoke({"name": "Learner", "language": "en"})
    print(f"  カスタムツール実行: {result}")

    print("\n" + "=" * 60)
    print("✅ すべてのテストが完了しました")
    print("💡 ヒント: 各ツールの実装を詳しく見るには、")
    print("   該当する関数にブレークポイントを設定してデバッグ実行してください")
    print("=" * 60)


if __name__ == "__main__":
    main()
