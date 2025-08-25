#!/usr/bin/env python
"""
メインアプリケーションのデモンストレーション

このスクリプトは、LangGraphエージェントの統合アプリケーションを
実際に実行するためのデモンストレーションです。

実行方法:
    # 対話モード
    python examples/demo_main.py
    
    # 単一クエリ
    python examples/demo_main.py "あなたの質問"
    
    # オプション付き
    python examples/demo_main.py --help

デバッグ方法:
    1. VSCodeでこのファイルを開く
    2. ブレークポイントを設定
    3. F5キーでデバッグ実行
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 環境変数の読み込み
from dotenv import load_dotenv
load_dotenv()

# ローカルモジュールのインポート
from graph_builder import GraphBuilder
from checkpointer import CheckpointManager
from tools import ToolsManager
from langchain_openai import ChatOpenAI


class LangGraphAgent:
    """LangGraphエージェントのメインクラス"""
    
    def __init__(self, config_path: str = "./config/settings.yaml", verbose: bool = True):
        """初期化"""
        self.config_path = config_path
        self.verbose = verbose
        
        # 必要な環境変数のチェック
        self._check_environment()
        
        # コンポーネントの初期化
        self._initialize_components()
    
    def _check_environment(self):
        """環境変数をチェック"""
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}")
            print("Please set them in your .env file")
            sys.exit(1)
        
        # ChromaDBパスの設定
        if not os.getenv("CHROMADB_PATH"):
            os.environ["CHROMADB_PATH"] = "../phase01-local/data/chromadb"
            if self.verbose:
                print(f"ℹ️ Using default ChromaDB path: {os.environ['CHROMADB_PATH']}")
    
    def _initialize_components(self):
        """コンポーネントを初期化"""
        if self.verbose:
            print("🚀 Initializing LangGraph Agent...")
        
        # LLMの初期化
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        )
        
        # ツールマネージャーの初期化
        self.tools_manager = ToolsManager(self.config_path)
        
        # チェックポイントマネージャーの初期化
        checkpoint_dir = os.getenv("LANGGRAPH_CHECKPOINT_DIR", "./data/checkpoints")
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            verbose=self.verbose
        )
        
        # グラフビルダーの初期化
        self.graph_builder = GraphBuilder(
            llm=self.llm,
            tools_manager=self.tools_manager,
            checkpoint_manager=self.checkpoint_manager,
            config_path=self.config_path,
            verbose=self.verbose
        )
        
        if self.verbose:
            print("✅ Initialization complete")
            print(f"   Available tools: {', '.join(self.tools_manager.get_tool_names())}")
    
    def run_interactive(self):
        """対話モードで実行"""
        print("\n" + "="*60)
        print("🤖 LangGraph Agent - Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  /exit    - 終了")
        print("  /clear   - 会話履歴をクリア")
        print("  /save    - チェックポイントを保存")
        print("  /load    - チェックポイントから復元")
        print("  /list    - チェックポイント一覧")
        print("  /tools   - 利用可能なツール一覧")
        print("  /help    - ヘルプを表示")
        print("="*60 + "\n")
        
        thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        while True:
            try:
                # ユーザー入力
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                # コマンド処理
                if user_input.startswith("/"):
                    self._handle_command(user_input, thread_id)
                    continue
                
                # エージェント実行
                print("\n🤖 Agent: Thinking...\n")
                result = self.graph_builder.run(
                    query=user_input,
                    thread_id=thread_id,
                    max_iterations=int(os.getenv("MAX_ITERATIONS", "10"))
                )
                
                # 結果表示
                print("\n" + "-"*60)
                print(f"🤖 Agent: {result['answer']}")
                print("-"*60)
                
                # 統計情報表示
                if self.verbose:
                    print(f"\n📊 Statistics:")
                    print(f"   Reasoning steps: {result.get('reasoning_steps', 0)}")
                    print(f"   Tool calls: {result.get('tool_calls', 0)}")
                    print(f"   Iterations: {result.get('iterations', 0)}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _handle_command(self, command: str, thread_id: str):
        """コマンドを処理"""
        cmd = command.lower().strip()
        
        if cmd == "/exit":
            print("👋 Goodbye!")
            sys.exit(0)
        
        elif cmd == "/clear":
            # 新しいスレッドIDを生成
            thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print("✅ Conversation cleared")
        
        elif cmd == "/save":
            print("💾 Saving checkpoint...")
            print("⚠️ Manual save not yet implemented")
        
        elif cmd == "/load":
            self._load_checkpoint()
        
        elif cmd == "/list":
            self._list_checkpoints()
        
        elif cmd == "/tools":
            print("\n🛠️ Available Tools:")
            for name, desc in self.tools_manager.get_tool_descriptions().items():
                print(f"  • {name}: {desc}")
        
        elif cmd == "/help":
            self._show_help()
        
        else:
            print(f"❓ Unknown command: {command}")
    
    def _load_checkpoint(self):
        """チェックポイントから復元"""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if not checkpoints:
            print("⚠️ No checkpoints available")
            return
        
        print("\n📂 Available Checkpoints:")
        for i, cp in enumerate(checkpoints, 1):
            print(f"  {i}. {cp['checkpoint_id']} ({cp['timestamp']})")
        
        try:
            choice = input("\nSelect checkpoint number (or 'c' to cancel): ").strip()
            if choice.lower() == 'c':
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                checkpoint_id = checkpoints[idx]['checkpoint_id']
                print(f"Loading checkpoint: {checkpoint_id}")
                print("⚠️ Checkpoint loading not yet fully implemented")
            else:
                print("❌ Invalid selection")
        
        except ValueError:
            print("❌ Invalid input")
    
    def _list_checkpoints(self):
        """チェックポイント一覧を表示"""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        if not checkpoints:
            print("⚠️ No checkpoints available")
            return
        
        print("\n📂 Checkpoints:")
        for cp in checkpoints:
            size = self.checkpoint_manager.get_checkpoint_size(cp['checkpoint_id'])
            size_mb = size / 1024 / 1024 if size else 0
            print(f"  • {cp['checkpoint_id']}")
            print(f"    Time: {cp['timestamp']}")
            print(f"    Step: {cp['step_name']}, Iteration: {cp['iteration']}")
            print(f"    Size: {size_mb:.2f} MB")
        
        total_size = self.checkpoint_manager.get_total_size()
        print(f"\n  Total size: {total_size / 1024 / 1024:.2f} MB")
    
    def _show_help(self):
        """ヘルプを表示"""
        print("\n" + "="*60)
        print("📚 LangGraph Agent Help")
        print("="*60)
        print("\n🎯 Usage:")
        print("  Type your question or request normally.")
        print("  The agent will use reasoning and tools to find the answer.")
        print("\n⚡ Commands:")
        print("  /exit    - Exit the program")
        print("  /clear   - Clear conversation history")
        print("  /save    - Save current state as checkpoint")
        print("  /load    - Load from a checkpoint")
        print("  /list    - List all checkpoints")
        print("  /tools   - Show available tools")
        print("  /help    - Show this help message")
        print("\n💡 Tips:")
        print("  • The agent can search documents, calculate, and access files")
        print("  • It uses ReAct pattern for step-by-step reasoning")
        print("  • Checkpoints allow you to resume from saved states")
        print("="*60)
    
    def run_single(self, query: str, checkpoint_id: str = None):
        """単一クエリを実行"""
        print(f"\n🚀 Running query: {query}")
        
        result = self.graph_builder.run(
            query=query,
            checkpoint_id=checkpoint_id,
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10"))
        )
        
        # 結果を表示
        print("\n" + "="*60)
        print("📝 Answer:")
        print(result['answer'])
        print("="*60)
        
        if self.verbose:
            print(f"\n📊 Statistics:")
            print(f"  Reasoning steps: {result.get('reasoning_steps', 0)}")
            print(f"  Tool calls: {result.get('tool_calls', 0)}")
            print(f"  Iterations: {result.get('iterations', 0)}")
        
        return result


def main():
    """メイン関数"""
    # コマンドラインパーサーの設定
    parser = argparse.ArgumentParser(description="LangGraph Agent - Phase 3")
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to process (if not provided, runs in interactive mode)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint ID"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/settings.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable verbose output"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Visualize the agent graph"
    )
    
    args = parser.parse_args()
    
    # エージェントの初期化
    agent = LangGraphAgent(
        config_path=args.config,
        verbose=not args.quiet
    )
    
    # グラフの可視化
    if args.visualize:
        print("📊 Generating graph visualization...")
        image = agent.graph_builder.visualize()
        if image:
            print("✅ Graph visualization saved to ./data/graph.png")
        return
    
    # 実行モードの選択
    if args.interactive or not args.query:
        # 対話モード
        agent.run_interactive()
    else:
        # 単一クエリ実行
        agent.run_single(args.query, checkpoint_id=args.resume)


if __name__ == "__main__":
    main()