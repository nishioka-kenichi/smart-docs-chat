#!/usr/bin/env python3
"""
CLIチャットインターフェース
ユーザーと対話的にRAGシステムを使用するためのCLI

主な機能:
1. 対話的な質問応答
2. コマンドによる操作（履歴表示、クリア、終了など）
3. 検索結果の詳細表示
4. 会話履歴の管理
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional
import readline  # 入力履歴とタブ補完のため
from dotenv import load_dotenv

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_chain import RAGChain

# 環境変数の読み込み
load_dotenv()

# カラー出力用のANSIエスケープコード
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CLIChat:
    """CLIチャットクラス"""
    
    def __init__(self):
        """初期化"""
        self.rag_chain = None
        self.session_start = datetime.now()
        self.question_count = 0
        
        # コマンドのヘルプ
        self.commands = {
            "/help": "利用可能なコマンドを表示",
            "/history": "会話履歴を表示",
            "/clear": "会話履歴をクリア",
            "/search": "ドキュメント検索のみ実行（回答生成なし）",
            "/sources": "最後の回答のソース詳細を表示",
            "/save": "会話履歴を保存",
            "/stats": "セッション統計を表示",
            "/exit": "チャットを終了",
            "/quit": "チャットを終了",
        }
        
        self.last_result = None  # 最後の回答結果を保存
    
    def print_colored(self, text: str, color: str = ""):
        """色付きテキストを出力"""
        print(f"{color}{text}{Colors.ENDC}")
    
    def print_header(self):
        """ヘッダーを表示"""
        self.print_colored("\n" + "=" * 70, Colors.OKCYAN)
        self.print_colored("   🤖 スマートドキュメントRAGチャット", Colors.BOLD + Colors.OKCYAN)
        self.print_colored("=" * 70, Colors.OKCYAN)
        print(f"\n📅 セッション開始: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💡 ヘルプを表示するには {Colors.OKGREEN}/help{Colors.ENDC} と入力してください")
        print(f"🚪 終了するには {Colors.WARNING}/exit{Colors.ENDC} と入力してください\n")
    
    def print_help(self):
        """ヘルプを表示"""
        self.print_colored("\n📚 利用可能なコマンド:", Colors.BOLD)
        for cmd, desc in self.commands.items():
            print(f"  {Colors.OKGREEN}{cmd:12}{Colors.ENDC} - {desc}")
        print()
    
    def initialize_rag(self):
        """RAGチェーンを初期化"""
        try:
            self.print_colored("\n⚙️  システムを初期化中...", Colors.OKCYAN)
            self.rag_chain = RAGChain()
            self.print_colored("✅ 初期化完了！チャットを開始できます。\n", Colors.OKGREEN)
            return True
        except Exception as e:
            self.print_colored(f"\n❌ 初期化エラー: {e}", Colors.FAIL)
            self.print_colored("\n以下を確認してください:", Colors.WARNING)
            print("  1. .envファイルにOPENAI_API_KEYが設定されているか")
            print("  2. indexer.pyを実行してベクトルストアが作成されているか")
            print("  3. ネットワーク接続が正常か\n")
            return False
    
    def process_command(self, command: str) -> bool:
        """
        コマンドを処理
        
        Args:
            command: コマンド文字列
        
        Returns:
            続行する場合True、終了する場合False
        """
        command_lower = command.lower().strip()
        
        if command_lower in ["/exit", "/quit"]:
            return False
        
        elif command_lower == "/help":
            self.print_help()
        
        elif command_lower == "/history":
            self.show_history()
        
        elif command_lower == "/clear":
            self.clear_history()
        
        elif command_lower.startswith("/search "):
            query = command[8:].strip()
            self.search_documents(query)
        
        elif command_lower == "/sources":
            self.show_sources()
        
        elif command_lower == "/save":
            self.save_history()
        
        elif command_lower == "/stats":
            self.show_stats()
        
        else:
            self.print_colored(f"❓ 不明なコマンド: {command}", Colors.WARNING)
            print(f"   {Colors.OKGREEN}/help{Colors.ENDC} でコマンド一覧を表示")
        
        return True
    
    def show_history(self):
        """会話履歴を表示"""
        history = self.rag_chain.get_conversation_history()
        
        if not history:
            self.print_colored("📭 会話履歴はまだありません", Colors.WARNING)
            return
        
        self.print_colored("\n💬 会話履歴:", Colors.BOLD)
        print("-" * 50)
        
        for i, msg in enumerate(history, 1):
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            role_color = Colors.OKBLUE if msg["role"] == "user" else Colors.OKGREEN
            
            print(f"\n{i}. {role_icon} ", end="")
            self.print_colored(msg["role"].capitalize(), role_color + Colors.BOLD)
            
            # 長いメッセージは省略
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            print(f"   {content}")
        
        print("-" * 50)
    
    def clear_history(self):
        """会話履歴をクリア"""
        self.rag_chain.clear_memory()
        self.last_result = None
        self.print_colored("💨 会話履歴をクリアしました", Colors.OKGREEN)
    
    def search_documents(self, query: str):
        """ドキュメント検索のみ実行"""
        self.print_colored(f"\n🔍 検索中: '{query}'", Colors.OKCYAN)
        
        try:
            results = self.rag_chain.search_similar_documents(query, k=10)
            
            if not results:
                self.print_colored("📭 関連ドキュメントが見つかりませんでした", Colors.WARNING)
                return
            
            self.print_colored(f"\n📚 検索結果 ({len(results)}件):", Colors.BOLD)
            print("-" * 50)
            
            for i, (doc, score) in enumerate(results, 1):
                title = doc.metadata.get("title", "無題")
                source = doc.metadata.get("source", "不明")
                
                # スコアに応じて色を変更
                if score > 0.8:
                    score_color = Colors.OKGREEN
                elif score > 0.6:
                    score_color = Colors.OKCYAN
                else:
                    score_color = Colors.WARNING
                
                print(f"\n{i}. ", end="")
                self.print_colored(f"[{source}] {title}", Colors.BOLD)
                print(f"   ", end="")
                self.print_colored(f"スコア: {score:.3f}", score_color)
                
                # 内容のプレビュー
                preview = doc.page_content[:150].replace("\n", " ")
                print(f"   {preview}...")
            
            print("-" * 50)
            
        except Exception as e:
            self.print_colored(f"❌ 検索エラー: {e}", Colors.FAIL)
    
    def show_sources(self):
        """最後の回答のソース詳細を表示"""
        if not self.last_result:
            self.print_colored("📭 表示するソース情報がありません", Colors.WARNING)
            return
        
        sources = self.last_result.get("sources", [])
        if not sources:
            self.print_colored("📭 ソース情報がありません", Colors.WARNING)
            return
        
        self.print_colored("\n📚 参照ソース詳細:", Colors.BOLD)
        print("-" * 50)
        
        for i, source in enumerate(sources, 1):
            print(f"\n{i}. ", end="")
            self.print_colored(f"[{source['source']}] {source['title']}", Colors.BOLD)
            self.print_colored(f"   スコア: {source['score']:.3f}", Colors.OKCYAN)
            print(f"   内容プレビュー:")
            print(f"   {source['content_preview']}")
        
        print("-" * 50)
    
    def save_history(self):
        """会話履歴を保存"""
        try:
            history = self.rag_chain.get_conversation_history()
            
            # 保存ファイル名を生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
            filepath = os.path.join("./data/chat_logs", filename)
            
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存データを作成
            save_data = {
                "session_start": self.session_start.isoformat(),
                "session_end": datetime.now().isoformat(),
                "question_count": self.question_count,
                "conversation": history
            }
            
            # JSONファイルに保存
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            self.print_colored(f"✅ 会話履歴を保存しました: {filepath}", Colors.OKGREEN)
            
        except Exception as e:
            self.print_colored(f"❌ 保存エラー: {e}", Colors.FAIL)
    
    def show_stats(self):
        """セッション統計を表示"""
        elapsed = datetime.now() - self.session_start
        elapsed_minutes = elapsed.total_seconds() / 60
        
        self.print_colored("\n📊 セッション統計:", Colors.BOLD)
        print("-" * 50)
        print(f"  開始時刻: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  経過時間: {elapsed_minutes:.1f}分")
        print(f"  質問数: {self.question_count}回")
        
        if self.question_count > 0:
            avg_time = elapsed_minutes / self.question_count
            print(f"  平均応答時間: {avg_time:.1f}分/質問")
        
        history = self.rag_chain.get_conversation_history()
        print(f"  会話履歴: {len(history)}メッセージ")
        print("-" * 50)
    
    def process_question(self, question: str):
        """質問を処理して回答を生成"""
        self.question_count += 1
        
        print()  # 改行
        self.print_colored(f"💭 考え中...", Colors.OKCYAN)
        
        try:
            # 回答を生成
            result = self.rag_chain.ask(question, verbose=False)
            self.last_result = result
            
            # 回答を表示
            print()
            self.print_colored("🤖 回答:", Colors.OKGREEN + Colors.BOLD)
            print("-" * 50)
            print(result["answer"])
            print("-" * 50)
            
            # 処理時間を表示
            elapsed = result.get("elapsed_time", 0)
            self.print_colored(f"\n⏱️  処理時間: {elapsed:.2f}秒", Colors.OKCYAN)
            
            # ソース情報を簡易表示
            sources = result.get("sources", [])[:3]
            if sources:
                print("\n📎 参照ソース:")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. [{source['source']}] {source['title']} (スコア: {source['score']:.3f})")
                print(f"\n💡 ソースの詳細は {Colors.OKGREEN}/sources{Colors.ENDC} で確認できます")
            
        except Exception as e:
            self.print_colored(f"\n❌ エラー: {e}", Colors.FAIL)
            import traceback
            if os.getenv("DEBUG", "").lower() == "true":
                traceback.print_exc()
    
    def run(self):
        """メインループ"""
        # ヘッダー表示
        self.print_header()
        
        # RAGチェーンの初期化
        if not self.initialize_rag():
            return
        
        # メインループ
        try:
            while True:
                try:
                    # プロンプト表示と入力取得
                    user_input = input(f"\n{Colors.OKBLUE}👤 You:{Colors.ENDC} ").strip()
                    
                    if not user_input:
                        continue
                    
                    # コマンド処理
                    if user_input.startswith("/"):
                        if not self.process_command(user_input):
                            break
                    else:
                        # 質問処理
                        self.process_question(user_input)
                
                except KeyboardInterrupt:
                    # Ctrl+Cで中断
                    print()
                    self.print_colored("\n⚠️  中断されました", Colors.WARNING)
                    continue
                
        except Exception as e:
            self.print_colored(f"\n❌ 予期しないエラー: {e}", Colors.FAIL)
            import traceback
            if os.getenv("DEBUG", "").lower() == "true":
                traceback.print_exc()
        
        finally:
            # 終了処理
            self.print_colored("\n" + "=" * 70, Colors.OKCYAN)
            self.show_stats()
            
            # 会話履歴の保存を提案
            if self.question_count > 0:
                save_input = input(f"\n💾 会話履歴を保存しますか？ (y/n): ").strip().lower()
                if save_input == "y":
                    self.save_history()
            
            self.print_colored("\n👋 ご利用ありがとうございました！", Colors.BOLD + Colors.OKCYAN)
            self.print_colored("=" * 70 + "\n", Colors.OKCYAN)


def main():
    """メイン関数"""
    cli_chat = CLIChat()
    cli_chat.run()


if __name__ == "__main__":
    main()