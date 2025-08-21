#!/usr/bin/env python3
"""
スマートドキュメントRAGチャット - メインエントリーポイント

このスクリプトは、RAGシステムの各コンポーネントを統合的に実行するための
エントリーポイントです。

使用方法:
    python main.py [オプション]

オプション:
    setup    - 初期セットアップ（ドキュメント取得とインデックス作成）
    chat     - チャットインターフェースを起動
    update   - ドキュメントの更新とインデックスの再構築
    test     - システムのテスト実行
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

# プロジェクトのsrcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# 環境変数の読み込み
load_dotenv()


def print_header(title: str):
    """ヘッダーを表示"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_environment():
    """環境設定をチェック"""
    print_header("🔍 環境チェック")
    
    required_vars = [
        "OPENAI_API_KEY",
        "NOTION_API_KEY",
    ]
    
    optional_vars = [
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        "GOOGLE_DRIVE_FOLDER_ID",
    ]
    
    missing_required = []
    missing_optional = []
    
    # 必須環境変数のチェック
    for var in required_vars:
        if os.getenv(var):
            print(f"  ✅ {var}: 設定済み")
        else:
            print(f"  ❌ {var}: 未設定")
            missing_required.append(var)
    
    # オプション環境変数のチェック
    for var in optional_vars:
        if os.getenv(var):
            print(f"  ✅ {var}: 設定済み")
        else:
            print(f"  ⚠️  {var}: 未設定（オプション）")
            missing_optional.append(var)
    
    if missing_required:
        print(f"\n❌ 必須環境変数が設定されていません: {', '.join(missing_required)}")
        print("   .envファイルを確認してください")
        return False
    
    if missing_optional:
        print(f"\n💡 Google Drive連携を使用する場合は以下を設定してください:")
        for var in missing_optional:
            print(f"   - {var}")
    
    return True


def test_connections():
    """API接続テスト"""
    print_header("🔌 API接続テスト")
    
    try:
        from test_connections import main as test_main
        test_main()
        return True
    except Exception as e:
        print(f"\n❌ 接続テストエラー: {e}")
        return False


def load_documents():
    """ドキュメントの取得"""
    print_header("📥 ドキュメント取得")
    
    success = True
    
    # Notionドキュメントの取得
    try:
        print("\n📘 Notionドキュメントを取得中...")
        from data_loader_notion import main as notion_main
        notion_main()
        print("  ✅ Notion取得完了")
    except Exception as e:
        print(f"  ❌ Notionエラー: {e}")
        success = False
    
    # Google Driveドキュメントの取得
    if os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"):
        try:
            print("\n📗 Google Driveドキュメントを取得中...")
            from data_loader_google import main as google_main
            google_main()
            print("  ✅ Google Drive取得完了")
        except Exception as e:
            print(f"  ❌ Google Driveエラー: {e}")
            success = False
    else:
        print("\n⚠️  Google Drive連携はスキップされました（認証情報未設定）")
    
    return success


def create_index():
    """インデックスの作成"""
    print_header("🔄 インデックス作成")
    
    try:
        from indexer import main as indexer_main
        indexer_main()
        return True
    except Exception as e:
        print(f"\n❌ インデックス作成エラー: {e}")
        return False


def run_chat():
    """チャットインターフェースの起動"""
    try:
        from cli_chat import main as chat_main
        chat_main()
    except Exception as e:
        print(f"\n❌ チャットエラー: {e}")


def test_rag():
    """RAGシステムのテスト"""
    print_header("🧪 RAGシステムテスト")
    
    try:
        from rag_chain import main as rag_test
        rag_test()
        return True
    except Exception as e:
        print(f"\n❌ RAGテストエラー: {e}")
        return False


def setup():
    """初期セットアップ"""
    print_header("🚀 初期セットアップ開始")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 環境チェック
    if not check_environment():
        return False
    
    # 2. API接続テスト
    print("\n" + "-" * 70)
    if not test_connections():
        print("\n⚠️  API接続に問題があります。続行しますか？ (y/n): ", end="")
        if input().strip().lower() != "y":
            return False
    
    # 3. ドキュメント取得
    print("\n" + "-" * 70)
    if not load_documents():
        print("\n⚠️  ドキュメント取得に問題がありました")
    
    # 4. インデックス作成
    print("\n" + "-" * 70)
    if not create_index():
        return False
    
    print_header("✅ セットアップ完了")
    print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 チャットを開始するには以下を実行してください:")
    print("   python main.py chat")
    
    return True


def update():
    """ドキュメントとインデックスの更新"""
    print_header("🔄 ドキュメント更新")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. ドキュメント取得
    if not load_documents():
        print("\n⚠️  ドキュメント取得に問題がありました")
    
    # 2. インデックス更新
    print("\n" + "-" * 70)
    if not create_index():
        return False
    
    print_header("✅ 更新完了")
    print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="スマートドキュメントRAGチャット",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  初期セットアップ:
    python main.py setup
  
  チャット開始:
    python main.py chat
  
  ドキュメント更新:
    python main.py update
  
  システムテスト:
    python main.py test
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="chat",
        choices=["setup", "chat", "update", "test"],
        help="実行するコマンド (デフォルト: chat)"
    )
    
    args = parser.parse_args()
    
    # コマンドの実行
    if args.command == "setup":
        success = setup()
        sys.exit(0 if success else 1)
    
    elif args.command == "chat":
        # インデックスの存在確認
        chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chromadb")
        if not os.path.exists(chroma_dir):
            print("⚠️  インデックスが見つかりません")
            print("   初期セットアップを実行しますか？ (y/n): ", end="")
            if input().strip().lower() == "y":
                if setup():
                    print("\n" + "-" * 70)
                    run_chat()
            else:
                print("\n💡 手動でセットアップを実行してください:")
                print("   python main.py setup")
        else:
            run_chat()
    
    elif args.command == "update":
        success = update()
        sys.exit(0 if success else 1)
    
    elif args.command == "test":
        print_header("🧪 システムテスト")
        
        # 環境チェック
        if not check_environment():
            sys.exit(1)
        
        print("\n" + "-" * 70)
        # API接続テスト
        if not test_connections():
            sys.exit(1)
        
        print("\n" + "-" * 70)
        # RAGテスト
        if not test_rag():
            sys.exit(1)
        
        print_header("✅ すべてのテストが成功しました")


if __name__ == "__main__":
    main()