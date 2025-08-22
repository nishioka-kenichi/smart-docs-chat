"""
Phase 2 Advanced RAG 設定管理
シンプルで明確な設定の一元管理
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# プロジェクトルートの決定
PROJECT_ROOT = Path(__file__).parent.parent

# .envファイルの読み込み（APIキーなどの秘匿情報）
load_dotenv(PROJECT_ROOT / ".env")

# settings.yamlの読み込み（アプリケーション設定）
with open(PROJECT_ROOT / "config" / "settings.yaml", 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ============================================
# APIキー（環境変数から取得）
# ============================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================
# モデル設定（settings.yamlから取得）
# ============================================
LLM_MODEL = settings["models"]["llm"]["model"]
LLM_TEMPERATURE = settings["models"]["llm"]["temperature"]
LLM_MAX_TOKENS = settings["models"]["llm"]["max_tokens"]
EMBEDDING_MODEL = settings["models"]["embedding"]["model"]

# ============================================
# データベース設定
# ============================================
CHROMADB_PATH = settings["database"]["chromadb"]["persist_directory"]
CHROMADB_COLLECTION = settings["database"]["chromadb"]["collection_name"]

# ============================================
# Advanced RAGコンポーネント設定
# ============================================
# HyDE設定
HYDE_ENABLED = settings["components"]["hyde"]["enabled"]
HYDE_NUM_HYPOTHETICAL = settings["components"]["hyde"]["num_hypothetical"]
HYDE_TEMPERATURE = settings["components"]["hyde"]["temperature"]
HYDE_MAX_LENGTH = settings["components"]["hyde"]["max_length"]

# RAG-Fusion設定
FUSION_ENABLED = settings["components"]["fusion"]["enabled"]
FUSION_NUM_QUERIES = settings["components"]["fusion"]["num_queries"]
FUSION_RRF_K = settings["components"]["fusion"]["rrf_k"]

# ============================================
# 検索設定
# ============================================
RETRIEVAL_INITIAL_K = settings["retrieval"]["initial_k"]
RETRIEVAL_FINAL_K = settings["retrieval"]["final_k"]

# ============================================
# ロギング設定
# ============================================
LOG_LEVEL = settings.get("logging", {}).get("level", "INFO")
LOG_FILE = settings.get("logging", {}).get("file", "logs/advanced_rag.log")

# ============================================
# 設定検証
# ============================================
def validate_config():
    """必須設定の検証"""
    errors = []
    
    # APIキーの検証
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set in .env")
    
    # ChromaDBパスの検証
    chromadb_full_path = PROJECT_ROOT / CHROMADB_PATH
    if not chromadb_full_path.exists():
        errors.append(f"ChromaDB not found at: {chromadb_full_path}")
    
    if errors:
        for error in errors:
            print(f"❌ {error}")
        return False
    
    return True

# 設定情報の表示（デバッグ用）
def print_config():
    """現在の設定を表示"""
    print("\n" + "="*60)
    print("📋 Phase 2 Advanced RAG Configuration")
    print("="*60)
    print(f"\n🔑 API Keys:")
    print(f"  - OpenAI: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
    
    print(f"\n🤖 Models:")
    print(f"  - LLM: {LLM_MODEL}")
    print(f"  - Embedding: {EMBEDDING_MODEL}")
    
    print(f"\n💾 Database:")
    print(f"  - ChromaDB: {CHROMADB_PATH}")
    print(f"  - Collection: {CHROMADB_COLLECTION}")
    
    print(f"\n🔧 Components:")
    print(f"  - HyDE: {'✅ Enabled' if HYDE_ENABLED else '❌ Disabled'}")
    print(f"  - RAG-Fusion: {'✅ Enabled' if FUSION_ENABLED else '❌ Disabled'}")
    
    print(f"\n📊 Retrieval:")
    print(f"  - Initial K: {RETRIEVAL_INITIAL_K}")
    print(f"  - Final K: {RETRIEVAL_FINAL_K}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # 設定のテスト
    print_config()
    if validate_config():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors")