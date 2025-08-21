"""
設定管理モジュール
環境変数と設定ファイルを統合して管理
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """設定クラス - 環境変数と設定ファイルを統合管理"""
    
    def __init__(self, env_file: str = ".env", settings_file: str = "config/settings.yaml"):
        """
        Args:
            env_file: 環境変数ファイルのパス
            settings_file: 設定ファイルのパス
        """
        # プロジェクトルートの決定
        self.project_root = Path(__file__).parent.parent
        
        # 環境変数の読み込み
        env_path = self.project_root / env_file
        if env_path.exists():
            load_dotenv(env_path)
        
        # 設定ファイルの読み込み
        settings_path = self.project_root / settings_file
        if settings_path.exists():
            with open(settings_path, 'r', encoding='utf-8') as f:
                self.settings = yaml.safe_load(f)
        else:
            self.settings = {}
        
        # API Keys（環境変数から取得）
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.notion_token = os.getenv('NOTION_TOKEN')
        self.notion_root_page_id = os.getenv('NOTION_ROOT_PAGE_ID')
        
        # Google Drive設定
        self.google_drive_credentials_path = os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH')
        self.google_drive_folder_ids = os.getenv('GOOGLE_DRIVE_FOLDER_IDS', '').split(',')
        
        # システム設定（settings.yamlから取得、環境変数で上書き可能）
        self.chromadb = self._get_chromadb_config()
        self.embedding = self._get_embedding_config()
        self.llm = self._get_llm_config()
        self.chunking = self._get_chunking_config()
        self.retriever = self._get_retriever_config()
    
    def _get_chromadb_config(self) -> Dict[str, Any]:
        """ChromaDB設定を取得"""
        config = self.settings.get('chromadb', {})
        # 環境変数での上書きを許可
        config['persist_directory'] = os.getenv('CHROMA_PERSIST_DIRECTORY', 
                                                config.get('persist_directory', './data/chromadb'))
        config['collection_name'] = os.getenv('CHROMA_COLLECTION_NAME',
                                             config.get('collection_name', 'phase01_documents'))
        return config
    
    def _get_embedding_config(self) -> Dict[str, Any]:
        """埋め込みモデル設定を取得"""
        config = self.settings.get('embedding', {})
        config['model'] = os.getenv('OPENAI_EMBEDDING_MODEL',
                                   config.get('model', 'text-embedding-3-small'))
        return config
    
    def _get_llm_config(self) -> Dict[str, Any]:
        """LLM設定を取得"""
        config = self.settings.get('llm', {})
        config['model'] = os.getenv('OPENAI_MODEL',
                                   config.get('model', 'gpt-4o-mini'))
        config['temperature'] = float(os.getenv('OPENAI_TEMPERATURE',
                                               config.get('temperature', 0.3)))
        config['max_tokens'] = int(os.getenv('OPENAI_MAX_TOKENS',
                                            config.get('max_tokens', 2000)))
        return config
    
    def _get_chunking_config(self) -> Dict[str, Any]:
        """チャンク分割設定を取得"""
        config = self.settings.get('chunking', {})
        config['chunk_size'] = int(os.getenv('CHUNK_SIZE',
                                            config.get('chunk_size', 500)))
        config['chunk_overlap'] = int(os.getenv('CHUNK_OVERLAP',
                                                config.get('chunk_overlap', 100)))
        return config
    
    def _get_retriever_config(self) -> Dict[str, Any]:
        """検索設定を取得"""
        config = self.settings.get('retriever', {})
        config['k'] = int(os.getenv('RETRIEVER_K',
                                   config.get('k', 10)))
        config['score_threshold'] = float(os.getenv('RETRIEVER_SCORE_THRESHOLD',
                                                    config.get('score_threshold', 0.2)))
        return config
    
    def validate(self) -> bool:
        """必須設定の検証"""
        errors = []
        
        # API Keyの検証
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is not set")
        if not self.notion_token:
            errors.append("NOTION_TOKEN is not set")
        if not self.google_drive_credentials_path:
            errors.append("GOOGLE_DRIVE_CREDENTIALS_PATH is not set")
        
        if errors:
            for error in errors:
                print(f"❌ {error}")
            return False
        return True


# シングルトンインスタンス
config = Config()