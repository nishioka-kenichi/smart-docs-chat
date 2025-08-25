"""
エージェント用ツール定義モジュール

LangGraphエージェントが使用する各種ツールを定義します。
RAG検索、Web検索、計算、ファイル操作などの機能を提供し、
エージェントがタスクを自律的に実行できるようにします。

主な機能:
1. RAG検索: Phase 1のChromaDBから関連文書を検索
2. Web検索: インターネットから最新情報を取得（Tavily API使用）
3. 計算機: 数式評価と数学演算
4. ファイル操作: テキストファイルの読み書き
"""

# 標準ライブラリのインポート
import os  # 環境変数へのアクセス
import json  # JSON形式のデータ処理
import yaml  # YAML設定ファイルの読み込み
from typing import Any, Dict, List, Optional  # 型ヒント用
from datetime import datetime  # タイムスタンプ記録用
from pathlib import Path  # ファイルパス操作用
from dotenv import load_dotenv  # .envファイルの読み込み

# LangChain関連のインポート
from langchain_core.tools import Tool, StructuredTool  # ツールの基底クラス
from langchain_community.vectorstores import Chroma  # ChromaDBベクトルストア
from langchain_openai import OpenAIEmbeddings  # OpenAI埋め込みモデル
from langchain_core.documents import Document  # ドキュメントクラス

# Pydanticモデル（入力検証用）
from pydantic import BaseModel, Field  # 型安全な入力スキーマ定義

# ChromaDB関連
import chromadb  # ChromaDBクライアント
from chromadb.config import Settings  # ChromaDB設定


class RAGSearchInput(BaseModel):
    """
    RAG検索ツールの入力スキーマ

    社内ドキュメントベクトルDBからの検索パラメータを定義します。
    """

    query: str = Field(description="検索クエリ")
    top_k: int = Field(default=5, description="取得する文書数")
    score_threshold: float = Field(default=0.3, description="最小スコア閾値")


class WebSearchInput(BaseModel):
    """
    Web検索ツールの入力スキーマ

    インターネットからの情報検索パラメータを定義します。
    """

    query: str = Field(description="検索クエリ")
    max_results: int = Field(default=3, description="最大結果数")


class CalculatorInput(BaseModel):
    """
    計算ツールの入力スキーマ

    数式評価のパラメータを定義します。
    """

    expression: str = Field(description="計算式（Pythonの式として評価）")


class FileReadInput(BaseModel):
    """
    ファイル読み込みツールの入力スキーマ

    ファイル読み込みのパラメータを定義します。
    """

    file_path: str = Field(description="読み込むファイルパス")
    encoding: str = Field(default="utf-8", description="文字エンコーディング")


class FileWriteInput(BaseModel):
    """
    ファイル書き込みツールの入力スキーマ

    ファイル書き込みのパラメータを定義します。
    """

    file_path: str = Field(description="書き込むファイルパス")
    content: str = Field(description="書き込む内容")
    mode: str = Field(default="w", description="書き込みモード（w: 上書き, a: 追記）")


class ToolsManager:
    """
    ツール管理クラス

    エージェントが使用可能な全ツールを管理し、
    設定に基づいて有効化・無効化を制御します。
    """

    def __init__(self, config_path: str = "./config/settings.yaml"):
        """
        ツールマネージャーの初期化

        設定ファイルを読み込み、各ツールのセットアップを行います。

        Args:
            config_path: 設定ファイルパス（YAML形式）
        """
        # .envファイルを読み込み（環境変数の設定）
        load_dotenv()

        self.config = self._load_config(config_path)  # 設定読み込み
        self.tools = []  # 登録されたツールのリスト
        self.vectorstore = None  # RAG検索用のベクトルストア
        self._initialize_tools()  # ツールの初期化

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込み

        YAML形式の設定ファイルから設定を読み込みます。

        Args:
            config_path: 設定ファイルのパス

        Returns:
            設定の辞書
        """
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _initialize_tools(self):
        """
        ツールを初期化

        設定ファイルに基づいて、各ツールの有効/無効を判定し、
        有効なツールのみをセットアップします。
        """
        tools_config = self.config.get("tools", {})

        # RAG検索ツール（Phase 1のChromaDBを使用）
        if tools_config.get("rag_search", {}).get("enabled", True):
            self._setup_rag_search()

        # Web検索ツール（Tavily APIを使用）
        if tools_config.get("web_search", {}).get("enabled", False):
            self._setup_web_search()

        # 計算ツール（数式評価）
        if tools_config.get("calculator", {}).get("enabled", True):
            self._setup_calculator()

        # ファイル操作ツール（読み書き）
        if tools_config.get("file_handler", {}).get("enabled", True):
            self._setup_file_handler()

    def _setup_rag_search(self):
        """
        RAG検索ツールをセットアップ

        Phase 1で作成済みのChromaDBベクトルストアに接続し、
        社内ドキュメント検索機能を提供します。
        """
        try:
            # ChromaDBクライアントの初期化
            chromadb_path = os.getenv("CHROMADB_PATH", "../phase01-local/data/chromadb")

            # Phase 1のChromaDBに接続
            client = chromadb.PersistentClient(
                path=chromadb_path, settings=Settings(allow_reset=False, anonymized_telemetry=False)
            )

            # 埋め込みモデルの初期化（Phase 1と同じモデルを使用）
            embeddings = OpenAIEmbeddings(
                model=self.config.get("chromadb", {}).get("embedding_model", "text-embedding-3-small")
            )

            # Vectorstoreの初期化
            collection_name = self.config.get("chromadb", {}).get("collection_name", "phase01_documents")
            self.vectorstore = Chroma(
                client=client, collection_name=collection_name, embedding_function=embeddings
            )

            # ツールを作成して登録
            tool = StructuredTool.from_function(
                func=self._rag_search,
                name="rag_search",
                description="社内ドキュメントを検索し、関連情報を取得する",
                args_schema=RAGSearchInput,
            )
            self.tools.append(tool)

        except Exception as e:
            print(f"Warning: RAG検索ツールの初期化に失敗: {e}")

    def _rag_search(self, query: str, top_k: int = 5, score_threshold: float = 0.3) -> str:
        """
        RAG検索を実行

        ChromaDBベクトルストアから類似文書を検索し、
        スコアでフィルタリングして結果を返します。

        Args:
            query: 検索クエリ
            top_k: 取得する文書数
            score_threshold: 最小スコア閾値（0-1の範囲）

        Returns:
            フォーマットされた検索結果
        """
        if not self.vectorstore:
            return "RAG検索ツールは利用できません"

        try:
            # 類似度検索を実行
            results = self.vectorstore.similarity_search_with_score(query=query, k=top_k)

            # スコアでフィルタリング
            filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]

            if not filtered_results:
                return "関連する文書が見つかりませんでした"

            # 結果をフォーマット
            formatted_results = []
            for i, (doc, score) in enumerate(filtered_results, 1):
                formatted_results.append(
                    f"[結果{i}] (スコア: {score:.3f})\n"
                    f"内容: {doc.page_content[:500]}...\n"
                    f"ソース: {doc.metadata.get('source', 'Unknown')}"
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            return f"検索エラー: {str(e)}"

    def _setup_web_search(self):
        """
        Web検索ツールをセットアップ

        Tavily APIを使用してインターネット検索機能を提供します。
        APIキーが設定されていない場合は自動的に無効化されます。
        """
        # Tavily APIキーがある場合のみ有効化
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            print("Warning: TAVILY_API_KEYが設定されていないため、Web検索は無効です")
            return

        try:
            from tavily import TavilyClient

            self.tavily_client = TavilyClient(api_key=tavily_api_key)

            tool = StructuredTool.from_function(
                func=self._web_search,
                name="web_search",
                description="インターネットから最新情報を検索する",
                args_schema=WebSearchInput,
            )
            self.tools.append(tool)

        except ImportError:
            print("Warning: tavily-pythonがインストールされていません")

    def _web_search(self, query: str, max_results: int = 3) -> str:
        """
        Web検索を実行

        Tavily APIを使用してインターネットから情報を検索します。

        Args:
            query: 検索クエリ
            max_results: 取得する最大結果数

        Returns:
            フォーマットされた検索結果
        """
        if not hasattr(self, "tavily_client"):
            return "Web検索ツールは利用できません"

        try:
            results = self.tavily_client.search(query=query, max_results=max_results)

            formatted_results = []
            for i, result in enumerate(results.get("results", []), 1):
                formatted_results.append(
                    f"[結果{i}]\n"
                    f"タイトル: {result.get('title', 'N/A')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                    f"概要: {result.get('content', 'N/A')[:300]}..."
                )

            return "\n\n".join(formatted_results) if formatted_results else "検索結果が見つかりませんでした"

        except Exception as e:
            return f"Web検索エラー: {str(e)}"

    def _setup_calculator(self):
        """
        計算ツールをセットアップ

        数式評価と数学演算機能を提供します。
        安全な評価のため、許可された関数のみを使用可能にします。
        """
        tool = StructuredTool.from_function(
            func=self._calculate,
            name="calculator",
            description="数式を計算する（Python式として評価）",
            args_schema=CalculatorInput,
        )
        self.tools.append(tool)

    def _calculate(self, expression: str) -> str:
        """
        計算を実行

        Pythonのeval関数を使用して数式を評価します。
        セキュリティのため、許可された関数のみを使用可能にしています。

        Args:
            expression: 計算式（Python式として評価可能な文字列）

        Returns:
            計算結果を含む文字列
        """
        try:
            # 安全な評価のために許可する関数を制限
            allowed_names = {
                "abs": abs,
                "round": round,
                "pow": pow,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "int": int,
                "float": float,
            }

            # 数学関数を追加
            import math

            for name in ["sin", "cos", "tan", "sqrt", "log", "exp", "pi", "e"]:
                if hasattr(math, name):
                    allowed_names[name] = getattr(math, name)

            # 式を評価
            result = eval(expression, {"__builtins__": {}}, allowed_names)

            # 精度設定
            precision = self.config.get("tools", {}).get("calculator", {}).get("precision", 4)
            if isinstance(result, float):
                result = round(result, precision)

            return f"計算結果: {expression} = {result}"

        except Exception as e:
            return f"計算エラー: {str(e)}"

    def _setup_file_handler(self):
        """
        ファイル操作ツールをセットアップ

        テキストファイルの読み書き機能を提供します。
        セキュリティのため、許可された拡張子のみを扱います。
        """
        # 読み込みツール
        read_tool = StructuredTool.from_function(
            func=self._read_file,
            name="read_file",
            description="ファイルを読み込む",
            args_schema=FileReadInput,
        )
        self.tools.append(read_tool)

        # 書き込みツール
        write_tool = StructuredTool.from_function(
            func=self._write_file,
            name="write_file",
            description="ファイルに書き込む",
            args_schema=FileWriteInput,
        )
        self.tools.append(write_tool)

    def _read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        ファイルを読み込む

        指定されたファイルを読み込み、内容を返します。
        大きなファイルは自動的に切り詰められます。

        Args:
            file_path: 読み込むファイルのパス
            encoding: 文字エンコーディング

        Returns:
            ファイル内容（大きい場合は切り詰め）
        """
        try:
            # 許可された拡張子をチェック
            allowed_exts = (
                self.config.get("tools", {})
                .get("file_handler", {})
                .get("allowed_extensions", [".txt", ".md", ".json", ".csv"])
            )

            path = Path(file_path)
            if path.suffix not in allowed_exts:
                return f"エラー: ファイル形式 {path.suffix} は許可されていません"

            if not path.exists():
                return f"エラー: ファイル {file_path} が見つかりません"

            with open(path, "r", encoding=encoding) as f:
                content = f.read()

            # 大きすぎるファイルは切り詰める
            max_length = 5000
            if len(content) > max_length:
                content = content[:max_length] + f"\n... (残り {len(content) - max_length} 文字省略)"

            return f"ファイル内容 ({file_path}):\n{content}"

        except Exception as e:
            return f"ファイル読み込みエラー: {str(e)}"

    def _write_file(self, file_path: str, content: str, mode: str = "w") -> str:
        """
        ファイルに書き込む

        指定されたファイルに内容を書き込みます。
        ディレクトリが存在しない場合は自動作成します。

        Args:
            file_path: 書き込むファイルのパス
            content: 書き込む内容
            mode: 書き込みモード（"w": 上書き、"a": 追記）

        Returns:
            実行結果のメッセージ
        """
        try:
            # 許可された拡張子をチェック
            allowed_exts = (
                self.config.get("tools", {})
                .get("file_handler", {})
                .get("allowed_extensions", [".txt", ".md", ".json", ".csv"])
            )

            path = Path(file_path)
            if path.suffix not in allowed_exts:
                return f"エラー: ファイル形式 {path.suffix} は許可されていません"

            # ディレクトリが存在しない場合は作成
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, mode, encoding="utf-8") as f:
                f.write(content)

            return f"ファイル {file_path} に正常に書き込みました（{len(content)} 文字）"

        except Exception as e:
            return f"ファイル書き込みエラー: {str(e)}"

    def get_tools(self) -> List[Tool]:
        """
        登録されているツールを取得

        エージェントが使用可能な全ツールのリストを返します。

        Returns:
            Toolオブジェクトのリスト
        """
        return self.tools

    def get_tool_names(self) -> List[str]:
        """
        ツール名のリストを取得

        登録されているツールの名前のみを返します。

        Returns:
            ツール名の文字列リスト
        """
        return [tool.name for tool in self.tools]

    def get_tool_descriptions(self) -> Dict[str, str]:
        """
        ツールの説明を取得

        各ツールの名前と説明のマッピングを返します。

        Returns:
            ツール名をキー、説明を値とする辞書
        """
        return {tool.name: tool.description for tool in self.tools}
