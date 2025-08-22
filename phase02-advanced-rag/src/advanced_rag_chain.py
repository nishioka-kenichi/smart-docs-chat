"""
Advanced RAG Chain - 全技術を統合
"""

# 必要なライブラリのインポート
from typing import Dict, List, Optional, Any  # 型ヒント用のライブラリ
from textwrap import dedent  # 複数行文字列のインデント調整用
from dataclasses import dataclass  # データクラス用（設定クラスを簡潔に定義）
import time  # 実行時間計測用
import os  # 環境変数取得用
from dotenv import load_dotenv  # .envファイル読み込み用
from langchain_openai import ChatOpenAI  # OpenAIのLLM
from langchain.prompts import ChatPromptTemplate  # チャット用プロンプトテンプレート
from langchain.schema import Document  # LangChainのドキュメント型
from loguru import logger  # ログ出力用

# Advanced RAGの各コンポーネントをインポート
from hyde import HyDE  # HyDE（仮想文書生成）モジュール
from rag_fusion import RAGFusion  # RAG-Fusion（複数クエリ生成）モジュール
from reranker import get_reranker  # Reranker（再順位付け）モジュール
import config  # 設定ファイル

# .envファイルから環境変数を読み込む
load_dotenv()


@dataclass  # データクラスで設定を管理
class AdvancedRAGConfig:
    """Advanced RAGの設定クラス：各コンポーネントのパラメータを一元管理"""

    use_hyde: bool = True  # HyDEを使用するかどうか
    use_fusion: bool = True  # RAG-Fusionを使用するかどうか
    use_reranker: bool = False  # Rerankerを使用するかどうか
    hyde_num_hypothetical: int = 2  # HyDEで生成する仮想回答数
    fusion_num_queries: int = 4  # RAG-Fusionで生成するクエリ数
    retrieval_k: int = 20  # 初期検索で取得する文書数
    final_k: int = 5  # 最紂的に使用する文書数
    reranker_type: str = "auto"  # Rerankerのタイプ


class AdvancedRAGChain:
    """すべてのAdvanced RAG技術を統合したチェーン"""

    def __init__(self, config: Optional[AdvancedRAGConfig] = None, llm: Optional[ChatOpenAI] = None):
        """
        初期化

        Args:
            config: 設定オブジェクト
            llm: 言語モデル
        """
        # 設定オブジェクトを初期化（指定されなければデフォルト設定を使用）
        self.config = config or AdvancedRAGConfig()
        # LLMを初期化（指定されなければ設定ファイルから読み込む）
        import config as cfg  # configモジュールをインポート

        self.llm = llm or ChatOpenAI(
            api_key=cfg.OPENAI_API_KEY,  # APIキー
            model=cfg.LLM_MODEL,  # モデル名
            temperature=cfg.LLM_TEMPERATURE,  # 生成の多様性
        )
        # Advanced RAGの各コンポーネントを初期化
        logger.info("Initializing Advanced RAG components...")

        # HyDEコンポーネントの初期化
        self.hyde = None
        if self.config.use_hyde:  # HyDEを使用する設定の場合
            try:
                self.hyde = HyDE()  # HyDEインスタンスを作成
                logger.info("HyDE initialized")
            except Exception as e:
                # 初期化に失敗した場合は警告を出して無効化
                logger.warning(f"HyDE initialization failed: {e}")
                self.config.use_hyde = False

        # RAG-Fusionコンポーネントの初期化
        self.fusion = None
        if self.config.use_fusion:  # RAG-Fusionを使用する設定の場合
            try:
                self.fusion = RAGFusion()  # RAG-Fusionインスタンスを作成
                logger.info("RAG-Fusion initialized")
            except Exception as e:
                # 初期化に失敗した場合は警告を出して無効化
                logger.warning(f"RAG-Fusion initialization failed: {e}")
                self.config.use_fusion = False

        # Rerankerコンポーネントの初期化
        self.reranker = None
        if self.config.use_reranker:  # Rerankerを使用する設定の場合
            try:
                self.reranker = get_reranker(self.config.reranker_type)  # Rerankerインスタンスを取得
                logger.info(f"Reranker initialized: {type(self.reranker).__name__}")
            except Exception as e:
                # 初期化に失敗した場合は警告を出して無効化
                logger.warning(f"Reranker initialization failed: {e}")
                self.config.use_reranker = False

        # 最低限一つの検索コンポーネントが必要（HyDEまたはRAG-Fusion）
        if not (self.hyde or self.fusion):
            raise ValueError("At least one retrieval method (HyDE or RAG-Fusion) must be available")

        # 最終回答生成用のプロンプトテンプレート
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                # システムメッセージ：AIアシスタントの役割と指示を定義
                (
                    "system",
                    dedent(
                        """
                あなたは親切で知識豊富なAIアシスタントです。
                与えられたコンテキストを使用して、ユーザーの質問に正確かつ詳細に答えてください。

                重要な指示：
                1. コンテキストに基づいて回答する
                2. コンテキストにない情報は「情報が見つかりません」と明記
                3. 技術的な内容は正確に、初心者にもわかりやすく説明
                4. 必要に応じて例や具体例を含める
                5. 日本語で回答する
            """
                    ).strip(),
                ),
                # ユーザーメッセージ：コンテキストと質問を挿入
                (
                    "user",
                    dedent(
                        """
                コンテキスト:
                {context}

                質問: {question}

                回答:
            """
                    ).strip(),
                ),
            ]
        )

    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Advanced RAG技術を使用して文書を検索

        Args:
            query: 検索クエリ

        Returns:
            検索された文書のリスト
        """
        # 検索結果を格納するデータ構造を初期化
        all_docs = []  # すべての文書を格納するリスト
        doc_scores = {}  # 文書ごとのスコアとソースを記録する辞書

        # ステップ1: HyDEによる検索（仮想文書を生成して検索）
        if self.config.use_hyde and self.hyde:
            logger.info("Executing HyDE search...")
            try:
                # HyDE検索を実行
                hyde_results = self.hyde.search_with_hyde(
                    query,  # 検索クエリ
                    k=self.config.retrieval_k,  # 取得する文書数
                    num_hypothetical=self.config.hyde_num_hypothetical,  # 仮想回答数
                )
                # 検索結果を統合リストに追加
                for doc, score in hyde_results["results"]:
                    doc_key = hash(doc.page_content)  # 文書の一意キーを生成
                    if doc_key not in doc_scores:
                        doc_scores[doc_key] = []  # スコアリストを初期化
                        all_docs.append(doc)  # 文書を追加
                    doc_scores[doc_key].append(("hyde", score))  # HyDEのスコアを記録
            except Exception as e:
                logger.error(f"HyDE search failed: {e}")

        # ステップ2: RAG-Fusionによる検索（複数クエリを生成して統合）
        if self.config.use_fusion and self.fusion:
            logger.info("Executing RAG-Fusion search...")
            try:
                # RAG-Fusion検索を実行
                fusion_results = self.fusion.search_with_fusion(
                    query,  # 検索クエリ
                    k=self.config.retrieval_k,  # 取得する文書数
                    num_queries=self.config.fusion_num_queries,  # 生成するクエリ数
                )
                # 検索結果を統合リストに追加
                for doc, score in fusion_results["results"]:
                    doc_key = hash(doc.page_content)  # 文書の一意キーを生成
                    if doc_key not in doc_scores:
                        doc_scores[doc_key] = []  # スコアリストを初期化
                        all_docs.append(doc)  # 文書を追加
                    doc_scores[doc_key].append(("fusion", score))  # RAG-Fusionのスコアを記録
            except Exception as e:
                logger.error(f"RAG-Fusion search failed: {e}")

        # 重複文書を除去（ハッシュ値で判定）
        unique_docs = {}  # ユニークな文書を格納する辞書
        for doc in all_docs:
            doc_key = hash(doc.page_content)  # 文書内容のハッシュ値を計算
            if doc_key not in unique_docs:
                unique_docs[doc_key] = doc  # まだ登録されていない文書を追加

        # 辞書からリストに変換
        documents = list(unique_docs.values())
        logger.info(f"Retrieved {len(documents)} unique documents")

        # ステップ3: Rerankerによる再順位付け（有効な場合）
        if self.config.use_reranker and self.reranker and documents:
            logger.info("Executing reranking...")
            try:
                # Rerankerで再順位付け
                reranked_results = self.reranker.rerank(
                    query,  # 検索クエリ
                    documents,  # 検索された文書
                    top_k=self.config.final_k,  # 最終的に使用する文書数
                )
                # 再順位付けされた文書のみを取り出す
                documents = [doc for doc, score in reranked_results]
                logger.info(f"Reranked to {len(documents)} documents")
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                # 失敗した場合は元の順序を維持
                documents = documents[: self.config.final_k]
        else:
            # Rerankerを使用しない場合は単純に上位を取得
            documents = documents[: self.config.final_k]

        return documents

    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        検索された文書を基に回答を生成

        Args:
            query: 質問
            documents: 検索された文書

        Returns:
            生成された回答
        """
        # 文書がない場合のデフォルトメッセージ
        if not documents:
            return "申し訳ございません。質問に関連する情報が見つかりませんでした。"

        # 検索された文書からコンテキストを作成
        context = "\n\n---\n\n".join(
            [  # 文書間の区切り文字
                f"[文書 {i+1}]\n{doc.page_content}" for i, doc in enumerate(documents)  # 文書番号と内容
            ]
        )

        # コンテキストの文字数制限（LLMのトークン制限対策）
        max_context_length = 8000  # 最大文字数
        if len(context) > max_context_length:
            # 長すぎる場合は切り詰め
            context = context[:max_context_length] + "\n\n[以下省略...]"

        # LLMで回答を生成
        chain = self.answer_prompt | self.llm  # プロンプトとLLMを連結
        response = chain.invoke(
            {"context": context, "question": query}  # 検索された文書のコンテキスト  # ユーザーの質問
        )

        return response.content  # 生成された回答を返す

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        エンドツーエンドの処理を実行

        Args:
            query: ユーザーの質問

        Returns:
            回答と統計情報
        """
        # 処理開始時間を記録
        start_time = time.time()

        # ステップ1: 文書検索を実行
        logger.info(f"Processing query: {query}")
        documents = self.retrieve_documents(query)
        retrieval_time = time.time() - start_time  # 検索時間を計測

        # ステップ2: 回答生成を実行
        answer = self.generate_answer(query, documents)
        total_time = time.time() - start_time  # 合計時間を計測

        # 結果と統計情報を辞書形式で返す
        return {
            "query": query,  # 元の質問
            "answer": answer,  # 生成された回答
            "source_documents": documents,  # 参照した文書
            "num_sources": len(documents),  # 文書数
            "retrieval_time": retrieval_time,  # 検索時間
            "total_time": total_time,  # 合計処理時間
            "config": {  # 使用した設定
                "use_hyde": self.config.use_hyde,
                "use_fusion": self.config.use_fusion,
                "use_reranker": self.config.use_reranker,
                "reranker_type": self.config.reranker_type if self.config.use_reranker else "none",
            },
        }

    def format_response(self, response: Dict[str, Any]) -> str:
        """
        レスポンスを見やすくフォーマット

        Args:
            response: invokeの返り値

        Returns:
            フォーマットされた文字列
        """
        # 出力用の文字列リストを初期化
        output = []
        # ヘッダー部分
        output.append("\n" + "=" * 60)
        output.append("🤖 Advanced RAG Response")
        output.append("=" * 60)

        # 質問と回答を表示
        output.append(f"\n❓ 質問: {response['query']}")
        output.append(f"\n💡 回答:")
        output.append(response["answer"])

        # 参照文書数を表示
        output.append(f"\n📚 参照した文書: {response['num_sources']}件")

        # ソース文書の抜粋を表示（最大3件）
        if response["num_sources"] > 0:
            output.append("\n📄 ソース文書（抜粋）:")
            for i, doc in enumerate(response["source_documents"][:3], 1):
                output.append(f"\n[文書 {i}]")
                output.append(doc.page_content[:200] + "...")  # 最初の200文字を表示

        # 使用した技術を表示
        output.append(f"\n⚙️ 使用した技術:")
        config = response["config"]
        if config["use_hyde"]:
            output.append("  ✅ HyDE (Hypothetical Document Embeddings)")
        if config["use_fusion"]:
            output.append("  ✅ RAG-Fusion")
        if config.get("use_reranker", False):
            output.append(f"  ✅ Reranker ({config.get('reranker_type', 'none')})")

        # パフォーマンス情報を表示
        output.append(f"\n⏱️ パフォーマンス:")
        output.append(f"  - 検索時間: {response['retrieval_time']:.2f}秒")
        output.append(f"  - 合計時間: {response['total_time']:.2f}秒")

        # フッター
        output.append("\n" + "=" * 60)

        # リストを改行で結合して文字列として返す
        return "\n".join(output)


# スクリプトが直接実行された場合のテストコード
if __name__ == "__main__":
    # Advanced RAGの設定をカスタマイズ
    test_config = AdvancedRAGConfig(
        use_hyde=True,  # HyDEを有効化
        use_fusion=True,  # RAG-Fusionを有効化
        use_reranker=config.RERANKER_ENABLED,  # Rerankerを設定ファイルに従って設定
        hyde_num_hypothetical=2,  # 仮想回答2つ
        fusion_num_queries=3,  # クエリ3つ生成
        retrieval_k=15,  # 初期検索15件
        final_k=5,  # 最終的に5件使用
        reranker_type=config.RERANKER_TYPE,  # Rerankerタイプ
    )

    # Advanced RAGチェーンの作成とテスト実行
    try:
        rag_chain = AdvancedRAGChain(config=test_config)  # チェーンを初期化

        # テスト用の質問リスト
        test_queries = [
            "スマートグラフの有給休暇は3年目だと何日ある？",
            "その月の残業時間はどのような計算式で取得できる？2025年9月の場合はどうなる？",
            "ChromaDBのインデックス構築のベストプラクティス",
        ]

        # 各質問に対してAdvanced RAGを実行
        for query in test_queries:
            print(f"\n{'🔍'*30}")  # 視覚的な区切り線
            response = rag_chain.invoke(query)  # RAG処理を実行
            print(rag_chain.format_response(response))  # 結果を整形して表示

    except Exception as e:
        # 初期化エラーの場合のエラーハンドリング
        logger.error(f"Failed to initialize Advanced RAG Chain: {e}")
        print(f"\n❌ エラー: {e}")
        print("\n💡 ヒント:")
        print("1. Phase 1のChromaDBが存在することを確認してください")
        print("2. 環境変数（OPENAI_API_KEY等）が設定されていることを確認してください")
        print("3. requirements.txtのパッケージがインストールされていることを確認してください")
