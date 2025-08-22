"""
RAG-Fusion Implementation

RAG-Fusionはユーザーの質問から複数の検索クエリを生成し、
それぞれの検索結果をReciprocal Rank Fusion (RRF)で統合する技術です。

例：「RAGとは？」という質問に対して
1. 複数のクエリを生成：「RAG 検索拡張生成」「Retrieval Augmented Generation」「検索ベースAI」
2. 各クエリで検索を実行し、異なる観点から文書を収集
3. RRFアルゴリズムで統合し、最も関連性の高い文書を選出
"""

# 必要なライブラリのインポート
from typing import List, Dict, Tuple, Optional  # 型ヒント用のライブラリ
from textwrap import dedent  # 複数行文字列のインデント調整用
from collections import defaultdict  # デフォルト値付き辞書（RRFスコア集計用）
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAIのLLMと埋め込みモデル
from langchain.prompts import PromptTemplate  # プロンプトテンプレート作成用
from langchain_chroma import Chroma  # ChromaDBベクトルストア用クライアント
from langchain.schema import Document  # LangChainのドキュメント型
from pathlib import Path  # ファイルパス操作用
from loguru import logger  # ログ出力用

# 設定をインポート
import config


class RAGFusion:
    """
    RAG-Fusion実装クラス

    このクラスは以下の流れで動作します：
    1. ユーザーの質問を受け取る
    2. LLMを使って異なる観点の複数クエリを生成
    3. 各クエリで並列的に検索を実行
    4. Reciprocal Rank Fusionで結果を統合
    5. 最もスコアの高い文書を返す
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        embeddings: Optional[OpenAIEmbeddings] = None,
        persist_directory: Optional[str] = None,
    ):
        """
        RAG-Fusionシステムの初期化

        このメソッドでは以下のコンポーネントをセットアップします：
        - LLM：複数の検索クエリを生成するためのAIモデル
        - Embeddings：テキストをベクトル化するモデル
        - ChromaDB：Phase 1で作成済みのベクトルデータベース

        Args:
            llm: 言語モデル（指定しない場合はconfigから読み込み）
            embeddings: 埋め込みモデル（指定しない場合はconfigから読み込み）
            persist_directory: ChromaDBの保存先（指定しない場合はconfigから読み込み）
        """
        # LLMの初期化：複数クエリを生成するためのAIモデル
        # 引数で渡されなければ、config.py経由で設定を読み込む
        self.llm = llm or ChatOpenAI(
            api_key=config.OPENAI_API_KEY,  # OpenAIのAPIキー（.envから）
            model=config.LLM_MODEL,  # 使用するモデル（gpt-4o-miniなど）
            temperature=config.LLM_TEMPERATURE,  # 生成の多様性（0.7=バランス良い）
        )

        # 埋め込みモデルの初期化：テキストを数値ベクトルに変換するモデル
        # 生成された複数クエリをベクトル化して検索に使用
        self.embeddings = embeddings or OpenAIEmbeddings(
            api_key=config.OPENAI_API_KEY,  # OpenAIのAPIキー
            model=config.EMBEDDING_MODEL,  # text-embedding-3-smallなど
        )

        # ChromaDBの初朞化：Phase 1で構築済みのベクトルデータベースに接続
        # 複数のクエリで検索した結果を統合するためのDB
        if persist_directory:
            chromadb_path = Path(persist_directory).resolve()
        else:
            chromadb_path = config.PROJECT_ROOT / config.CHROMADB_PATH

        # ChromaDBディレクトリの存在確認
        # Phase 1を実行していない場合はエラーになる
        if not chromadb_path.exists():
            raise FileNotFoundError(f"ChromaDB not found at: {chromadb_path}")

        # ChromaDBの読み込み状況をログに記録（デバッグ用）
        logger.info(f"Loading ChromaDB from: {chromadb_path}")

        # Chromaベクトルストアを初期化して既存のDBに接続
        # ここで接続したDBを使って、複数クエリの検索を実行
        self.vectorstore = Chroma(
            persist_directory=str(chromadb_path),  # ChromaDBの保存場所
            embedding_function=self.embeddings,  # 検索時に使う埋め込みモデル
            collection_name=config.CHROMADB_COLLECTION,  # phase01_documentsコレクション
        )

        # RAG-Fusion用のプロンプトテンプレート：複数クエリを生成するための指示文
        # このプロンプトがRAG-Fusionの核心部分！
        # LLMに「異なる観点からの検索クエリ」を生成させる
        self.query_generation_prompt = PromptTemplate(
            input_variables=[
                "original_query",
                "num_queries",
            ],  # {original_query}と{num_queries}が入る
            template=dedent(
                """
                元の質問から、異なる観点や言い換えを含む{num_queries}個の検索クエリを生成してください。

                要件：
                - それぞれ異なる概側面から質問を捉える
                - 専門用語と一般的な表現の両方を含む
                - 具体的かつ検索しやすい形式
                - 類義語や関連念を活用

                元の質問: {original_query}

                検索クエリ（1行に1つ、番号付きで）:
            """
            ).strip(),
        )

    def generate_queries(self, original_query: str, num_queries: int = 5) -> List[str]:
        """
        複数の検索クエリを生成する（RAG-Fusionのステップ1）

        このメソッドはユーザーの質問を「異なる観点」から捉え直し、
        複数の検索クエリを生成します。複数生成する理由は：
        1. 表現の多様性：同じ意味でも異なる表現で検索
        2. 観点の多様性：技術的/一般的/具体的/抽象的など
        3. カバレッジの向上：より幅広い関連文書を収集

        Args:
            original_query: ユーザーの元の質問（例：「RAGとは？」）
            num_queries: 生成するクエリ数（デフォルト5つ）

        Returns:
            クエリのリスト（元のクエリを含む）
        """
        # プロンプトテンプレートとLLMを連結（LangChainのパイプ機能）
        # プロンプト | LLM という記法でチェーンを作成
        chain = self.query_generation_prompt | self.llm

        # LLMにクエリ生成を依頼
        # LLMが「この質問を異なる角度から表現すると...」と考えて生成
        response = chain.invoke(
            {
                "original_query": original_query,  # 元の質問
                "num_queries": num_queries,  # 生成するクエリ数
            }
        )

        # LLMのレスポンスをパースしてクエリリストを抽出
        # LLMは「1. クエリ1\n2. クエリ2\n...」の形式で返す
        queries = []

        # レスポンスを改行で分割して各行を処理
        for line in response.content.split("\n"):
            line = line.strip()  # 前後の空白を除去

            # 番号付きリスト（"1. クエリ"）または箇条書き（"- クエリ"）を検出
            if line and (line[0].isdigit() or line[0] == "-"):
                # "1. クエリ" -> "クエリ" のように番号を除去
                query = line.split(".", 1)[-1].strip()  # ピリオドで分割
                query = query.lstrip("- ").strip()  # ハイフンも除去

                if query:  # 空文字列でなければリストに追加
                    queries.append(query)

        # 元のクエリをリストの先頭に追加
        # 重要：元の質問も検索に使う（ユーザーの意図を最も正確に表現）
        queries.insert(0, original_query)

        # 生成されたクエリ数をログに記録（デバッグや監視用）
        logger.info(f"Generated {len(queries)} queries")

        # 指定数+1個（元のクエリ含む）までに制限して返す
        return queries[: num_queries + 1]

    def reciprocal_rank_fusion(
        self, results_dict: Dict[str, List[Tuple[Document, float]]], k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Reciprocal Rank Fusionで複数の検索結果を統合（RAG-Fusionのコアアルゴリズム）

        RRFは複数のランキングを統合するためのシンプルかつ効果的な方法です。
        スコア = Σ(1 / (rank + k)) という式で計算します。

        例：ある文書が
        - クエリ1で1位 → 1/(1+60) = 0.0164
        - クエリ2で3位 → 1/(3+60) = 0.0159
        - 合計RRFスコア = 0.0323

        Args:
            results_dict: クエリごとの検索結果の辞書
            k: RRFのハイパーパラメータ（大きいほどランク差を縮小、デフォルト60）

        Returns:
            統合された結果（文書とRRFスコアのタプルのリスト）
        """
        # Reciprocal Rank Fusionのスコア計算用のデータ構造を初期化
        # 同じ文書が複数のクエリでヒットした場合、スコアを累積
        doc_scores = defaultdict(float)  # 文書ごとのRRFスコアを累積する辞書
        doc_objects = {}  # 文書オブジェクトを保存する辞書

        # 各クエリの検索結果を処理
        for query, results in results_dict.items():
            # 各文書にランク付け（順位は1から開始）
            for rank, (doc, original_score) in enumerate(results, 1):
                # 文書内容のハッシュ値を使って一意なキーを作成
                # 同じ内容の文書は同じキーになる
                doc_key = hash(doc.page_content)

                # Reciprocal Rank Fusionの公式：1 / (rank + k)
                # rankが小さい（上位）ほどスコアが高くなる
                # kはスムージングパラメータ（大きいほどランク差が縮小）
                k_value = k if k is not None else config.FUSION_RRF_K  # デフォルト60
                rrf_score = 1 / (rank + k_value)

                # 同じ文書が複数クエリでヒットした場合、スコアを累積
                doc_scores[doc_key] += rrf_score

                # 文書オブジェクトを初回のみ保存（重複を避ける）
                if doc_key not in doc_objects:
                    doc_objects[doc_key] = doc

        # RRFスコアで降順ソート
        # 複数クエリで頻繁にヒットした文書が上位に来る
        sorted_docs = sorted(
            doc_scores.items(),  # (文書キー, RRFスコア)のタプルのリスト
            key=lambda x: x[1],  # RRFスコアでソート
            reverse=True,  # 降順（スコアが高い順）
        )

        # ソート結果を(Document, score)の形式に変換して返す
        # これが最終的な統合結果となる
        return [
            (doc_objects[doc_key], score)  # 文書オブジェクトとRRFスコア
            for doc_key, score in sorted_docs  # ソート済みリストから取得
        ]

    def search_with_fusion(self, original_query: str, k: int = 10, num_queries: int = 5) -> Dict:
        """
        RAG-Fusionを使用した検索のメインメソッド

        処理の流れ：
        1. ユーザーの質問から複数の検索クエリを生成
        2. 各クエリでChromaDBを検索（並列的に実行）
        3. Reciprocal Rank Fusionで結果を統合
        4. 上位k件を最終結果として返す

        Args:
            original_query: ユーザーの質問
            k: 最終的に取得する文書数（デフォルト10件）
            num_queries: 生成するクエリ数（デフォルト5つ）

        Returns:
            検索結果と統計情報を含む辞書
        """
        # ステップ1: 元のクエリから複数の検索クエリを生成
        # ここがRAG-Fusionの最初の重要なステップ
        logger.info("Generating multiple queries...")
        queries = self.generate_queries(original_query, num_queries)
        # ステップ2: 各クエリで検索を実行（概念的には並列）
        # ここがRAG-Fusionの核心：異なる観点からの検索結果を収集
        logger.info("Executing parallel searches...")
        results_dict = {}  # クエリごとの検索結果を格納する辞書

        # 各クエリでベクトル類似度検索を実行
        for query in queries:
            # ChromaDBでベクトル類似度検索を実行
            # クエリをベクトル化 → DB内の文書ベクトルと比較 → 類似度が高いものを取得
            results = self.vectorstore.similarity_search_with_score(
                query,  # 検索クエリ（生成されたクエリまたは元のクエリ）
                k=k * 2,  # 統合で絞り込まれるので、多めに取得（最終k件の2倍）
            )
            results_dict[query] = results  # 結果をクエリごとに保存

            # 検索結果数をログに記録（デバッグや監視用）
            logger.info(f"Found {len(results)} results for: {query[:50]}...")

        # ステップ3: Reciprocal Rank Fusionで複数の検索結果を統合
        # 複数クエリで頻繁にヒットした文書が高スコアになる
        logger.info("Applying Reciprocal Rank Fusion...")
        fused_results = self.reciprocal_rank_fusion(results_dict)  # RRFアルゴリズムを適用

        # ステップ4: 統合結果から上位k件を最終結果として選択
        # 指定された数だけ取得（デフォルトは10件）
        final_results = fused_results[:k]

        # 検索結果と統計情報を辞書形式で返す
        # この情報は後でユーザーへの回答生成やデバッグに使用
        return {
            "original_query": original_query,  # 元の質問
            "generated_queries": queries,  # 生成された複数の検索クエリ
            "results": final_results,  # RRF統合後の最終結果（文書とスコアのタプル）
            "num_queries": len(queries),  # 使用したクエリ数
            "total_unique_docs": len(fused_results),  # 統合後のユニーク文書総数
            "top_rrf_score": (final_results[0][1] if final_results else None),  # 最高RRFスコア
        }

    def format_results(self, search_results: Dict) -> str:
        """
        検索結果を人間が読みやすい形式にフォーマット

        ターミナルやログに出力するための整形済みテキストを作成します。
        絵文字を使って視覚的にわかりやすくしています。

        Args:
            search_results: search_with_fusionメソッドの返り値（辞書）

        Returns:
            フォーマットされた文字列（直接printできる）
        """
        # 出力用の文字列リストを初期化
        output = []

        # ヘッダー部分：元のクエリを目立つように表示
        output.append(f"\n{'='*60}")
        output.append(f"元のクエリ: {search_results['original_query']}")
        output.append(f"{'='*60}\n")

        # 生成された複数の検索クエリをリスト表示
        # これらのクエリが異なる観点から検索を実行
        output.append("🔍 生成されたクエリ:")
        for i, query in enumerate(search_results["generated_queries"], 1):
            output.append(f"  {i}. {query}")  # 番号付きリストで表示

        # 検索結果の統計情報を表示
        output.append(f"\n📊 統計情報:")
        output.append(f"- クエリ数: {search_results['num_queries']}")
        output.append(f"- ユニーク文書数: {search_results['total_unique_docs']}")

        # RRFスコアがNoneの場合（検索結果が0件）は表示しない
        if search_results["top_rrf_score"]:
            output.append(
                f"- 最高RRFスコア: {search_results['top_rrf_score']:.6f}"
            )  # RRFスコアは小数点以下6桁

        # 上位5件の検索結果を詳細表示
        output.append(f"\n📚 上位検索結果（RRF統合後）:")

        # 最大5件まで、各文書の概要を表示
        for i, (doc, score) in enumerate(search_results["results"][:5], 1):
            output.append(f"\n[結果 {i}] RRFスコア: {score:.6f}")
            # 文書内容のプレビュー（200文字まで）
            output.append(f"内容: {doc.page_content[:200]}...")
            # メタデータ（ファイル名やページ番号など）があれば表示
            if doc.metadata:
                output.append(f"メタデータ: {doc.metadata}")

        # リストを改行で結合して、1つの文字列として返す
        # これをprint()すると整形された結果が表示される
        return "\n".join(output)


# スクリプトが直接実行された場合のテストコード
# python rag_fusion.py で実行すると、以下のテストが動作する
if __name__ == "__main__":
    # 設定ファイルが正しく読み込まれているか検証
    if not config.validate_config():
        exit(1)

    # RAG-Fusionインスタンスを作成
    # config.pyから自動的にAPIキーやモデル設定を読み込む
    fusion = RAGFusion()

    # テスト用の質問リスト
    # 実際のPhase 1のChromaDBに含まれている内容に関する質問を用意
    test_queries = [
        "スマートグラフの有給休暇は3年目だと何日ある？",
        "その月の残業時間はどのような計算式で取得できる？2025年9月の場合はどうなる？",
        "ChromaDBのインデックス構築のベストプラクティス",
    ]

    # 各質問に対してRAG-Fusion検索を実行
    for query in test_queries:
        print(f"\n{'🔍'*30}")  # 虫眼鏡の絵文字で区切り線

        # RAG-Fusion検索を実行
        # k=5: 最終的に5件の文書を取得
        # num_queries=4: 4つの異なるクエリを生成
        results = fusion.search_with_fusion(query, k=5, num_queries=4)

        # 結果を見やすく整形して表示
        print(fusion.format_results(results))
