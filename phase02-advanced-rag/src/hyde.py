"""
HyDE (Hypothetical Document Embeddings) Implementation

HyDEはユーザーの質問に対して、AIが「仮想的な理想的回答」を生成し、
その回答を使って検索することで、より精度の高い文書を見つける技術です。

例：「有給休暇は何日？」という質問に対して
1. AIが「入社3年目の社員の有給休暇は15日で...」という仮想回答を生成
2. その仮想回答で検索することで、実際の就業規則文書を見つけやすくする
"""

# 必要なライブラリのインポート
from typing import List, Optional, Dict  # 型ヒント用のライブラリ
from textwrap import dedent  # 複数行文字列のインデント調整用
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAIのLLMと埋め込みモデル
from langchain.prompts import PromptTemplate  # プロンプトテンプレート作成用
from langchain_chroma import Chroma  # ChromaDBベクトルストア用クライアント
from langchain.schema import Document  # LangChainのドキュメント型
from pathlib import Path  # ファイルパス操作用
from loguru import logger  # type: ignore  # ログ出力用（型チェック無視）

# 設定をインポート
import config


class HyDE:
    """
    Hypothetical Document Embeddings実装クラス
    
    このクラスは以下の流れで動作します：
    1. ユーザーの質問を受け取る
    2. LLMを使って複数の「仮想的な理想回答」を生成
    3. 各仮想回答をベクトル化して類似文書を検索
    4. 複数の検索結果を統合して最終結果を返す
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        embeddings: Optional[OpenAIEmbeddings] = None,
        persist_directory: Optional[str] = None,
    ):
        """
        HyDEシステムの初期化
        
        このメソッドでは以下のコンポーネントをセットアップします：
        - LLM：仮想回答を生成するためのAIモデル
        - Embeddings：テキストをベクトル化するモデル
        - ChromaDB：Phase 1で作成済みのベクトルデータベース

        Args:
            llm: 言語モデル（指定しない場合はconfigから読み込み）
            embeddings: 埋め込みモデル（指定しない場合はconfigから読み込み）
            persist_directory: ChromaDBの保存先（指定しない場合はconfigから読み込み）
        """
        # LLMの初期化：仮想回答を生成するためのAIモデル
        # 引数で渡されなければ、config.py経由で設定を読み込む
        self.llm = llm or ChatOpenAI(
            api_key=config.OPENAI_API_KEY,  # OpenAIのAPIキー（.envから）
            model=config.LLM_MODEL,  # 使用するモデル（gpt-4o-miniなど）
            temperature=config.HYDE_TEMPERATURE,  # 生成の多様性（0.7=バランス良い）
        )

        # 埋め込みモデルの初期化：テキストを数値ベクトルに変換するモデル
        # このベクトルを使って意味的な類似度を計算できる
        self.embeddings = embeddings or OpenAIEmbeddings(
            api_key=config.OPENAI_API_KEY,  # OpenAIのAPIキー
            model=config.EMBEDDING_MODEL  # text-embedding-3-smallなど
        )

        # ChromaDBの初期化：Phase 1で構築済みのベクトルデータベースに接続
        # ChromaDBにはすでに文書がベクトル化されて保存されている
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
        # ここで接続したDBを使って、後で類似文書検索を行う
        self.vectorstore = Chroma(
            persist_directory=str(chromadb_path),  # ChromaDBの保存場所
            embedding_function=self.embeddings,  # 検索時に使う埋め込みモデル
            collection_name=config.CHROMADB_COLLECTION  # コレクション名を指定
        )

        # HyDE用のプロンプトテンプレート：仮想回答を生成するための指示文
        # このプロンプトがHyDEの核心部分！
        # LLMに「実際の文書に書かれていそうな内容」を生成させる
        self.hyde_prompt = PromptTemplate(
            input_variables=["question"],  # {question}にユーザーの質問が入る
            template=dedent("""
                あなたは質問に対する理想的な回答を生成する専門家です。
                以下の質問に対して、実際のドキュメントに含まれている可能性が高い
                詳細で具体的な回答を書いてください。
                
                重要な点：
                - 具体的な用語や概念を含める
                - 実際の文書に書かれているような文体で
                - 専門用語を適切に使用する
                - 約200-300文字程度で
                
                質問: {question}
                
                理想的な回答:
            """).strip(),
        )

    def generate_hypothetical_documents(
        self, question: str, num_documents: int = 3
    ) -> List[str]:
        """
        複数の仮想回答を生成する（HyDEのステップ1）
        
        このメソッドはユーザーの質問に対して、「実際の文書に書かれていそうな」
        仮想的な回答を複数生成します。複数生成する理由は：
        1. 観点の多様性：異なる角度からの回答を生成
        2. カバレッジの向上：より多くの関連文書を見つける可能性

        Args:
            question: ユーザーの質問（例：「有給休暇は何日？」）
            num_documents: 生成する仮想回答の数（デフォルト3つ）

        Returns:
            仮想回答のリスト（各回答は200-300文字程度）
        """
        # 生成された仮想回答を格納するリスト
        # 複数の回答を生成して、検索の精度を上げる
        hypothetical_docs = []

        # 指定された数だけ仮想回答を生成
        for i in range(num_documents):
            # 温度パラメータを動的に変更（重要なテクニック！）
            # 1つ目：0.5（より確実性の高い回答）
            # 2つ目：0.7（バランスの取れた回答）  
            # 3つ目：0.9（より創造的な回答）
            # これにより異なる観点の回答を生成できる
            self.llm.temperature = 0.5 + (i * 0.2)

            # プロンプトテンプレートとLLMを連結（LangChainのパイプ機能）
            # プロンプト | LLM という記法でチェーンを作成
            chain = self.hyde_prompt | self.llm
            
            # 質問を入力として仮想回答を生成
            # LLMが「もし私がこの質問に答える文書だったら...」と考えて回答
            response = chain.invoke({"question": question})
            
            # 生成された回答をリストに追加
            hypothetical_docs.append(response.content)

            # 進捗状況をログに記録（デバッグや監視用）
            logger.info(f"Generated hypothetical document {i+1}/{num_documents}")

        return hypothetical_docs

    def search_with_hyde(
        self, question: str, k: int = 10, num_hypothetical: int = 3
    ) -> Dict:
        """
        HyDEを使用した検索のメインメソッド
        
        処理の流れ：
        1. ユーザーの質問から複数の仮想回答を生成
        2. 各仮想回答をベクトル化してChromaDBで検索
        3. 複数の検索結果から重複を除去
        4. スコア順にソートして上位k件を返す

        Args:
            question: ユーザーの質問
            k: 最終的に取得する文書数（デフォルト10件）
            num_hypothetical: 生成する仮想回答数（デフォルト3つ）

        Returns:
            検索結果と統計情報を含む辞書
        """
        # ステップ1: 質問から複数の仮想回答を生成
        # ここがHyDEの最初の重要なステップ
        logger.info("Generating hypothetical documents...")
        hypothetical_docs = self.generate_hypothetical_documents(
            question, num_hypothetical
        )

        # ステップ2: 各仮想回答をクエリとして類似文書を検索
        # ここがHyDEの核心：仮想回答と似た内容の実際の文書を探す
        all_results = []  # すべての検索結果を格納
        seen_contents = set()  # 重複チェック用（同じ文書を何度も取得しないように）

        # 各仮想回答を使って検索を実行
        for hyde_doc in hypothetical_docs:
            # ChromaDBでベクトル類似度検索を実行
            # 仮想回答をベクトル化 → DB内の文書ベクトルと比較 → 類似度が高いものを取得
            results = self.vectorstore.similarity_search_with_score(hyde_doc, k=k)

            # 重複文書を除去しながら結果を統合
            for doc, score in results:
                # 文書内容のハッシュ値を計算（同じ内容かどうか判定）
                content_hash = hash(doc.page_content)
                
                # まだ見ていない文書なら追加
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)  # 「この文書は見た」と記録
                    all_results.append((doc, score))  # 結果リストに追加

        # ステップ3: スコアでソート
        # ChromaDBは「距離」を返すので、値が小さいほど類似度が高い
        # （距離0 = 完全に同じ、距離が大きい = 似ていない）
        all_results.sort(key=lambda x: x[1])  # スコア（距離）で昇順ソート

        # ステップ4: 上位k件の文書を最終結果として選択
        # 指定された数だけ取得（デフォルトは10件）
        final_results = all_results[:k]

        # 検索結果と統計情報を辞書形式で返す
        # この情報は後でユーザーへの回答生成やデバッグに使用
        return {
            "question": question,  # 元の質問
            "hypothetical_docs": hypothetical_docs,  # 生成された仮想回答のリスト
            "results": final_results,  # 最終的な検索結果（文書とスコアのタプル）
            "num_unique_docs": len(all_results),  # 重複除去後の総文書数
            "top_score": (
                final_results[0][1] if final_results else None
            ),  # 最高スコア（最も距離が近い=似ている文書）
        }

    def format_results(self, search_results: Dict) -> str:
        """
        検索結果を人間が読みやすい形式にフォーマット
        
        ターミナルやログに出力するための整形済みテキストを作成します。
        絵文字を使って視覚的にわかりやすくしています。

        Args:
            search_results: search_with_hydeメソッドの返り値（辞書）

        Returns:
            フォーマットされた文字列（直接printできる）
        """
        # 出力用の文字列リストを初期化
        output = []
        
        # ヘッダー部分：質問を目立つように表示
        output.append(f"\n{'='*60}")
        output.append(f"質問: {search_results['question']}")
        output.append(f"{'='*60}\n")

        # 生成された仮想回答を表示
        # 長すぎると見にくいので、最初の200文字だけ表示
        output.append("📝 生成された仮想回答:")
        for i, hyde_doc in enumerate(search_results["hypothetical_docs"], 1):
            output.append(f"\n[仮想回答 {i}]")
            output.append(hyde_doc[:200] + "...")  # プレビューとして200文字まで

        # 検索結果の統計情報を表示
        output.append(f"\n{'='*60}")
        output.append(f"📊 検索結果統計:")
        output.append(f"- ユニーク文書数: {search_results['num_unique_docs']}")
        
        # スコアがNoneの場合（検索結果が0件）は表示しない
        if search_results['top_score'] is not None:
            output.append(
                f"- 最高スコア: {search_results['top_score']:.4f}"
            )  # 距離なので、小さいほど良い

        # 上位5件の検索結果を詳細表示
        output.append(f"\n📚 上位検索結果:")
        
        # 最大5件まで、各文書の概要を表示
        for i, (doc, score) in enumerate(
            search_results["results"][:5], 1
        ):
            output.append(f"\n[結果 {i}] スコア: {score:.4f}")
            # 文書内容のプレビュー（200文字まで）
            output.append(
                f"内容: {doc.page_content[:200]}..."
            )
            # メタデータ（ファイル名やページ番号など）があれば表示
            if doc.metadata:
                output.append(f"メタデータ: {doc.metadata}")

        # リストを改行で結合して、1つの文字列として返す
        # これをprint()すると整形された結果が表示される
        return "\n".join(output)


# スクリプトが直接実行された場合のテストコード
# python hyde.py で実行すると、以下のテストが動作する
if __name__ == "__main__":
    # 設定ファイルが正しく読み込まれているか検証
    if not config.validate_config():
        exit(1)
    
    # HyDEインスタンスを作成
    # config.pyから自動的にAPIキーやモデル設定を読み込む
    hyde = HyDE()

    # テスト用の質問リスト
    # 実際のPhase 1のChromaDBに含まれている内容に関する質問を用意
    test_queries = [
        "スマートグラフの有給休暇は3年目だと何日ある？",
        "その月の残業時間はどのような計算式で取得できる？2025年9月の場合はどうなる？",
        "ChromaDBのインデックス構築のベストプラクティス",
    ]

    # 各質問に対してHyDE検索を実行
    for query in test_queries:
        print(f"\n{'🔍'*30}")  # 虫眼鏡の絵文字で区切り線
        
        # HyDE検索を実行
        # k=5: 最終的に5件の文書を取得
        # num_hypothetical=2: 2つの仮想回答を生成
        results = hyde.search_with_hyde(query, k=5, num_hypothetical=2)
        
        # 結果を見やすく整形して表示
        print(hyde.format_results(results))
