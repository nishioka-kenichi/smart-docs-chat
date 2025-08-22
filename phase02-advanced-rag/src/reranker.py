"""
Reranker Implementation
"""

# 必要なライブラリのインポート
from typing import List, Tuple, Optional  # 型ヒント用のライブラリ
from abc import ABC, abstractmethod  # 抽象基底クラス用
from langchain.schema import Document  # LangChainのドキュメント型
from loguru import logger  # ログ出力用

# 設定をインポート
import config

# Cohereは使用しないためコメントアウト
# try:
#     import cohere  # Cohere API用ライブラリ
#     COHERE_AVAILABLE = True
# except ImportError:
COHERE_AVAILABLE = False  # Cohereは使用しない

try:
    from sentence_transformers import CrossEncoder  # Cross-Encoderモデル用ライブラリ

    CROSS_ENCODER_AVAILABLE = True  # Cross-Encoderが利用可能かどうかのフラグ
except ImportError:
    CROSS_ENCODER_AVAILABLE = False  # Cross-Encoderが利用不可
    logger.warning("Cross-Encoder not available. Install with: pip install sentence-transformers")


class BaseReranker(ABC):
    """Rerankerの基底クラス：すべてのRerankerが実装すべきインターフェースを定義"""

    @abstractmethod  # サブクラスで必ず実装する必要がある抽象メソッド
    def rerank(
        self,
        query: str,  # 検索クエリ
        documents: List[Document],  # 再順位付け対象の文書リスト
        top_k: int = 10,  # 返す文書数
    ) -> List[Tuple[Document, float]]:
        """文書を再順位付けする抽象メソッド"""
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoderモデルを使用したReranker

    Bi-Encoderと異なり、クエリと文書を同時に処理して
    関連性を直接判定する。より正確だが計算コストが高い。
    MS MARCOデータセットで学習済みのモデルを使用。
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        初期化

        Args:
            model_name: 使用するCross-Encoderモデル
        """
        # sentence-transformersライブラリがインストールされているか確認
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. Install with: pip install sentence-transformers"
            )

        # Cross-Encoderモデルをロード（MS MARCOデータセットで学習済みのモデル）
        self.model = CrossEncoder(model_name)
        logger.info(f"Loaded Cross-Encoder model: {model_name}")

    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Cross-Encoderで文書を再順位付け

        Args:
            query: クエリ
            documents: 再順位付けする文書
            top_k: 返す文書数

        Returns:
            再順位付けされた文書とスコア
        """
        # 文書が空の場合は空リストを返す
        if not documents:
            return []

        # Cross-Encoder用にクエリと文書のペアを作成
        pairs = [[query, doc.page_content] for doc in documents]  # [クエリ, 文書]のペアリスト

        # Cross-Encoderモデルでスコアを計算（クエリと文書の関連性を直接判定）
        scores = self.model.predict(pairs)

        # 文書とスコアをペアにして降順ソート
        doc_scores = list(zip(documents, scores))  # (文書, スコア)のタプルリスト
        doc_scores.sort(key=lambda x: x[1], reverse=True)  # スコアで降順ソート

        # 上位k件を最終結果として返す
        result = doc_scores[:top_k]
        logger.info(f"Reranked {len(result)} documents with Cross-Encoder")

        return result


class HybridReranker(BaseReranker):
    """複数のRerankerを組み合わせたハイブリッドReranker

    CohereとCross-Encoderなど、複数のRerankerを並列実行し、
    それらのスコアを統合（平均）することで、より堅牢な
    再順位付けを実現する。各Rerankerの長所を活かせる。
    """

    def __init__(self, use_cohere: bool = True, use_cross_encoder: bool = True):
        """
        初期化

        Args:
            use_cohere: Cohere Rerankを使用するか
            use_cross_encoder: Cross-Encoderを使用するか
        """
        # 使用するRerankerのリストを初期化
        self.rerankers = []

        # Cohereは使用しない
        # if use_cohere and COHERE_AVAILABLE:
        #     try:
        #         self.rerankers.append(CohereReranker())
        #         logger.info("Cohere Reranker enabled")
        #     except Exception as e:
        #         logger.warning(f"Cohere Reranker initialization failed: {e}")

        # Cross-Encoder Rerankerの初期化を試行
        if use_cross_encoder and CROSS_ENCODER_AVAILABLE:  # Cross-Encoderを使用し、ライブラリが利用可能な場合
            try:
                self.rerankers.append(CrossEncoderReranker())  # Rerankerリストに追加
                logger.info("Cross-Encoder Reranker enabled")
            except Exception as e:
                logger.warning(f"Cross-Encoder Reranker initialization failed: {e}")

        # どのRerankerも利用できない場合の警告
        if not self.rerankers:
            logger.warning("No rerankers available. Using fallback ranker.")

    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        複数のRerankerで再順位付けし、結果を統合

        Args:
            query: クエリ
            documents: 再順位付けする文書
            top_k: 返す文書数

        Returns:
            再順位付けされた文書とスコア
        """
        # 文書が空の場合は空リストを返す
        if not documents:
            return []

        # 利用可能なRerankerが一つもない場合はフォールバック処理
        if not self.rerankers:
            logger.warning("Using fallback ranker (no reranking)")
            # 元の順序を維持し、順位に応じたダミースコアを付与（1.0, 0.99, 0.98...）
            return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(documents[:top_k])]

        # 複数のRerankerで並列的に再順位付けを実行
        all_results = []  # 各Rerankerの結果を格納
        for reranker in self.rerankers:
            try:
                # 各Rerankerで再順位付け（統合のためtop_kの2倍取得）
                results = reranker.rerank(query, documents, top_k=top_k * 2)
                all_results.append(results)  # 結果をリストに追加
            except Exception as e:
                logger.error(f"Reranker failed: {e}")

        # すべてのRerankerが失敗した場合のフォールバック処理
        if not all_results:
            # 元の順序を維持し、順位に応じたダミースコアを付与（1.0, 0.99, 0.98...）
            return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(documents[:top_k])]

        # 複数のRerankerの結果を統合するための変数を初期化
        # 各文書のスコアを累積し、後で平均を計算する
        doc_scores = {}  # 文書ごとの累積スコア
        doc_counts = {}  # 文書ごとの出現回数（平均計算用）
        doc_map = {}  # ハッシュキーから元の文書オブジェクトへのマッピング

        # 各Rerankerの結果を処理し、文書ごとにスコアを集計
        for results in all_results:
            for doc, score in results:
                # 文書の内容をハッシュ化して一意なキーを生成
                doc_key = hash(doc.page_content)

                # 初めて出現する文書の場合は初期化
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = 0  # スコアの累積値を0で初期化
                    doc_counts[doc_key] = 0  # 出現回数を0で初期化
                    doc_map[doc_key] = doc  # ハッシュキーと文書を紐付け

                # スコアと出現回数を更新
                doc_scores[doc_key] += score  # スコアを累積
                doc_counts[doc_key] += 1  # 出現回数をカウント

        # 各文書の平均スコアを計算し、最終結果を生成
        final_results = []
        for doc_key in doc_scores:
            # 累積スコアを出現回数で割って平均を算出
            avg_score = doc_scores[doc_key] / doc_counts[doc_key]
            # 元の文書オブジェクトと平均スコアをペアにして追加
            final_results.append((doc_map[doc_key], avg_score))

        # スコアの高い順（降順）にソート
        final_results.sort(key=lambda x: x[1], reverse=True)

        return final_results[:top_k]


class SimpleReranker(BaseReranker):
    """シンプルなフォールバック用Reranker

    高度なRerankerが利用できない場合のフォールバック実装。
    元の検索順序を維持し、順位に応じた単純なスコアを付与する。
    実質的に再順位付けは行わない。
    """

    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        単純な再順位付け（元の順序を維持）

        Args:
            query: クエリ
            documents: 再順位付けする文書
            top_k: 返す文書数

        Returns:
            文書とスコア
        """
        # 元の順序を維持し、順位に応じたダミースコアを付与（1.0, 0.99, 0.98...）
        return [(doc, 1.0 - i * 0.01) for i, doc in enumerate(documents[:top_k])]


# 利用可能なRerankerを取得するファクトリー関数
def get_reranker(reranker_type: str = "auto") -> BaseReranker:
    """
    利用可能なRerankerを取得するファクトリー関数

    環境や設定に応じて適切なRerankerを選択・初期化する。
    "auto"を指定すると、利用可能な最適なRerankerを自動選択。

    Args:
        reranker_type: "cohere", "cross_encoder", "hybrid", "auto", "simple"

    Returns:
        Rerankerインスタンス
    """
    # 設定ファイルから指定がある場合はそれを優先
    if reranker_type == "auto":
        reranker_type = config.RERANKER_TYPE if config.RERANKER_ENABLED else "simple"

    if reranker_type == "auto":
        # 利用可能なRerankerを自動選択
        # Cross-Encoder（ローカルで動作、無料）
        if CROSS_ENCODER_AVAILABLE:
            try:
                return CrossEncoderReranker()
            except:
                pass  # 初期化失敗時は次の選択肢へ

        # 失敗した場合はフォールバック
        logger.warning("No advanced reranker available. Using simple reranker.")
        return SimpleReranker()

    # elif reranker_type == "cohere":
    #     return CohereReranker()  # Cohereは使用しない

    elif reranker_type == "cross_encoder":
        return CrossEncoderReranker()

    elif reranker_type == "hybrid":
        return HybridReranker()

    else:
        return SimpleReranker()


# スクリプトが直接実行された場合のテストコード
if __name__ == "__main__":
    # 設定の検証
    if not config.validate_config():
        exit(1)

    # テスト実行
    # テスト用の文書（実際のRAG-Fusion検索結果から抜粋）
    test_docs = [
        Document(page_content="労働基準法では月45時間を超える残業には特別条項が必要です"),
        Document(page_content="月次の勤怠集計では、深夜残業と休日出勤を分けて計算する必要があります"),
        Document(page_content="残業時間の計算式: 月の総労働時間 - 法定労働時間（160時間）"),
        Document(page_content="2025年9月の法定労働時間は、営業日20日 × 8時間 = 160時間です"),
        Document(page_content="残業代の計算には、時間外労働の割増率1.25倍を適用します"),
    ]

    # テストクエリ
    query = "その月の残業時間はどのような計算式で取得できる？2025年9月の場合はどうなる？"

    # 利用可能なRerankerを自動選択
    print("\n" + "=" * 60)
    print("Reranker Test")
    print("=" * 60)

    reranker = get_reranker("auto")
    print(f"Using reranker: {type(reranker).__name__}")

    reranked = reranker.rerank(query, test_docs, top_k=3)

    for i, (doc, score) in enumerate(reranked, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"Content: {doc.page_content}")
