#!/usr/bin/env python3
"""
RAGチェーン実装
ChromaDBから関連ドキュメントを検索し、LLMで回答を生成する

主な機能:
1. ChromaDBベクトルストアの読み込み
2. 質問に対する関連ドキュメントの検索
3. OpenAI GPTモデルでの回答生成
4. 会話履歴の管理
"""

# 標準ライブラリのインポート
import os  # OS関連の操作（環境変数の取得など）
from typing import (
    List,
    Dict,
    Optional,
    Tuple,
)  # 型ヒント用（リスト、辞書、オプション、タプル）
from datetime import datetime  # 日時操作用（処理時間の計測など）
from dotenv import load_dotenv  # .envファイルから環境変数を読み込む

# LangChain関連のインポート
from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI,
)  # OpenAIの埋め込みモデルとチャットモデル
from langchain_chroma import Chroma  # ChromaDBベクトルデータベース操作
from langchain.schema import (
    Document,
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)  # LangChainの基本スキーマ
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)  # プロンプトテンプレート作成
from langchain.chains import (
    ConversationalRetrievalChain,
)  # 会話型検索チェーン（未使用だが互換性のため残す）
from langchain.memory import ConversationBufferMemory  # 会話履歴を保持するメモリ
from langchain_core.prompts import (
    PromptTemplate,
)  # コアプロンプトテンプレート（未使用だが互換性のため残す）
from langchain_core.output_parsers import StrOutputParser  # LLMの出力を文字列にパース
from langchain_core.runnables import (
    RunnablePassthrough,
)  # パイプラインで値をそのまま通す

# 環境変数の読み込み（.envファイルからOPENAI_API_KEYなどを読み込む）
load_dotenv()

# 新しい設定管理システムを使用
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# ========================================
# 設定
# ========================================

# ChromaDBの設定
CHROMA_PERSIST_DIRECTORY = config.chromadb['persist_directory']
CHROMA_COLLECTION_NAME = config.chromadb['collection_name']

# OpenAI Embeddingモデル（テキストをベクトルに変換するモデル）
EMBEDDING_MODEL = config.embedding['model']

# OpenAI LLMモデル（大規模言語モデル）の設定
LLM_MODEL = config.llm['model']
LLM_TEMPERATURE = config.llm['temperature']
LLM_MAX_TOKENS = config.llm['max_tokens']

# 検索設定
RETRIEVER_K = config.retriever['k']
RETRIEVER_SCORE_THRESHOLD = config.retriever['score_threshold']


class RAGChain:
    """RAGチェーンクラス - 検索拡張生成(Retrieval-Augmented Generation)の実装"""

    def __init__(self):
        """初期化 - RAGシステムの各コンポーネントを初期化"""
        self.vectorstore = None  # ChromaDBベクトルストア（ドキュメントの保存先）
        self.retriever = None  # 検索器（関連ドキュメントを検索）
        self.llm = None  # 大規模言語モデル（回答生成用）
        self.chain = None  # RAGチェーン（検索と生成を繋ぐパイプライン）
        self.memory = ConversationBufferMemory(  # 会話履歴を管理するメモリ
            memory_key="chat_history",  # プロンプトテンプレート内でのキー名
            return_messages=True,  # メッセージオブジェクトとして返す（Falseなら文字列）
            output_key="answer",  # 出力のキー名（LLMの回答を保存）
        )

        # 初期化実行（コンストラクタから自動実行）
        self._initialize()

    def _initialize(self):
        """コンポーネントの初期化 - プライベートメソッド"""
        # OpenAI APIキーの確認（必須）
        if not os.getenv(
            "OPENAI_API_KEY"
        ):  # 環境変数OPENAI_API_KEYが設定されているか確認
            raise ValueError(
                "OPENAI_API_KEYが設定されていません"
            )  # 未設定の場合はエラー

        # ChromaDBの存在確認（indexer.pyで作成済みである必要がある）
        if not os.path.exists(
            CHROMA_PERSIST_DIRECTORY
        ):  # ChromaDBディレクトリが存在するか確認
            raise ValueError(  # 存在しない場合はエラーメッセージ付きで例外を投げる
                f"ChromaDBが見つかりません: {CHROMA_PERSIST_DIRECTORY}\n"
                "先にindexer.pyを実行してベクトルストアを作成してください"
            )

        # 1. ベクトルストアの読み込み
        print("📚 ベクトルストアを読み込み中...")  # 読み込み開始メッセージ
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL
        )  # OpenAI埋め込みモデルのインスタンス作成
        self.vectorstore = Chroma(  # ChromaDBベクトルストアを初期化
            collection_name=CHROMA_COLLECTION_NAME,  # コレクション名を指定
            embedding_function=embeddings,  # 埋め込み関数を設定（クエリのベクトル化に使用）
            persist_directory=CHROMA_PERSIST_DIRECTORY,  # 永続化ディレクトリを指定
        )

        # ドキュメント数の確認（正常に読み込めたか確認）
        collection = (
            self.vectorstore._collection
        )  # ChromaDBの内部コレクションオブジェクトを取得
        doc_count = collection.count()  # コレクション内のドキュメント数をカウント
        print(f"  ✅ {doc_count} 個のチャンクを読み込み完了")  # 読み込み完了メッセージ

        # 2. Retrieverの設定（検索器の設定）
        self.retriever = self.vectorstore.as_retriever(  # ベクトルストアから検索器を作成
            search_type="similarity_score_threshold",  # 検索タイプ：類似度スコアの閾値を使用
            search_kwargs={  # 検索パラメータの辞書
                "k": RETRIEVER_K,  # 取得する文書の最大数（上位K件）
                "score_threshold": RETRIEVER_SCORE_THRESHOLD,  # 類似度スコアの最小閾値（これ以上のスコアのみ返す）
            },
        )

        # 3. LLMの設定（大規模言語モデルの設定）
        self.llm = ChatOpenAI(  # OpenAIのチャットモデルを初期化
            model=LLM_MODEL,  # 使用するモデル名（gpt-4o-miniなど）
            temperature=LLM_TEMPERATURE,  # 温度パラメータ（0=決定的、1=創造的）
            max_tokens=LLM_MAX_TOKENS,  # 生成する最大トークン数（回答の長さ制限）
        )

        # 4. プロンプトテンプレートの作成（LLMへの指示文）
        system_template = """あなたは親切で知識豊富なアシスタントです。
以下のコンテキスト情報を使用して、ユーザーの質問に正確かつ詳細に回答してください。

回答の際は以下の点に注意してください：
1. コンテキストに基づいて回答すること
2. コンテキストに情報がない場合は、その旨を明確に伝えること
3. 推測や憶測は避け、事実に基づいた回答をすること
4. 必要に応じて、関連する追加情報も提供すること
5. 日本語で丁寧に回答すること

コンテキスト:
{context}

会話履歴:
{chat_history}

質問: {question}

回答:"""  # プロンプトテンプレート文字列（{変数}は実行時に置換される）

        self.prompt = ChatPromptTemplate.from_template(
            system_template
        )  # テンプレート文字列からプロンプトオブジェクトを作成

        # 5. チェーンの構築（検索と生成を繋ぐパイプライン）
        self._build_chain()  # プライベートメソッドを呼び出してチェーンを構築

    def _build_chain(self):
        """RAGチェーンを構築 - LCEL（LangChain Expression Language）を使用"""

        # ドキュメントをフォーマットする内部関数
        def format_docs(docs: List[Document]) -> str:
            """検索されたドキュメントを文字列に変換（プロンプトに含めるため）"""
            formatted_docs = []  # フォーマット済みドキュメントを格納するリスト
            for i, doc in enumerate(docs, 1):  # 各ドキュメントを1から番号付けでループ
                source = doc.metadata.get(
                    "source", "不明"
                )  # メタデータからソース取得（デフォルト：不明）
                title = doc.metadata.get(
                    "title", "無題"
                )  # メタデータからタイトル取得（デフォルト：無題）
                content = doc.page_content  # ドキュメントの実際の内容テキスト

                formatted_docs.append(  # フォーマット済みテキストをリストに追加
                    f"【ドキュメント {i}】\n"  # ドキュメント番号
                    f"ソース: {source}\n"  # ソース情報（NotionまたはGoogle Drive）
                    f"タイトル: {title}\n"  # ドキュメントのタイトル
                    f"内容:\n{content}\n"  # ドキュメントの内容
                )

            return "\n---\n".join(
                formatted_docs
            )  # 各ドキュメントを「---」で区切って結合

        # チェーンの構築（LCEL - LangChain Expression Language使用）
        self.chain = (  # パイプライン演算子（|）で処理を連結
            {  # 辞書で複数の入力を並列処理
                "context": self.retriever
                | format_docs,  # 検索器で文書を取得し、フォーマット関数で文字列化
                "chat_history": lambda x: self._format_chat_history(),  # 会話履歴をフォーマット（ラムダ関数）
                "question": RunnablePassthrough(),  # 質問をそのまま通す（変換なし）
            }  # この辞書がプロンプトテンプレートの変数に対応
            | self.prompt  # プロンプトテンプレートに辞書の値を埋め込む
            | self.llm  # LLMで回答を生成
            | StrOutputParser()  # LLMの出力を文字列として取り出す
        )

    def _format_chat_history(self) -> str:
        """会話履歴をフォーマット - プロンプトに含めるための文字列化"""
        messages = (
            self.memory.chat_memory.messages
        )  # メモリから会話履歴のメッセージリストを取得
        if not messages:  # 履歴が空の場合
            return "（会話履歴なし）"  # デフォルトメッセージを返す

        formatted = []  # フォーマット済みメッセージを格納するリスト
        for msg in messages[-10:]:  # 直近10件まで（メモリ制限のため古いものは除外）
            if isinstance(msg, HumanMessage):  # ユーザーのメッセージの場合
                formatted.append(
                    f"User: {msg.content}"
                )  # 「User:」プレフィックスを付けて追加
            elif isinstance(msg, AIMessage):  # AIのメッセージの場合
                formatted.append(
                    f"Assistant: {msg.content}"
                )  # 「Assistant:」プレフィックスを付けて追加

        return "\n".join(formatted)  # 改行で結合して1つの文字列にする

    def search_similar_documents(
        self, query: str, k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        類似ドキュメントを検索（スコア付き）

        Args:
            query: 検索クエリ（ユーザーの質問文）
            k: 検索する文書数（Noneの場合はデフォルト値を使用）

        Returns:
            (Document, score)のタプルのリスト - スコアが高いほど類似度が高い
        """
        if k is None:  # k引数が指定されていない場合
            k = RETRIEVER_K  # デフォルト値（環境変数から読み込んだ値）を使用

        results = self.vectorstore.similarity_search_with_score(
            query, k=k
        )  # ChromaDBで類似検索（スコア付き）
        return results  # [(doc1, score1), (doc2, score2), ...] の形式で返す

    def ask(self, question: str, verbose: bool = False) -> Dict:
        """
        質問に回答 - RAGパイプラインのメイン処理

        Args:
            question: ユーザーの質問文
            verbose: 詳細情報を出力するか（True: 処理状況を表示、False: 静かに処理）

        Returns:
            回答と関連情報を含む辞書（answer, sources, elapsed_time等）
        """
        start_time = datetime.now()  # 処理開始時刻を記録（処理時間計測用）

        # 関連ドキュメントの検索
        if verbose:  # 詳細モードの場合
            print(f"\n🔍 検索中: '{question}'")  # 検索開始メッセージ

        retrieved_docs = self.search_similar_documents(
            question
        )  # 類似ドキュメントを検索

        if verbose:  # 詳細モードの場合
            print(
                f"  📄 {len(retrieved_docs)} 個の関連ドキュメントを発見"
            )  # 検索結果数を表示
            for i, (doc, score) in enumerate(retrieved_docs[:3], 1):  # 上位3件を表示
                title = doc.metadata.get("title", "無題")  # ドキュメントタイトル取得
                source = doc.metadata.get("source", "不明")  # ソース情報取得
                print(
                    f"    {i}. [{source}] {title} (スコア: {score:.3f})"
                )  # 検索結果を表示

        # 回答の生成
        if verbose:  # 詳細モードの場合
            print(f"\n💭 回答を生成中...")  # 生成開始メッセージ

        answer = self.chain.invoke(question)  # RAGチェーンを実行して回答を生成

        # 会話履歴に追加（次の質問で文脈として使用される）
        self.memory.chat_memory.add_user_message(question)  # ユーザーの質問を履歴に追加
        self.memory.chat_memory.add_ai_message(answer)  # AIの回答を履歴に追加

        # 処理時間の計算
        elapsed_time = (
            datetime.now() - start_time
        ).total_seconds()  # 経過時間を秒単位で計算

        # 結果をまとめる（APIレスポンス形式の辞書）
        result = {
            "question": question,  # 元の質問文
            "answer": answer,  # 生成された回答
            "sources": [  # 参照したソース情報のリスト
                {
                    "title": doc.metadata.get("title", "無題"),  # ドキュメントタイトル
                    "source": doc.metadata.get(
                        "source", "不明"
                    ),  # ソース（NotionかGoogle Drive）
                    "score": float(
                        score
                    ),  # 類似度スコア（0-1の値、高いほど関連性が高い）
                    "content_preview": doc.page_content[:200]
                    + "...",  # 内容の先頭200文字プレビュー
                }
                for doc, score in retrieved_docs  # 検索された全ドキュメントを処理
            ],
            "elapsed_time": elapsed_time,  # 処理時間（秒）
            "timestamp": datetime.now().isoformat(),  # タイムスタンプ（ISO形式）
        }

        if verbose:  # 詳細モードの場合
            print(f"  ✅ 完了 ({elapsed_time:.2f}秒)")  # 処理完了メッセージと処理時間

        return result  # 結果辞書を返す

    def clear_memory(self):
        """会話履歴をクリア - 新しい会話を開始する際に使用"""
        self.memory.clear()  # ConversationBufferMemoryの内容を削除
        print("💨 会話履歴をクリアしました")  # クリア完了メッセージ

    def get_conversation_history(self) -> List[Dict]:
        """
        会話履歴を取得 - 保存や表示用

        Returns:
            会話履歴のリスト [{"role": "user/assistant", "content": "..."}]
        """
        messages = self.memory.chat_memory.messages  # メモリから全メッセージを取得
        history = []  # 返却用のリスト

        for msg in messages:  # 各メッセージを処理
            if isinstance(msg, HumanMessage):  # ユーザーメッセージの場合
                history.append(
                    {"role": "user", "content": msg.content}
                )  # role=userで追加
            elif isinstance(msg, AIMessage):  # AIメッセージの場合
                history.append(
                    {"role": "assistant", "content": msg.content}
                )  # role=assistantで追加

        return history  # 会話履歴のリストを返す


def main():
    """テスト実行 - このファイルを直接実行した時のテスト関数"""
    print("=" * 60)  # 区切り線
    print("🤖 RAGチェーンテスト")  # テストタイトル
    print("=" * 60)  # 区切り線

    try:  # エラーハンドリング開始
        # RAGチェーンの初期化
        rag = RAGChain()  # RAGチェーンインスタンスを作成（初期化処理が自動実行される）

        # テスト質問（実際のユースケースを想定した質問）
        test_questions = [
            "スマートグラフの就業規則について教えてください",  # 会社固有の情報を問う質問
            "会社の労働時間について教えてください",  # 労務関連の質問
            "有給休暇の取得方法は？",  # 具体的な手続きに関する質問
        ]

        for question in test_questions:  # 各テスト質問を順番に処理
            print(f"\n❓ 質問: {question}")  # 質問を表示
            print("-" * 40)  # 区切り線

            # 回答を取得
            result = rag.ask(
                question, verbose=True
            )  # RAGチェーンで回答生成（詳細モードON）

            # 回答を表示
            print(f"\n📝 回答:")  # 回答セクションのヘッダー
            print(result["answer"])  # 生成された回答を表示

            # ソース情報を表示
            print(f"\n📚 参照ソース:")  # ソースセクションのヘッダー
            for i, source in enumerate(
                result["sources"][:3], 1
            ):  # 上位3件のソースを表示
                print(
                    f"  {i}. [{source['source']}] {source['title']} (スコア: {source['score']:.3f})"
                )  # ソース情報

            print("\n" + "=" * 60)  # セクション終了の区切り線

            # 次の質問まで少し待機（API制限対策）
            import time  # time モジュールをインポート

            time.sleep(1)  # 1秒待機

        # 会話履歴の表示（テスト終了後の確認用）
        print("\n💬 会話履歴:")  # 履歴セクションのヘッダー
        history = rag.get_conversation_history()  # 会話履歴を取得
        for i, msg in enumerate(history, 1):  # 各メッセージを番号付きで処理
            role = (
                "👤" if msg["role"] == "user" else "🤖"
            )  # ロールに応じてアイコンを選択
            content = (
                msg["content"][:100] + "..."
                if len(msg["content"]) > 100
                else msg["content"]
            )  # 長い内容は省略
            print(f"  {i}. {role} {content}")  # 履歴を表示

    except Exception as e:  # エラーが発生した場合
        print(f"\n❌ エラー: {e}")  # エラーメッセージを表示
        import traceback  # スタックトレース表示用モジュール

        traceback.print_exc()  # 詳細なエラー情報（スタックトレース）を表示


# このスクリプトが直接実行された場合（importされた場合は実行されない）
if __name__ == "__main__":
    main()  # テスト関数を実行
