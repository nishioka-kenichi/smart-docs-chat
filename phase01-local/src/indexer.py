#!/usr/bin/env python3
"""
ドキュメントインデクサー
取得したドキュメントをベクトル化してChromaDBに保存する

主な機能:
1. NotionとGoogle Driveから取得したドキュメントの読み込み
2. テキストのチャンク分割
3. OpenAI Embeddingsでベクトル化
4. ChromaDBへの保存と永続化
"""

# 標準ライブラリのインポート
import os  # OS関連の操作（ファイルパス、環境変数など）
import json  # JSONファイルの読み書き用
import glob  # ファイルパターンマッチング用（*.txtなど）
import time  # スリープ処理用（APIレート制限対策）
from typing import List, Dict, Optional  # 型ヒント用
from datetime import datetime  # 日時操作用
from dotenv import load_dotenv  # .envファイルから環境変数を読み込む

# LangChain関連のインポート
from langchain_text_splitters import RecursiveCharacterTextSplitter  # テキストを再帰的にチャンク分割
from langchain_openai import OpenAIEmbeddings  # OpenAIの埋め込みモデルを使用
from langchain_chroma import Chroma  # ChromaDBベクトルデータベースの操作
from langchain.schema import Document  # LangChainのDocumentオブジェクト
from langchain_community.vectorstores.utils import filter_complex_metadata  # メタデータのフィルタリング

# 環境変数の読み込み（.envファイルからOPENAI_API_KEYなどを読み込む）
load_dotenv()

# 新しい設定管理システムを使用
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# ========================================
# 設定
# ========================================

# ドキュメントの保存ディレクトリ（data_loaderで保存したファイルがある場所）
NOTION_DOCS_DIR = "./data/documents/notion"  # Notionから取得したドキュメント
GOOGLE_DOCS_DIR = "./data/documents/google"  # Google Driveから取得したドキュメント

# ChromaDBの保存先（ベクトルデータベースの永続化ディレクトリ）
CHROMA_PERSIST_DIRECTORY = config.chromadb['persist_directory']
CHROMA_COLLECTION_NAME = config.chromadb['collection_name']

# チャンク分割の設定（長いテキストを小さい断片に分割）
CHUNK_SIZE = config.chunking['chunk_size']
CHUNK_OVERLAP = config.chunking['chunk_overlap']

# OpenAI Embeddingモデル（テキストをベクトルに変換するモデル）
EMBEDDING_MODEL = config.embedding['model']


def load_documents_from_directory(directory: str, source_type: str) -> List[Document]:
    """
    指定ディレクトリからドキュメントを読み込み、LangChain Documentオブジェクトに変換

    Args:
        directory: ドキュメントが保存されているディレクトリ
        source_type: ソースタイプ（"notion" または "google_drive"）

    Returns:
        Document オブジェクトのリスト
    """
    documents = []  # 読み込んだDocumentオブジェクトを格納するリスト

    # メタデータファイルを読み込む（data_loaderが作成したファイル情報）
    metadata_file = os.path.join(directory, "metadata.json")  # metadata.jsonのパスを作成
    metadata = {}  # メタデータを格納する辞書
    if os.path.exists(metadata_file):  # metadata.jsonが存在する場合
        with open(metadata_file, "r", encoding="utf-8") as f:  # UTF-8でファイルを開く
            metadata = json.load(f)  # JSONをパースしてPythonの辞書に変換

    # ディレクトリ内の全てのテキストファイルを処理
    file_patterns = ["*.txt", "*.md", "*.csv", "*.html", "*.xml", "*.json"]  # 読み込むファイルの拡張子パターン

    for pattern in file_patterns:  # 各ファイルパターンを処理
        for filepath in glob.glob(os.path.join(directory, pattern)):  # パターンにマッチするファイルを検索
            # metadata.jsonはスキップ（メタデータファイル自体は処理しない）
            if filepath.endswith("metadata.json"):
                continue  # 次のファイルに進む

            try:  # エラー処理の開始
                # ファイル内容を読み込む
                with open(filepath, "r", encoding="utf-8") as f:  # ファイルをUTF-8で開く
                    content = f.read()  # ファイルの全内容を文字列として読み込む

                # ファイル名からメタデータを探す
                file_metadata = {}  # このファイルのメタデータを格納する辞書
                filename = os.path.basename(filepath)  # ファイル名を取得（パスからファイル名部分だけを抽出）

                # metadata.jsonからこのファイルの情報を探す
                pages_data = metadata.get("pages", {})  # metadataからpagesデータを取得（なければ空の辞書）
                for page_id, page_info in pages_data.items():  # 各ページの情報をループ
                    if page_info.get("path", "").endswith(filename):  # パスが現在のファイル名で終わる場合
                        file_metadata = {  # メタデータ辞書を作成
                            "source": source_type,  # データソース（notionまたはgoogle_drive）
                            "title": page_info.get("title", filename),  # ドキュメントのタイトル
                            "file_path": filepath,  # ファイルのフルパス
                            "parent_titles_json": json.dumps(  # 親階層のタイトルをJSON文字列として保存
                                page_info.get("parent_titles", []), ensure_ascii=False  # 日本語をそのまま保存
                            ),  # ChromaDBはリスト型を受け付けないためJSON文字列に変換
                            "page_id": page_id,  # ページの一意識別子
                            "type": page_info.get("type", "file"),  # ドキュメントタイプ（page、database、fileなど）
                        }

                        # Notion固有のプロパティを追加
                        if source_type == "notion":  # Notionの場合
                            file_metadata["url"] = page_info.get("properties", {}).get(  # NotionWebページのURLを追加
                                "url"
                            )
                        # Google Drive固有のプロパティを追加
                        elif source_type == "google_drive":  # Google Driveの場合
                            props = page_info.get("properties", {})  # プロパティ辞書を取得
                            file_metadata["drive_link"] = props.get("drive_link")  # Google Driveの共有リンク
                            file_metadata["modified_time"] = props.get("modified_time")  # 最終更新日時
                        break  # マッチしたらループを抜ける

                # メタデータが見つからない場合は基本情報のみ
                if not file_metadata:  # metadata.jsonにこのファイルの情報がない場合
                    file_metadata = {  # 最小限のメタデータを作成
                        "source": source_type,  # データソース
                        "title": filename,  # ファイル名をタイトルとして使用
                        "file_path": filepath,  # ファイルパス
                    }

                # parent_titlesリストをJSON文字列に変換（ChromaDBのメタデータ制限対応）
                if "parent_titles" in file_metadata and isinstance(  # parent_titlesキーが存在し、かつ
                    file_metadata["parent_titles"], list  # リスト型である場合
                ):
                    file_metadata["parent_titles_json"] = json.dumps(  # JSON文字列に変換して新しいキーに保存
                        file_metadata["parent_titles"], ensure_ascii=False  # 日本語をそのまま保存
                    )
                    del file_metadata["parent_titles"]  # 元のリスト型のキーを削除（ChromaDBはリストを受け付けない）

                # Documentオブジェクトを作成
                doc = Document(page_content=content, metadata=file_metadata)  # LangChainのDocumentオブジェクトを作成
                documents.append(doc)  # ドキュメントリストに追加

                print(f"  ✅ 読み込み完了: {filename} ({len(content)} 文字)")  # 読み込み成功メッセージ

            except Exception as e:  # ファイル読み込み中のエラーをキャッチ
                print(f"  ❌ エラー: {filename} - {e}")  # エラーメッセージを表示

    return documents  # 読み込んだ全ドキュメントを返す


def split_documents(documents: List[Document]) -> List[Document]:
    """
    ドキュメントをチャンクに分割（最適化版）
    
    改良点:
    - ドキュメントタイプに応じた適切なチャンクサイズと分割方法を適用
    - 就業規則などの法的文書は条文単位で分割
    - 一般文書は段落単位で分割

    Args:
        documents: 分割前のDocumentリスト

    Returns:
        分割後のDocumentリスト
    """
    split_docs = []  # 分割後のドキュメントを格納するリスト
    
    for doc in documents:
        # ドキュメントタイプを判定（メタデータやタイトルから推測）
        title = doc.metadata.get("title", "")
        file_path = doc.metadata.get("file_path", "")
        content = doc.page_content.lower()  # 内容を小文字化して判定用
        
        # ドキュメントタイプの判定と適切なスプリッターの選択
        if "就業規則" in title or "規程" in title or "第一条" in content[:500]:
            # 法的文書・就業規則用のスプリッター
            # 条文単位で分割し、意味のまとまりを保持
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,  # 小さめのチャンク（条文は短いため）
                chunk_overlap=50,  # 少なめのオーバーラップ（条文は独立性が高い）
                separators=[
                    "\n第",  # 条文の区切り（「第１条」「第２条」など）
                    "\n（",   # 項目の区切り（「（目的）」「（定義）」など）
                    "\n\n",   # 段落の区切り
                    "。\n",   # 文末で改行
                    "。",     # 句点
                    "",       # 最終手段
                ],
                length_function=len,
            )
            doc.metadata["doc_type"] = "legal"  # ドキュメントタイプをメタデータに記録
            
        elif "労働" in title or "勤務" in title or "休暇" in title:
            # 労務関連文書用のスプリッター
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,  # 中程度のチャンク
                chunk_overlap=100,  # 適度なオーバーラップ
                separators=[
                    "\n##",   # マークダウンのセクション
                    "\n\n",   # 段落
                    "\n",     # 改行
                    "。",     # 句点
                    "",       # 最終手段
                ],
                length_function=len,
            )
            doc.metadata["doc_type"] = "hr"  # HR（人事・労務）文書
            
        else:
            # 一般文書用のスプリッター（デフォルト）
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # 標準サイズ
                chunk_overlap=150,  # 標準オーバーラップ
                separators=["\n\n", "\n", "。", ".", " ", ""],
                length_function=len,
            )
            doc.metadata["doc_type"] = "general"  # 一般文書
        
        # ドキュメントを分割
        doc_chunks = text_splitter.split_documents([doc])
        
        # 各チャンクに追加情報を付与
        for j, chunk in enumerate(doc_chunks):
            # 元のドキュメント内でのチャンク位置
            chunk.metadata["chunk_index_in_doc"] = j
            chunk.metadata["total_chunks_in_doc"] = len(doc_chunks)
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            # キーワードを自動抽出してメタデータに追加（検索精度向上）
            keywords = []
            keyword_patterns = [
                "有給休暇", "年次有給休暇", "特別休暇",
                "フレックスタイム", "コアタイム",
                "時間外労働", "残業", "休日出勤",
                "就業規則", "労働契約", "給与", "賞与",
            ]
            chunk_content_lower = chunk.page_content.lower()
            for keyword in keyword_patterns:
                if keyword.lower() in chunk_content_lower:
                    keywords.append(keyword)
            
            if keywords:
                chunk.metadata["keywords"] = ",".join(keywords)
        
        split_docs.extend(doc_chunks)
    
    # 全体を通したチャンクインデックスを付与
    for i, doc in enumerate(split_docs):
        doc.metadata["global_chunk_index"] = i
    
    print(f"  🔄 チャンク分割完了:")
    print(f"     - 法的文書: {sum(1 for d in split_docs if d.metadata.get('doc_type') == 'legal')} チャンク")
    print(f"     - HR文書: {sum(1 for d in split_docs if d.metadata.get('doc_type') == 'hr')} チャンク")
    print(f"     - 一般文書: {sum(1 for d in split_docs if d.metadata.get('doc_type') == 'general')} チャンク")

    return split_docs  # 分割済みドキュメントを返す


def create_or_update_vectorstore(documents: List[Document]) -> Chroma:
    """
    ChromaDBのベクトルストアを作成または更新
    バッチ処理でトークン制限を回避

    Args:
        documents: ベクトル化するDocumentリスト

    Returns:
        Chromaベクトルストアインスタンス
    """
    # ChromaDBが受け付けない複雑なメタデータをフィルタリング
    # リスト、辞書、Noneなどを除去し、str/int/float/boolのみを残す
    documents = filter_complex_metadata(documents)  # ChromaDBが処理できる形式にメタデータをクリーンアップ

    # OpenAI Embeddingsを初期化
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)  # OpenAIの埋め込みモデルを初期化

    # バッチサイズの設定（トークン制限を考慮）
    # text-embedding-ada-002は最大8191トークン/リクエスト、最大300,000トークン/分
    # 安全のため、1バッチあたり100チャンクに制限
    BATCH_SIZE = 100  # 1回のAPIリクエストで処理するチャンク数

    # 既存のベクトルストアがあるかチェック
    vectorstore = None  # ベクトルストアオブジェクトを初期化

    if os.path.exists(CHROMA_PERSIST_DIRECTORY):  # ChromaDBの保存ディレクトリが既に存在する場合
        print(f"\n📂 既存のベクトルストアを更新: {CHROMA_PERSIST_DIRECTORY}")

        # 既存のベクトルストアを読み込む
        vectorstore = Chroma(  # ChromaDBインスタンスを作成
            collection_name=CHROMA_COLLECTION_NAME,  # コレクション名を指定
            embedding_function=embeddings,  # 埋め込み関数を指定
            persist_directory=CHROMA_PERSIST_DIRECTORY,  # 保存ディレクトリを指定
        )

        # バッチごとに新しいドキュメントを追加
        if documents:  # 追加するドキュメントがある場合
            total_docs = len(documents)  # ドキュメントの総数をカウント
            for i in range(0, total_docs, BATCH_SIZE):  # BATCH_SIZEごとにループ
                batch_end = min(i + BATCH_SIZE, total_docs)  # バッチの終了位置を計算（最後のバッチは小さい可能性）
                batch = documents[i:batch_end]  # 現在のバッチを切り出し

                print(  # 進捗表示
                    f"  ⏳ バッチ {i//BATCH_SIZE + 1}/{(total_docs-1)//BATCH_SIZE + 1}: {len(batch)} チャンクを処理中..."  # バッチ番号/総バッチ数: チャンク数
                )

                try:  # エラー処理開始
                    vectorstore.add_documents(batch)  # バッチをベクトルストアに追加
                    print(f"     ✅ 完了")  # 成功メッセージ
                except Exception as e:  # エラーが発生した場合
                    print(f"     ❌ エラー: {e}")  # エラーメッセージを表示
                    # エラーが発生してもバッチ処理を継続（一部のチャンクが失敗しても全体を止めない）
                    continue  # 次のバッチに進む

                # APIレート制限対策のため、バッチ間で少し待機
                if batch_end < total_docs:  # 最後のバッチでない場合
                    time.sleep(1)  # 1秒待機（OpenAI APIのレート制限を避ける）

            print(f"  ➕ {total_docs} 個のチャンクを追加完了")  # 追加完了メッセージ
    else:  # ChromaDBがまだ存在しない場合（初回実行）
        print(f"\n🆕 新規ベクトルストアを作成: {CHROMA_PERSIST_DIRECTORY}")

        # 新規作成（最初のバッチで作成）
        if documents:  # ドキュメントがある場合
            total_docs = len(documents)  # ドキュメントの総数をカウント

            # 最初のバッチでベクトルストアを作成
            first_batch_size = min(BATCH_SIZE, total_docs)  # 最初のバッチサイズを計算（BATCH_SIZEまたは総数の小さい方）
            first_batch = documents[:first_batch_size]  # 最初のバッチを切り出し

            print(  # 進捗表示
                f"  ⏳ バッチ 1/{(total_docs-1)//BATCH_SIZE + 1}: {len(first_batch)} チャンクを作成中..."  # 最初のバッチ情報
            )

            vectorstore = Chroma.from_documents(  # ChromaDBを新規作成（from_documentsメソッド使用）
                documents=first_batch,  # 最初のバッチのドキュメント
                embedding=embeddings,  # 埋め込みモデル
                collection_name=CHROMA_COLLECTION_NAME,  # コレクション名
                persist_directory=CHROMA_PERSIST_DIRECTORY,  # 保存ディレクトリ
            )
            print(f"     ✅ 完了")  # 作成完了メッセージ

            # 残りのドキュメントをバッチで追加
            for i in range(first_batch_size, total_docs, BATCH_SIZE):  # 2番目以降のバッチを処理
                batch_end = min(i + BATCH_SIZE, total_docs)  # バッチの終了位置を計算
                batch = documents[i:batch_end]  # 現在のバッチを切り出し

                print(  # 進捗表示
                    f"  ⏳ バッチ {i//BATCH_SIZE + 1}/{(total_docs-1)//BATCH_SIZE + 1}: {len(batch)} チャンクを追加中..."  # バッチ情報
                )

                try:  # エラー処理開始
                    vectorstore.add_documents(batch)  # バッチをベクトルストアに追加
                    print(f"     ✅ 完了")  # 成功メッセージ
                except Exception as e:  # エラーが発生した場合
                    print(f"     ❌ エラー: {e}")  # エラーメッセージを表示
                    # エラーが発生してもバッチ処理を継続（一部のチャンクが失敗しても全体を止めない）
                    continue  # 次のバッチに進む

                # APIレート制限対策のため、バッチ間で少し待機
                if batch_end < total_docs:  # 最後のバッチでない場合
                    time.sleep(1)  # 1秒待機（OpenAI APIのレート制限を避ける）

            print(f"  ✅ {total_docs} 個のチャンクを作成完了")  # 作成完了メッセージ

    return vectorstore  # 作成または更新したベクトルストアを返す


def main():
    """メイン処理"""
    print("=" * 60)  # 区切り線
    print("📚 ドキュメントインデクサー開始")  # 開始メッセージ
    print("=" * 60)  # 区切り線

    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):  # 環境変数OPENAI_API_KEYが設定されていない場合
        print("❌ エラー: OPENAI_API_KEYが設定されていません")  # エラーメッセージ
        print("   対処: .envファイルに追加してください")  # 対処方法
        return  # 処理を終了

    all_documents = []  # 全ドキュメントを格納するリスト

    # 1. Notionドキュメントを読み込む
    if os.path.exists(NOTION_DOCS_DIR):  # Notionドキュメントディレクトリが存在する場合
        print(f"\n📘 Notionドキュメントを読み込み中...")  # 読み込み開始メッセージ
        notion_docs = load_documents_from_directory(NOTION_DOCS_DIR, "notion")  # Notionドキュメントを読み込む
        all_documents.extend(notion_docs)  # 全ドキュメントリストに追加
        print(f"  → {len(notion_docs)} 個のドキュメントを読み込み")  # 読み込み数を表示
    else:  # Notionドキュメントディレクトリが存在しない場合
        print(f"\n⚠️  Notionドキュメントディレクトリが見つかりません: {NOTION_DOCS_DIR}")  # 警告メッセージ

    # 2. Google Driveドキュメントを読み込む
    if os.path.exists(GOOGLE_DOCS_DIR):  # Google Driveドキュメントディレクトリが存在する場合
        print(f"\n📗 Google Driveドキュメントを読み込み中...")  # 読み込み開始メッセージ
        google_docs = load_documents_from_directory(GOOGLE_DOCS_DIR, "google_drive")  # Google Driveドキュメントを読み込む
        all_documents.extend(google_docs)  # 全ドキュメントリストに追加
        print(f"  → {len(google_docs)} 個のドキュメントを読み込み")  # 読み込み数を表示
    else:  # Google Driveドキュメントディレクトリが存在しない場合
        print(  # 警告メッセージ（複数行にまたがるため）
            f"\n⚠️  Google Driveドキュメントディレクトリが見つかりません: {GOOGLE_DOCS_DIR}"
        )

    if not all_documents:  # ドキュメントが1つも読み込まれなかった場合
        print("\n❌ 処理するドキュメントがありません")  # エラーメッセージ
        print(  # 対処方法を表示
            "   先にdata_loader_notion.pyまたはdata_loader_google.pyを実行してください"
        )
        return  # 処理を終了

    # 3. ドキュメントをチャンクに分割
    print(f"\n✂️  ドキュメントをチャンクに分割中...")  # 分割開始メッセージ
    print(f"  設定: チャンクサイズ={CHUNK_SIZE}, オーバーラップ={CHUNK_OVERLAP}")  # 分割設定を表示
    split_docs = split_documents(all_documents)  # ドキュメントをチャンクに分割
    print(f"  → {len(split_docs)} 個のチャンクに分割完了")  # 分割結果を表示

    # 4. ベクトル化してChromaDBに保存
    print(f"\n🔄 ベクトル化とChromaDB保存中...")  # ベクトル化開始メッセージ
    print(f"  Embeddingモデル: {EMBEDDING_MODEL}")  # 使用する埋め込みモデルを表示

    try:  # エラー処理開始
        vectorstore = create_or_update_vectorstore(split_docs)  # ベクトルストアを作成または更新

        # 統計情報を表示
        collection = vectorstore._collection  # ChromaDBのコレクションオブジェクトを取得
        count = collection.count()  # コレクション内のドキュメント数をカウント

        print("\n" + "=" * 60)  # 区切り線
        print("✅ インデックス作成完了")  # 完了メッセージ
        print(f"  総チャンク数: {count}")  # 保存されたチャンク数を表示
        print(f"  保存先: {CHROMA_PERSIST_DIRECTORY}")  # ChromaDBの保存先を表示
        print(f"  コレクション名: {CHROMA_COLLECTION_NAME}")  # コレクション名を表示
        print("=" * 60)  # 区切り線

        # テスト検索
        print("\n🔍 テスト検索を実行...")  # テスト検索開始メッセージ
        test_query = "スマートグラフ"  # テスト用の検索クエリ
        results = vectorstore.similarity_search(test_query, k=3)  # 類似度検索を実行（上位3件を取得）

        if results:  # 検索結果がある場合
            print(f"  クエリ: '{test_query}'")  # 検索クエリを表示
            print(f"  検索結果: {len(results)} 件")  # 検索結果数を表示
            for i, doc in enumerate(results, 1):  # 各検索結果をループ（1から番号付け）
                title = doc.metadata.get("title", "不明")  # ドキュメントのタイトルを取得
                source = doc.metadata.get("source", "不明")  # ドキュメントのソースを取得
                content_preview = doc.page_content[:100].replace("\n", " ")  # 内容の先頭100文字を取得（改行をスペースに置換）
                print(f"    {i}. [{source}] {title}: {content_preview}...")  # 検索結果を表示

    except Exception as e:  # エラーが発生した場合
        print(f"\n❌ ベクトル化エラー: {e}")  # エラーメッセージを表示
        print("   OpenAI APIキーとネットワーク接続を確認してください")  # 対処方法を表示
        return  # 処理を終了


if __name__ == "__main__":  # このスクリプトが直接実行された場合（importされた場合は実行されない）
    main()  # メイン関数を実行
