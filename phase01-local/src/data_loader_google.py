#!/usr/bin/env python3
"""
Google Driveクローラー（Notion版構成準拠）
ルートフォルダからリンクされている全アイテムを再帰的に取得し、
テキストファイル形式でローカルに保存する。

【主な機能】
1. 指定したルートフォルダから開始して、全ファイル/フォルダを自動探索
2. ファイルの内容をテキストとして抽出し、ローカルに保存
3. 処理済みアイテムを記録し、中断後の再開が可能 (metadata.json)
4. エラー処理とレート制限対策を実装
"""

import os
import re
import json
import time
import traceback
import unicodedata  # Unicode正規化用（PDFテキストのクレンジングに使用）
from typing import List, Dict, Optional
from dotenv import load_dotenv
from datetime import datetime

# Google Drive API関連のインポート
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
import io

# PDF処理用ライブラリ（高精度レイアウト解析）
import pdfplumber

# ========================================
# PDFテキストクレンジング関数（RAG検索精度向上用）
# ========================================

def clean_pdf_text(text: str, keep_page_markers: bool = False) -> str:
    """
    PDFから抽出したテキストをクレンジングして検索精度を向上させる
    
    目的:
    - RAGシステムの検索精度向上のため、PDFから抽出した生のテキストに含まれる
      ノイズ（ページ番号、制御文字、過剰な空白など）を除去する
    
    処理内容:
    1. ページ区切り文字の除去（オプション）
    2. PDFの制御文字・特殊文字の除去
    3. 余分な空白・改行の正規化
    4. 文字化けの修正
    5. Unicode正規化
    
    Args:
        text: クレンジング対象のテキスト
        keep_page_markers: ページ区切りを残すかどうか（デフォルトはFalse）
    
    Returns:
        クレンジング済みのテキスト
    """
    # ステップ1: ページ区切り文字の処理
    # 「--- ページ X/Y ---」形式のマーカーを除去または正規化
    if not keep_page_markers:
        # ページマーカーを完全に除去（検索ノイズになるため）
        text = re.sub(r'---\s*ページ\s*\d+/\d+\s*---', '', text)
    else:
        # ページマーカーをシンプルな形式に正規化
        text = re.sub(r'---\s*ページ\s*(\d+)/(\d+)\s*---', r'[P\1]', text)
    
    # ステップ2: PDFの制御文字・不要な特殊文字を除去
    # 0x00-0x1F の制御文字（タブと改行を除く）を削除
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    
    # ステップ3: 連続する空白・改行の正規化
    # 3つ以上の連続改行を2つの改行に統一（段落区切りを保持）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 行内の連続スペースを1つに統一
    text = re.sub(r' {2,}', ' ', text)
    
    # 全角スペースを半角スペースに統一
    text = text.replace('　', ' ')
    
    # ステップ4: 文字化けや不要な記号の除去
    # よくある文字化けパターンを修正
    text = text.replace('', '')  # 文字化け記号
    text = text.replace('□', '')  # 判読不能文字の代替記号
    text = text.replace('●', '・')  # 箇条書き記号を統一
    
    # ステップ5: Unicode正規化（NFKC形式）
    # 半角カナを全角に、機種依存文字を標準文字に変換
    text = unicodedata.normalize('NFKC', text)
    
    # ステップ6: 前後の空白を除去
    text = text.strip()
    
    return text

# ========================================
# 初期設定とグローバル変数
# ========================================

# .envファイルから環境変数を読み込む（GOOGLE_DRIVE_CREDENTIALS_PATHなど）
load_dotenv()

# 新しい設定管理システムを使用
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# ファイル保存に関する設定
SAVE_DIR = "./data/documents/google/"  # ファイルの保存先ディレクトリ
METADATA_FILE = os.path.join(
    SAVE_DIR, "metadata.json"
)  # 処理状況を記録するメタデータファイル

# Google Drive APIサービスクライアントを格納するグローバル変数
# initialize_drive_service()関数で初期化される
drive_service = None

# 訪問済みページIDを記録するセット
# 同じアイテム（ファイル/フォルダ）を重複して処理しないよう、また循環参照を防ぐために使用
visited_page_ids: set = set()

# メタデータを保存する辞書
# クロール処理の状況、処理済みページ情報、エラー情報などを記録
metadata = {
    "crawl_date": datetime.now().isoformat(),  # クロール開始日時
    "pages": {},  # 処理済みページ（ファイル）の詳細情報
    "total_pages": 0,  # 処理済みページの総数
    "error_pages": [],  # エラーが発生したページ（ファイル/フォルダ）の情報
}

# Google DriveのMIMEタイプとエクスポート設定のマッピング
# テキスト系ファイルのみを対象とし、画像やバイナリファイルは除外
MIME_TYPE_MAPPING = {
    # Google Workspace形式（エクスポートが必要）
    "application/vnd.google-apps.document": {
        "export_mime": "text/plain",
        "extension": ".txt",
    },
    "application/vnd.google-apps.spreadsheet": {
        "export_mime": "text/csv",
        "extension": ".csv",
    },
    "application/vnd.google-apps.presentation": {
        "export_mime": "text/plain",
        "extension": ".txt",
    },
    # PDFファイル（テキスト抽出対象）
    "application/pdf": {"export_mime": None, "extension": ".txt", "is_pdf": True},
    # テキスト形式のファイルのみ（直接ダウンロード）
    "text/plain": {"export_mime": None, "extension": ".txt"},
    "text/csv": {"export_mime": None, "extension": ".csv"},
    "text/html": {"export_mime": None, "extension": ".html"},
    "text/xml": {"export_mime": None, "extension": ".xml"},
    "application/json": {"export_mime": None, "extension": ".json"},
    "text/markdown": {"export_mime": None, "extension": ".md"},
    "application/xml": {"export_mime": None, "extension": ".xml"},
    # プログラムソースコード
    "text/x-python": {"export_mime": None, "extension": ".py"},
    "text/javascript": {"export_mime": None, "extension": ".js"},
    "text/x-java": {"export_mime": None, "extension": ".java"},
}

# 除外するMIMEタイプのパターン（画像、動画、バイナリファイル）
# PDFは除外リストから削除（テキスト抽出対象とする）
EXCLUDED_MIME_PATTERNS = [
    "image/",  # 全ての画像形式（image/png, image/jpeg, image/gif等）
    "video/",  # 全ての動画形式
    "audio/",  # 全ての音声形式
    "application/zip",  # 圧縮ファイル
    "application/x-rar",  # RARファイル
    "application/octet-stream",  # 一般的なバイナリファイル
    "application/vnd.openxmlformats",  # MS Office形式（.docx, .xlsx等）
    "application/postscript",  # PostScriptファイル（.ai等）
]


def load_existing_metadata():
    """
    既存のメタデータを読み込む（再開可能にするため）。
    スクリプト開始時に呼び出され、前回の処理状況を復元する。
    """
    # グローバル変数をこの関数内で変更するためglobal宣言
    global metadata, visited_page_ids

    # metadata.jsonファイルが存在するかチェック
    if os.path.exists(METADATA_FILE):
        try:
            # JSONファイルを読み込みモードでオープン
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                # JSONデータをPythonの辞書として読み込み
                existing_data = json.load(f)

                # 既存のpagesデータをメタデータにコピー（キーが存在しない場合は空の辞書をデフォルト値とする）
                metadata["pages"] = existing_data.get("pages", {})
                # 処理済みページ数を復元（キーが存在しない場合は0をデフォルト値とする）
                metadata["total_pages"] = existing_data.get("total_pages", 0)
                # エラーページのリストを復元（キーが存在しない場合は空リストをデフォルト値とする）
                metadata["error_pages"] = existing_data.get("error_pages", [])

                # 重要：pagesデータのキー（ページID）をsetに変換してvisited_page_idsに格納
                # これにより、既に処理したページを再処理しないようにする
                visited_page_ids = set(metadata["pages"].keys())
                print(f"📂 既存データ読み込み: {len(visited_page_ids)}ページ処理済み")
                return True  # 読み込み成功
        except Exception as e:
            # JSON読み込み中、またはデータ形式が不正な場合にエラーを出力
            print(f"⚠️  既存メタデータ読み込みエラー: {e}")
    return False  # ファイルが存在しないか、読み込みに失敗した場合


def save_metadata():
    """
    現在のメタデータをJSONファイルとして保存する。
    各アイテム処理後に呼び出され、プログラムが中断されても進捗が失われないようにする。
    """
    try:
        # メタデータ保存用のファイルを開く（書き込みモード）
        # ensure_ascii=Falseで日本語がそのまま保存され、indent=2で見やすい形式にする
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # ファイルへの書き込み権限がない場合などのエラーを捕捉
        print(f"❌ メタデータ保存エラー: {e}")


def is_excluded_mime_type(mime_type: str) -> bool:
    """
    MIMEタイプが除外対象かどうかをチェックする。
    画像、動画、バイナリファイルなどを除外する。
    """
    if not mime_type:
        return False
    
    # 除外パターンのいずれかに該当するかチェック
    for pattern in EXCLUDED_MIME_PATTERNS:
        if mime_type.startswith(pattern):
            return True
    return False


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """
    ファイル名として使えない文字を置換し、長さを制限する。
    OSのファイルシステムで安全に扱えるファイル名を生成する。
    """
    # 正規表現でOSで使用できない文字（\ / * ? : " < > | および改行など）をアンダースコアに置換
    name = re.sub(r'[\\/*?:"<>|\n\r\t]', "_", name)
    # 連続したアンダースコアを単一のアンダースコアに置換（例: "file___name" -> "file_name"）
    name = re.sub(r"_+", "_", name)
    # 先頭と末尾の不要なスペースやタブを削除
    name = name.strip()
    # ファイル名が指定の最大長（多くのファイルシステムで255文字制限）を超える場合は切り詰める
    if len(name) > max_length:
        name = name[:max_length]
    # 全ての処理の結果、ファイル名が空文字列になった場合は "Untitled" を返す
    return name if name else "Untitled"


def initialize_drive_service():
    """
    Google Drive APIサービスを初期化し、グローバル変数に格納する。
    メイン処理の開始前に一度だけ呼び出される。
    """
    # グローバル変数をこの関数内で変更するためglobal宣言
    global drive_service
    # .envファイルからサービスアカウントの認証情報ファイルへのパスを取得
    creds_path = config.google_drive_credentials_path
    
    # 相対パスの場合、プロジェクトルートからの相対パスとして解決
    if creds_path and not os.path.isabs(creds_path):
        # スクリプトの場所から2階層上がプロジェクトルート
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        creds_path = os.path.join(project_root, creds_path.lstrip('./'))
    
    # 認証情報ファイルが存在しない場合はエラーを発生させる
    if not creds_path or not os.path.exists(creds_path):
        raise FileNotFoundError(f"認証ファイルが見つかりません: {creds_path}")
    try:
        # サービスアカウントキーファイルから認証情報を生成
        # スコープは 'drive.readonly' とし、読み取り専用の権限を要求する
        credentials = service_account.Credentials.from_service_account_file(
            creds_path, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        # 認証情報を使用してGoogle Drive APIのクライアントを構築
        drive_service = build("drive", "v3", credentials=credentials)
        print("✅ Google Drive API接続成功")
    except Exception as e:
        # 認証情報の形式が不正な場合や、APIへの接続に失敗した場合のエラー
        print(f"❌ Google Drive API初期化エラー: {e}")
        raise


def download_file_content(file_id: str, mime_type: str) -> Optional[str]:
    """
    指定されたファイルのIDとMIMEタイプに基づき、内容をダウンロードしてテキストとして返す。
    Googleドキュメントなどはエクスポートし、PDFはテキスト抽出、その他は直接ダウンロードする。
    """
    try:
        # MIMEタイプに対応するエクスポート設定を取得
        mime_info = MIME_TYPE_MAPPING.get(mime_type, {})
        export_mime = mime_info.get("export_mime")
        is_pdf = mime_info.get("is_pdf", False)

        # Googleドキュメントなど、エクスポートが必要なファイルの場合
        if export_mime:
            request = drive_service.files().export_media(
                fileId=file_id, mimeType=export_mime
            )
        # PDFなど、直接ダウンロード可能なファイルの場合
        else:
            request = drive_service.files().get_media(fileId=file_id)

        # ダウンロードした内容をメモリ上のバイナリデータとして保持
        file_content_io = io.BytesIO()
        # ダウンロードを実行するダウンローダーを初期化
        downloader = MediaIoBaseDownload(file_content_io, request)

        # チャンクごとにダウンロードを実行
        done = False
        while not done:
            _, done = downloader.next_chunk()  # statusは使用しない

        # PDFファイルの場合、pdfplumberでテキスト抽出
        if is_pdf:
            try:
                # バイトデータを先頭に巻き戻す
                file_content_io.seek(0)
                
                # pdfplumberでPDFを開く
                with pdfplumber.open(file_content_io) as pdf:
                    text_content = ""
                    total_pages = len(pdf.pages)
                    
                    # 各ページからテキストを抽出
                    for i, page in enumerate(pdf.pages, 1):
                        # pdfplumberの高精度テキスト抽出
                        # extract_text()は自動的にレイアウトを解析し、適切な改行を維持
                        page_text = page.extract_text()
                        
                        if page_text:  # テキストが抽出できた場合
                            # ページ区切りを追加（後でクレンジングで処理される）
                            text_content += f"\n--- ページ {i}/{total_pages} ---\n"
                            text_content += page_text
                
                # テキストが抽出できなかった場合
                if not text_content.strip():
                    print(f"      ⚠️ PDFからテキストを抽出できませんでした")
                    return None
                
                # 抽出したPDFテキストをクレンジング
                # RAG検索精度向上のため、ノイズを除去
                original_length = len(text_content)
                text_content = clean_pdf_text(text_content, keep_page_markers=False)
                cleaned_length = len(text_content)
                
                print(f"      📄 PDFから {original_length} 文字を抽出")
                print(f"      🧹 クレンジング後: {cleaned_length} 文字 (削減率: {100*(1-cleaned_length/original_length):.1f}%)")
                
                return text_content
                
            except Exception as pdf_error:
                print(f"      ❌ PDF処理エラー: {pdf_error}")
                # デバッグ情報を追加
                print(f"         エラータイプ: {type(pdf_error).__name__}")
                return None
        
        # PDF以外のファイルの場合、通常のテキストデコード処理
        else:
            # ダウンロードしたバイナリデータをテキスト（UTF-8）にデコード
            try:
                return file_content_io.getvalue().decode("utf-8")
            # UTF-8でデコードできない場合は、latin-1でエラーを無視して強制的にデコード
            except UnicodeDecodeError:
                return file_content_io.getvalue().decode("latin-1", errors="ignore")

    except Exception as e:
        # エラーの詳細を表示
        error_type = type(e).__name__
        print(f"    ❌ ダウンロードエラー ({error_type}): {e}")
        
        # HTTPエラーの場合は追加情報を表示
        if isinstance(e, HttpError) and hasattr(e, 'resp') and e.resp:
            print(f"       HTTPステータス: {e.resp.status}")
        
        return None


def get_file_properties(file_info: Dict) -> Dict:
    """
    Google Drive APIから取得したファイル情報から、主要なプロパティを抽出・整形して返す。
    メタデータとして保存するためのヘルパー関数。
    """
    # 所有者情報が存在する場合、最初の所有者の名前とメールアドレスを取得
    owner_info = file_info.get("owners", [{}])[0]

    # 整形したプロパティを格納する辞書
    properties = {
        "drive_id": file_info.get("id"),
        "name": file_info.get("name"),
        "mime_type": file_info.get("mimeType"),
        "drive_link": file_info.get("webViewLink"),
        "created_time": file_info.get("createdTime"),
        "modified_time": file_info.get("modifiedTime"),
        "size_bytes": file_info.get("size"),
        "owner_name": owner_info.get("displayName"),
        "owner_email": owner_info.get("emailAddress"),
    }
    return properties


def traverse_and_save(item_id: str, parent_titles: List[str], depth: int = 0):
    """
    Driveのアイテム（ファイル/フォルダ）を再帰的に辿り、コンテンツをローカルに保存する中心的な関数。
    parent_titlesには親フォルダのタイトルがリストとして渡される（Notion版と同じ構造）。
    """
    # 1. 訪問済みチェック: 既に処理済みのアイテムはスキップし、無限ループを防ぐ
    if item_id in visited_page_ids:
        print(f"{'  ' * depth}⏭️  既訪問: {item_id[:15]}...")
        return

    # 2. レート制限対策: APIへの過度なリクエストを防ぐため、短い待機時間を挿入
    time.sleep(0.1)

    try:
        # 3. アイテムのメタデータをAPIから取得
        item_info = (
            drive_service.files()
            .get(
                fileId=item_id,
                fields="id, name, mimeType, webViewLink, createdTime, modifiedTime, owners, size",
            )
            .execute()
        )

        # アイテム名とMIMEタイプを取得
        item_name = item_info.get("name", "Untitled")
        item_mime_type = item_info.get("mimeType")

        # 4. 進捗表示: フォルダかファイルかでアイコンを変えて表示
        icon = "📁" if item_mime_type == "application/vnd.google-apps.folder" else "📄"
        print(f"{'  ' * depth}{icon} 処理中: {item_name} ({item_id[:15]}...)")

        # 5. アイテムの種類（フォルダかファイルか）に応じて処理を分岐
        # 【フォルダの場合】
        if item_mime_type == "application/vnd.google-apps.folder":
            # フォルダ自体を訪問済みとして記録
            visited_page_ids.add(item_id)
            # 現在のパスタイトルのリストを更新（親タイトル + 現在のフォルダ名）
            current_path_titles = parent_titles + [item_name]

            # フォルダ内のアイテムリストを取得するためのクエリ
            try:
                query = f"'{item_id}' in parents and trashed=false"
                # APIを呼び出して子アイテムのリストを取得
                response = drive_service.files().list(q=query, fields="files(id)").execute()

                # 各子アイテムに対して、この関数自身を再帰的に呼び出す
                for child_item in response.get("files", []):
                    traverse_and_save(child_item["id"], current_path_titles, depth + 1)
            except HttpError as folder_error:
                # フォルダアクセスエラーの場合、エラーを記録して処理を継続
                print(f"{'  ' * (depth+1)}⚠️  フォルダアクセスエラー: {folder_error.resp.status if hasattr(folder_error, 'resp') else folder_error}")
                metadata["error_pages"].append({
                    "id": item_id,
                    "error": str(folder_error),
                    "error_type": "HttpError",
                    "title": item_name,
                    "timestamp": datetime.now().isoformat()
                })
                save_metadata()

        # 【ファイルの場合】
        elif item_mime_type in MIME_TYPE_MAPPING:
            # ファイルを訪問済みとして記録
            visited_page_ids.add(item_id)

            # ファイルの内容をテキストとしてダウンロード
            content = download_file_content(item_id, item_mime_type)
            # ダウンロードに失敗した場合はエラーとして記録し、スキップ
            if content is None:
                print(f"{'  ' * (depth+1)}⚠️  ダウンロード失敗: {item_name}")
                metadata["error_pages"].append({
                    "id": item_id,
                    "error": "コンテンツのダウンロードに失敗しました。",
                    "error_type": "DownloadError",
                    "title": item_name,
                    "timestamp": datetime.now().isoformat()
                })
                save_metadata()
                return  # このファイルの処理をスキップ

            # ファイルのプロパティ情報を整形して取得
            properties = get_file_properties(item_info)

            # 現在のパスタイトルを更新（親タイトル + 現在のファイル名）
            current_path_titles = parent_titles + [item_name]
            
            # Notion版と同じ形式でファイル名を生成（階層をハイフンで連結）
            # 例: "Root-Folder1-Folder2-Document.txt"
            file_extension = MIME_TYPE_MAPPING[item_mime_type].get("extension", ".txt")
            filename = sanitize_filename("-".join(current_path_titles)) + file_extension
            # フラット構造で保存（Notion版と同じ）
            filepath = os.path.join(SAVE_DIR, filename)

            # ファイルにコンテンツを書き込む（UTF-8エンコーディング）
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"{'  ' * (depth+1)}✅ 保存完了: {filename}")

            # メタデータ辞書にこのファイルの情報を追加（Notion版と同じ構造）
            metadata["pages"][item_id] = {
                "title": item_name,  # "name"から"title"に変更
                "type": "file",
                "path": filepath,
                "parent_titles": parent_titles,  # 親タイトルのリストを追加
                "properties": properties,
                "child_pages": []  # 子ページリスト（ファイルなので空）
            }
            # 総ページ数をインクリメント
            metadata["total_pages"] += 1
            # 変更を即座に`metadata.json`に保存
            save_metadata()

        # 【未対応のファイル形式の場合】
        else:
            # 除外対象のMIMEタイプかチェック
            if is_excluded_mime_type(item_mime_type):
                print(
                    f"{'  ' * (depth+1)}🚫 除外: {item_name} ({item_mime_type}) - バイナリ/画像ファイル"
                )
            else:
                print(
                    f"{'  ' * (depth+1)}⏭️  未対応形式のためスキップ: {item_name} ({item_mime_type})"
                )

    # 6. エラー処理: APIエラーやその他の例外が発生した場合
    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        
        print(f"{'  ' * depth}❌ エラー発生 (ID: {item_id[:15]}...)")
        print(f"{'  ' * (depth+1)}  タイプ: {error_type}")
        print(f"{'  ' * (depth+1)}  詳細: {error_details}")
        
        # デバッグ用にスタックトレースの最終行を表示
        tb_lines = traceback.format_exc().split('\n')
        last_error_line = [line for line in tb_lines if line.strip() and 'line' in line.lower()][-1:]
        if last_error_line:
            print(f"{'  ' * (depth+1)}  発生箇所: {last_error_line[0].strip()}")
        
        # エラー情報をメタデータに追加
        metadata["error_pages"].append({
            "id": item_id,
            "error": error_details,
            "error_type": error_type,
            "title": locals().get('item_name', 'Unknown'),
            "timestamp": datetime.now().isoformat()
        })
        save_metadata()


def main():
    """
    メイン実行関数。
    全体の処理フローを制御する。
    """
    # 処理開始のヘッダーを表示
    print("\n" + "=" * 60)
    print("🚀 Google Drive クローラー開始")
    print("=" * 60)
    
    # .envファイルから処理対象のルートフォルダID（カンマ区切り）を取得
    root_folder_ids_str = ','.join(config.google_drive_folder_ids) if config.google_drive_folder_ids else ""
    # 文字列をカンマで分割し、リストに変換
    root_folder_ids = [
        id.strip() for id in root_folder_ids_str.split(",") if id.strip()
    ]

    # 1. 環境変数チェック
    creds_path = config.google_drive_credentials_path
    if not creds_path:
        print("❌ エラー: GOOGLE_DRIVE_CREDENTIALS_PATHが設定されていません")
        print("   対処: .envファイルに追加してください")
        return
    
    # 相対パスの場合、プロジェクトルートからの相対パスとして解決
    if not os.path.isabs(creds_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        creds_path = os.path.join(project_root, creds_path.lstrip('./'))
    
    if not os.path.exists(creds_path):
        print(f"❌ エラー: 認証ファイルが見つかりません: {creds_path}")
        return
        
    if not root_folder_ids:
        print("❌ エラー: GOOGLE_DRIVE_FOLDER_IDSが設定されていません")
        print("   対処: .envファイルに追加してください")
        return

    # 2. Google Drive APIサービスを初期化
    try:
        print("\n📡 Google Drive API接続中...")
        initialize_drive_service()
    except Exception as e:
        print(f"❌ API初期化エラー: {e}")
        print(f"   エラータイプ: {type(e).__name__}")
        return

    # 3. 保存先ディレクトリを作成
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📁 保存先: {SAVE_DIR}")
    print(f"🏠 ルートフォルダ数: {len(root_folder_ids)}")

    # 4. 既存のメタデータを読み込み（再開処理）
    if load_existing_metadata():
        print("📝 前回の処理を継続します")
    else:
        print("🆕 新規処理を開始します")
    print("-" * 60)

    # 処理時間の計測を開始
    start_time = time.time()
    # メタデータにクロール開始日時を記録し、初回保存
    metadata["crawl_date"] = datetime.now().isoformat()
    save_metadata()

    # 5. 各ルートフォルダから処理を開始
    for folder_id in root_folder_ids:
        # 空のリストを親タイトルとして開始（ルートレベル）
        traverse_and_save(folder_id, [], 0)

    # 処理時間の計測を終了
    elapsed_time = time.time() - start_time

    # 6. 最終統計をメタデータに更新して保存
    metadata["crawl_completed"] = datetime.now().isoformat()
    metadata["total_time_seconds"] = elapsed_time
    save_metadata()

    # 7. 最終結果のサマリーを表示
    print("-" * 60)
    print("📈 統計情報")
    print(f"  - 総処理ページ数: {metadata['total_pages']}")
    print(f"  - エラーページ数: {len(metadata['error_pages'])}")
    print(f"  - 処理時間: {elapsed_time:.2f}秒")
    print("-" * 60)

    # エラーが発生したページがあれば簡潔にリストアップ
    if metadata["error_pages"]:
        print("\n⚠️  エラーが発生したページ:")
        for error in metadata["error_pages"]:
            title = error.get('title', 'N/A')
            error_msg = error['error'][:100] + '...' if len(error['error']) > 100 else error['error']
            print(f"  - {title}: {error_msg}")
        print("-" * 60)

    print("✅ すべての処理が完了しました。")
    print(f"📊 メタデータは {METADATA_FILE} に保存されました。")
    print("=" * 60)


if __name__ == "__main__":
    # このスクリプトが直接実行された場合にmain()関数を呼び出す
    main()
