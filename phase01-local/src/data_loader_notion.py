#!/usr/bin/env python3
"""
Notionクローラー（改善版）
ルートページからリンクされている全ページを再帰的に取得し、
Markdown形式でローカルに保存する。

【主な機能】
1. 指定したルートページから開始して、リンクされている全ページを自動探索
2. ページ、データベース、子ページを再帰的に取得
3. Notion APIから取得したコンテンツをMarkdown形式に変換
4. 処理済みページの記録により、中断後の再開が可能
5. エラー処理とレート制限対策を実装
"""

import os
import re
import json
import time
import unicodedata  # Unicode正規化用（テキストクレンジングに使用）
from typing import List, Dict, Set, Optional, Tuple
from dotenv import load_dotenv
import notion_client  # Notion公式APIクライアント（v2.4.0）
from datetime import datetime

# 新しい設定管理システムを使用
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import config  # 一時的にコメントアウト

# ========================================
# Notionテキストクレンジング関数（RAG検索精度向上用）
# ========================================

def clean_notion_text(text: str) -> str:
    """
    Notionから取得したテキストをクレンジングして検索精度を向上させる
    
    目的:
    - RAGシステムの検索精度向上のため、Notionから取得したMarkdownテキストに含まれる
      ノイズ（HTMLタグ、Notion固有の記法、過剰な装飾など）を除去する
    
    処理内容:
    1. HTMLタグの除去（<u>、<details>、<summary>など）
    2. 画像・動画・ファイル埋め込みの簡略化
    3. リンクの簡略化
    4. Notion固有の記法の除去（絵文字アイコンなど）
    5. 余分な空白・改行の正規化
    6. Unicode正規化
    
    Args:
        text: クレンジング対象のテキスト
    
    Returns:
        クレンジング済みのテキスト
    """
    # ステップ1: HTMLタグの除去
    # <u>下線</u>、<details>、<summary>などのHTMLタグを除去
    text = re.sub(r'<[^>]+>', '', text)
    
    # ステップ2: 画像・動画・ファイル埋め込みの除去または簡略化
    # ![alt](url) 形式の画像を除去（検索ノイズになるため）
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    
    # [📹 Video: ...](url) 形式の動画リンクを簡略化
    text = re.sub(r'\[📹[^\]]+\]\([^\)]+\)', '[動画]', text)
    
    # [📎 File: ...](url) 形式のファイルリンクを簡略化
    text = re.sub(r'\[📎[^\]]+\]\([^\)]+\)', '[ファイル]', text)
    
    # [📄 PDF: ...](url) 形式のPDFリンクを簡略化
    text = re.sub(r'\[📄[^\]]+\]\([^\)]+\)', '[PDF]', text)
    
    # ステップ3: Notion固有の記法の除去
    # notion://xxxxx 形式の内部リンクを除去
    text = re.sub(r'notion://[^\s\)]+', '', text)
    
    # [🔖 ...](url) 形式のブックマークを簡略化
    text = re.sub(r'\[🔖[^\]]+\]\([^\)]+\)', '', text)
    
    # [🌐 Embed: ...](url) 形式の埋め込みを除去
    text = re.sub(r'\[🌐[^\]]+\]\([^\)]+\)', '', text)
    
    # [[TOC]] 目次記法を除去
    text = re.sub(r'\[\[TOC\]\]', '', text)
    
    # ステップ4: 絵文字アイコンの正規化（必要に応じて除去）
    # 🗄️、📊、📄などのアイコンを除去（オプション）
    # text = re.sub(r'[🗄📊📄📎📹🔖🌐💡]', '', text)
    
    # ステップ5: Markdownの装飾記号の簡略化
    # 過剰な装飾（太字、斜体の組み合わせなど）を簡略化
    text = re.sub(r'\*{3,}', '**', text)  # ***を**に
    text = re.sub(r'~{3,}', '~~', text)    # ~~~を~~に
    
    # ステップ6: コードブロックの言語指定を除去（検索時のノイズ削減）
    # ```python → ```
    text = re.sub(r'```[a-zA-Z]+\n', '```\n', text)
    
    # ステップ7: 連続する空白・改行の正規化
    # 3つ以上の連続改行を2つの改行に統一
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 行内の連続スペースを1つに統一
    text = re.sub(r' {2,}', ' ', text)
    
    # 全角スペースを半角スペースに統一
    text = text.replace('　', ' ')
    
    # ステップ8: 区切り線の正規化
    # 様々な形式の区切り線を統一
    text = re.sub(r'^-{3,}$', '---', text, flags=re.MULTILINE)
    text = re.sub(r'^={3,}$', '---', text, flags=re.MULTILINE)
    
    # ステップ9: Unicode正規化（NFKC形式）
    # 半角カナを全角に、機種依存文字を標準文字に変換
    text = unicodedata.normalize('NFKC', text)
    
    # ステップ10: 前後の空白を除去
    text = text.strip()
    
    return text

# ========================================
# 初期設定とグローバル変数
# ========================================

# .envファイルから環境変数を読み込む（NOTION_TOKENなど）
load_dotenv()

# Notion APIクライアントを初期化
# NOTION_TOKEN: Notion Integration Tokenで、Notion APIへのアクセス権限を持つ
NOTION_TOKEN = None
notion = None  # main()関数内で初期化

# ファイル保存に関する設定
SAVE_DIR = "./data/documents/notion/"  # Markdownファイルの保存先ディレクトリ
METADATA_FILE = os.path.join(SAVE_DIR, "metadata.json")  # 処理状況を記録するメタデータファイル

# 訪問済みページIDを記録するセット
# 同じページを重複して処理しないよう、また循環参照による無限ループを防ぐために使用
visited_page_ids = set()

# メタデータを保存する辞書
# クロール処理の状況、処理済みページ情報、エラー情報などを記録
metadata = {
    "crawl_date": datetime.now().isoformat(),  # クロール開始日時
    "pages": {},  # 処理済みページの詳細情報
    "total_pages": 0,  # 処理済みページの総数
    "error_pages": [],  # エラーが発生したページの情報
}


def load_existing_metadata():
    """既存のメタデータを読み込む（再開可能にするため）"""
    # グローバル変数を変更するためglobal宣言
    global metadata, visited_page_ids

    # metadata.jsonファイルが存在するかチェック
    if os.path.exists(METADATA_FILE):
        try:
            # JSONファイルを読み込みモードでオープン
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                # JSONデータをPythonの辞書として読み込み
                existing_data = json.load(f)
                
                # 既存のpagesデータをメタデータにコピー（空の辞書がデフォルト）
                metadata["pages"] = existing_data.get("pages", {})
                # 処理済みページ数を復元（0がデフォルト）
                metadata["total_pages"] = existing_data.get("total_pages", 0)
                # エラーページのリストを復元（空リストがデフォルト）
                metadata["error_pages"] = existing_data.get("error_pages", [])
                
                # 重要：pagesデータのキー（ページID）をsetに変換してvisited_page_idsに格納
                # これにより、既に処理したページを再処理しないようにする
                visited_page_ids = set(metadata["pages"].keys())
                print(f"📂 既存データ読み込み: {len(visited_page_ids)}ページ処理済み")
                return True  # 読み込み成功
        except Exception as e:
            # JSON読み込みエラー時の処理
            print(f"⚠️  既存メタデータ読み込みエラー: {e}")
    return False  # ファイルが存在しないか読み込み失敗


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """ファイル名として使えない文字を置換し、長さを制限する"""
    # 正規表現でOSで使用できない文字をアンダースコアに置換
    # \\ : バックスラッシュ（Windowsのパス区切り文字）
    # /   : スラッシュ（Unix系のパス区切り文字）
    # *   : ワイルドカード
    # ?   : ワイルドカード（単一文字）
    # :   : コロン（Windowsのドライブ文字）
    # "   : ダブルクォート
    # <>  : リダイレクト文字
    # |   : パイプ
    # \n\r\t : 改行、キャリッジリターン、タブ
    name = re.sub(r'[\\/*?:"<>|\n\r\t]', "_", name)
    
    # 連続したアンダースコアを単一のアンダースコアに置換
    # 例: "test___file" → "test_file"
    name = re.sub(r"_+", "_", name)
    
    # 先頭と末尾のスペース、タブ、改行などを削除
    name = name.strip()
    
    # ファイル名が指定の最大長を超える場合は切り詰める
    # 一部のファイルシステムは255文字制限があるため
    if len(name) > max_length:
        name = name[:max_length]  # 先頭からmax_length文字までを取得
    
    # ファイル名が空文字列になった場合はデフォルト名を返す
    # 三項演算子: nameが真値（空でない）ならname、偶値（空）なら"Untitled"
    return name if name else "Untitled"


def extract_text_from_rich_text(rich_text_array: List[Dict]) -> str:
    """リッチテキスト配列からテキストを抽出し、マークダウン形式で返す"""
    # 配列が空またはNoneの場合は空文字列を返す
    if not rich_text_array:
        return ""

    # 変換結果を格納するリスト
    result = []
    
    # リッチテキスト配列の各要素を処理
    for rt in rich_text_array:
        # plain_textフィールドからプレーンテキストを取得（なければ空文字列）
        text = rt.get("plain_text", "")
        
        # テキストが空の場合は次の要素へスキップ
        if not text:
            continue

        # annotationsフィールドから装飾情報を取得（なければ空の辞書）
        annotations = rt.get("annotations", {})

        # hrefフィールドがある場合はMarkdownのリンク形式に変換
        # [表示テキスト](URL)の形式
        if rt.get("href"):
            text = f"[{text}]({rt['href']})"

        # トリッキーな部分：装飾の適用順序が重要
        # リンクを先に処理し、その後に装飾を適用する
        
        # bold=Trueの場合、**で囲む（Markdownの太字）
        if annotations.get("bold"):
            text = f"**{text}**"
        
        # italic=Trueの場合、*で囲む（Markdownの斜体）
        if annotations.get("italic"):
            text = f"*{text}*"
        
        # strikethrough=Trueの場合、~~で囲む（Markdownの取り消し線）
        if annotations.get("strikethrough"):
            text = f"~~{text}~~"
        
        # underline=Trueの場合、<u>タグで囲む（Markdownには下線がないのでHTML）
        if annotations.get("underline"):
            text = f"<u>{text}</u>"
        
        # code=Trueの場合、`で囲む（インラインコード）
        if annotations.get("code"):
            text = f"`{text}`"

        # 変換後のテキストを結果リストに追加
        result.append(text)

    # リストの全要素を連結して1つの文字列として返す
    return "".join(result)


def convert_block_to_markdown(
    block: Dict, indent_level: int = 0
) -> Tuple[str, List[str]]:
    """
    Notionの1つのブロックをMarkdown文字列に変換する
    
    【処理内容】
    Notionのコンテンツは「ブロック」単位で構成されている。
    この関数は、各ブロックタイプ（見出し、段落、リスト、画像など）を
    適切なMarkdown記法に変換する。また、子ページへのリンクがある場合は
    そのIDを収集して返す。
    
    【対応ブロックタイプ】
    - heading_1/2/3: 見出し（#、##、###）
    - paragraph: 段落
    - bulleted_list_item: 箇条書き（*）
    - numbered_list_item: 番号付きリスト（1.）
    - to_do: ToDoリスト（- [ ] または - [x]）
    - toggle: 折りたたみ可能なコンテンツ（<details>タグ）
    - code: コードブロック（```）
    - quote: 引用（>）
    - callout: コールアウト（絵文字付き引用）
    - divider: 区切り線（---）
    - image/video/file/pdf: メディアファイル
    - bookmark: ブックマーク
    - equation: 数式（$$）
    - table_of_contents: 目次
    - link_to_page/child_page/child_database: ページリンク
    - embed: 埋め込みコンテンツ
    - table: テーブル
    - column_list/column: カラムレイアウト
    - synced_block: 同期ブロック
    
    Args:
        block: Notion APIから取得したブロックオブジェクト
        indent_level: インデントレベル（ネストされたブロック用）
    
    Returns:
        Tuple[str, List[str]]: 
            - Markdown形式に変換されたテキスト
            - 発見した子ページ/データベースのIDリスト
    """
    block_type = block.get("type")  # ブロックのタイプを取得
    content = block.get(block_type, {})  # ブロックタイプに対応するコンテンツを取得
    indent = "  " * indent_level  # インデント用のスペース
    child_page_ids = []  # 子ページIDを格納するリスト

    # リッチテキストの処理（ほとんどのブロックで使用）
    text_content = extract_text_from_rich_text(content.get("rich_text", []))

    # ========================================
    # ブロックタイプ別の処理
    # ========================================
    if block_type == "heading_1":
        return f"{indent}# {text_content}\n\n", child_page_ids
    elif block_type == "heading_2":
        return f"{indent}## {text_content}\n\n", child_page_ids
    elif block_type == "heading_3":
        return f"{indent}### {text_content}\n\n", child_page_ids
    elif block_type == "paragraph":
        return f"{indent}{text_content}\n\n" if text_content else "", child_page_ids
    elif block_type == "bulleted_list_item":
        return f"{indent}* {text_content}\n", child_page_ids
    elif block_type == "numbered_list_item":
        return f"{indent}1. {text_content}\n", child_page_ids
    elif block_type == "to_do":
        checked = "x" if content.get("checked", False) else " "
        return f"{indent}- [{checked}] {text_content}\n", child_page_ids
    elif block_type == "toggle":
        return (
            f"{indent}<details>\n{indent}<summary>{text_content}</summary>\n\n",
            child_page_ids,
        )
    elif block_type == "code":
        language = content.get("language", "")
        code_text = extract_text_from_rich_text(content.get("rich_text", []))
        return f"{indent}```{language}\n{code_text}\n{indent}```\n\n", child_page_ids
    elif block_type == "quote":
        return f"{indent}> {text_content}\n\n", child_page_ids
    elif block_type == "callout":
        emoji = (
            content.get("icon", {}).get("emoji", "💡") if content.get("icon") else "💡"
        )
        return f"{indent}> {emoji} {text_content}\n\n", child_page_ids
    elif block_type == "divider":
        return f"{indent}---\n\n", child_page_ids
    elif block_type == "image":
        image_url = ""
        if content.get("type") == "external":
            image_url = content.get("external", {}).get("url", "")
        elif content.get("type") == "file":
            image_url = content.get("file", {}).get("url", "")
        caption = extract_text_from_rich_text(content.get("caption", []))
        return f"{indent}![{caption}]({image_url})\n\n", child_page_ids
    elif block_type == "video":
        video_url = ""
        if content.get("type") == "external":
            video_url = content.get("external", {}).get("url", "")
        elif content.get("type") == "file":
            video_url = content.get("file", {}).get("url", "")
        caption = extract_text_from_rich_text(content.get("caption", []))
        return (
            f"{indent}[📹 Video: {caption or 'Video'}]({video_url})\n\n",
            child_page_ids,
        )
    elif block_type == "file":
        file_url = ""
        if content.get("type") == "external":
            file_url = content.get("external", {}).get("url", "")
        elif content.get("type") == "file":
            file_url = content.get("file", {}).get("url", "")
        caption = extract_text_from_rich_text(content.get("caption", []))
        return f"{indent}[📎 File: {caption or 'File'}]({file_url})\n\n", child_page_ids
    elif block_type == "pdf":
        pdf_url = ""
        if content.get("type") == "external":
            pdf_url = content.get("external", {}).get("url", "")
        elif content.get("type") == "file":
            pdf_url = content.get("file", {}).get("url", "")
        caption = extract_text_from_rich_text(content.get("caption", []))
        return f"{indent}[📄 PDF: {caption or 'PDF'}]({pdf_url})\n\n", child_page_ids
    elif block_type == "bookmark":
        url = content.get("url", "")
        caption = extract_text_from_rich_text(content.get("caption", []))
        return f"{indent}[🔖 {caption or url}]({url})\n\n", child_page_ids
    elif block_type == "equation":
        expression = content.get("expression", "")
        return f"{indent}$$\n{expression}\n$$\n\n", child_page_ids
    elif block_type == "table_of_contents":
        return f"{indent}[[TOC]]\n\n", child_page_ids
    elif block_type == "link_to_page":
        page_id = content.get("page_id", "")
        if page_id:
            child_page_ids.append(page_id)
            return f"{indent}[📄 Page Link](notion://{page_id})\n\n", child_page_ids
    elif block_type == "child_page":
        page_id = block.get("id", "")
        title = content.get("title", "Untitled")
        if page_id:
            child_page_ids.append(page_id)
            return f"{indent}[📄 {title}](notion://{page_id})\n\n", child_page_ids
    elif block_type == "child_database":
        db_id = block.get("id", "")
        title = content.get("title", "Database")
        if db_id:
            child_page_ids.append(db_id)
            return f"{indent}[🗄️ {title}](notion://{db_id})\n\n", child_page_ids
    elif block_type == "embed":
        url = content.get("url", "")
        caption = extract_text_from_rich_text(content.get("caption", []))
        return f"{indent}[🌐 Embed: {caption or url}]({url})\n\n", child_page_ids
    elif block_type == "table":
        # テーブルの処理（簡易版）
        return f"{indent}[Table - Please view in Notion]\n\n", child_page_ids
    elif block_type == "column_list":
        return "", child_page_ids  # カラムリストは子要素で処理
    elif block_type == "column":
        return "", child_page_ids  # カラムは子要素で処理
    elif block_type == "synced_block":
        # 同期ブロックの処理
        synced_from = content.get("synced_from")
        if synced_from:
            return f"{indent}[Synced Block]\n\n", child_page_ids
        return "", child_page_ids

    # メンションやリンクの検出
    # Notionのテキスト内にページやデータベースへのメンションが含まれる場合、
    # それらのIDを収集して後で再帰的に処理する
    if "rich_text" in content:
        for rt in content.get("rich_text", []):
            if rt.get("type") == "mention":
                mention = rt.get("mention", {})
                if mention.get("type") == "page":
                    page_id = mention.get("page", {}).get("id")
                    if page_id:
                        child_page_ids.append(page_id)
                elif mention.get("type") == "database":
                    db_id = mention.get("database", {}).get("id")
                    if db_id:
                        child_page_ids.append(db_id)

    return "", child_page_ids


def process_blocks_recursively(
    blocks: List[Dict], notion_client, indent_level: int = 0
) -> Tuple[str, List[str]]:
    """
    ブロックを再帰的に処理してMarkdownとリンクされたページIDを取得
    
    【処理内容】
    Notionのブロックは階層構造を持ち、ブロックの中に子ブロックを含むことができる。
    この関数は、ブロックのリストを受け取り、各ブロックをMarkdownに変換し、
    has_childrenフラグが立っているブロックは再帰的に子ブロックを処理する。
    
    【特別な処理】
    - toggleブロック: 子要素の処理後に</details>タグを追加
    - インデント: ネストの深さに応じてインデントを追加
    
    Args:
        blocks: 処理するブロックのリスト
        notion_client: Notion APIクライアントオブジェクト
        indent_level: 現在のインデントレベル
    
    Returns:
        Tuple[str, List[str]]: 
            - すべてのブロックを変換したMarkdown文字列
            - 発見した子ページIDのリスト
    """
    markdown_content = ""
    all_child_page_ids = []

    for block in blocks:
        # 各ブロックをMarkdownに変換
        block_md, child_ids = convert_block_to_markdown(block, indent_level)
        markdown_content += block_md
        all_child_page_ids.extend(child_ids)

        # 子ブロックがある場合は再帰的に処理
        if block.get("has_children", False):
            try:
                # 子ブロックをAPIから取得
                child_blocks = notion_client.blocks.children.list(
                    block_id=block["id"]
                ).get("results", [])
                
                # 子ブロックを再帰的に処理（インデントレベルを上げる）
                child_md, more_child_ids = process_blocks_recursively(
                    child_blocks, notion_client, indent_level + 1
                )
                markdown_content += child_md
                all_child_page_ids.extend(more_child_ids)

                # toggleブロックの場合は終了タグを追加
                if block.get("type") == "toggle":
                    markdown_content += "  " * indent_level + "</details>\n\n"
            except Exception as e:
                print(f"    ⚠️  子ブロック取得エラー: {e}")

    return markdown_content, all_child_page_ids


def get_page_properties(page_info: Dict) -> Dict:
    """
    Notionページのプロパティを取得して辞書形式で返す
    
    【処理内容】
    Notionのページには様々なプロパティ（タイトル、日付、タグ、チェックボックスなど）
    を設定できる。この関数は、APIから取得したプロパティ情報を
    Pythonの辞書形式に整形して返す。
    
    【対応プロパティタイプ】
    - title: タイトル
    - rich_text: リッチテキスト
    - number: 数値
    - select: 単一選択
    - multi_select: 複数選択
    - date: 日付
    - people: ユーザー
    - files: ファイル
    - checkbox: チェックボックス
    - url: URL
    - email: メールアドレス
    - phone_number: 電話番号
    - formula: 数式
    - relation: リレーション
    - rollup: ロールアップ
    - created_time: 作成日時
    - created_by: 作成者
    - last_edited_time: 最終編集日時
    - last_edited_by: 最終編集者
    
    Args:
        page_info: Notion APIから取得したページ情報
    
    Returns:
        Dict: プロパティ名をキー、値をバリューとする辞書
    """
    properties = {}
    for prop_name, prop_value in page_info.get("properties", {}).items():
        prop_type = prop_value.get("type")  # プロパティのタイプを取得

        if prop_type == "title":
            properties[prop_name] = "".join(
                [t.get("plain_text", "") for t in prop_value.get("title", [])]
            )
        elif prop_type == "rich_text":
            properties[prop_name] = "".join(
                [t.get("plain_text", "") for t in prop_value.get("rich_text", [])]
            )
        elif prop_type == "number":
            properties[prop_name] = prop_value.get("number")
        elif prop_type == "select":
            select = prop_value.get("select")
            properties[prop_name] = select.get("name") if select else None
        elif prop_type == "multi_select":
            properties[prop_name] = [
                s.get("name") for s in prop_value.get("multi_select", [])
            ]
        elif prop_type == "date":
            date = prop_value.get("date")
            if date:
                properties[prop_name] = {
                    "start": date.get("start"),
                    "end": date.get("end"),
                }
        elif prop_type == "people":
            properties[prop_name] = [
                p.get("name", p.get("id")) for p in prop_value.get("people", [])
            ]
        elif prop_type == "files":
            files = prop_value.get("files", [])
            properties[prop_name] = [
                f.get(
                    "name",
                    f.get("external", {}).get("url", f.get("file", {}).get("url")),
                )
                for f in files
            ]
        elif prop_type == "checkbox":
            properties[prop_name] = prop_value.get("checkbox", False)
        elif prop_type == "url":
            properties[prop_name] = prop_value.get("url")
        elif prop_type == "email":
            properties[prop_name] = prop_value.get("email")
        elif prop_type == "phone_number":
            properties[prop_name] = prop_value.get("phone_number")
        elif prop_type == "formula":
            formula = prop_value.get("formula", {})
            properties[prop_name] = formula.get(formula.get("type"))
        elif prop_type == "relation":
            properties[prop_name] = [
                r.get("id") for r in prop_value.get("relation", [])
            ]
        elif prop_type == "rollup":
            rollup = prop_value.get("rollup", {})
            properties[prop_name] = rollup.get(rollup.get("type"))
        elif prop_type == "created_time":
            properties[prop_name] = prop_value.get("created_time")
        elif prop_type == "created_by":
            created_by = prop_value.get("created_by", {})
            properties[prop_name] = created_by.get("name", created_by.get("id"))
        elif prop_type == "last_edited_time":
            properties[prop_name] = prop_value.get("last_edited_time")
        elif prop_type == "last_edited_by":
            edited_by = prop_value.get("last_edited_by", {})
            properties[prop_name] = edited_by.get("name", edited_by.get("id"))

    return properties


def traverse_and_save(page_id: str, parent_titles: List[str], depth: int = 0):
    """
    ページを再帰的に辿り、Markdownとして保存する中心的な関数
    
    【処理フロー】
    1. 訪問済みチェック: 循環参照を防ぐため、処理済みページはスキップ
    2. レート制限対策: APIのレート制限を避けるためsleepを挿入
    3. ページ/データベース判定: APIエラーによりタイプを判別
    4. コンテンツ取得: ブロックを取得してMarkdownに変換
    5. ファイル保存: タイトル階層をファイル名にして保存
    6. メタデータ更新: 進捗状況を即座に保存（中断時の再開用）
    7. 再帰処理: 発見した子ページを再帰的に処理

    Args:
        page_id (str): 現在処理中のページID
        parent_titles (List[str]): 親ページのタイトル階層（ファイル名生成用）
        depth (int): 現在の深さ（コンソール出力のインデント用）
    """
    # ========================================
    # 1. 訪問済みチェック
    # ========================================
    if page_id in visited_page_ids:
        print(
            f"{'  ' * depth}⏭️  既訪問: {page_id[:8]}... [処理済: {len(visited_page_ids)}]"
        )
        return
    visited_page_ids.add(page_id)  # 訪問済みとして記録

    # ========================================
    # 2. レート制限対策
    # ========================================
    time.sleep(0.3)  # Notion APIのレート制限を避けるための待機

    # ========================================
    # 3. 進捗表示
    # ========================================
    progress_info = f"[{len(visited_page_ids)}ページ処理中]"
    print(
        f"{'  ' * depth}📄 処理中: {' > '.join(parent_titles[-3:])} - {page_id[:8]}... {progress_info}"
    )

    try:
        # 変数の初期化
        is_database = False  # データベースかどうかのフラグ
        current_title = "Untitled"  # ページのタイトル（デフォルト値）
        page_info = None  # ページ情報を格納する変数

        # トリッキーな処理：NotionではIDがページかデータベースか事前に判断できない
        # そのため、まずページAPIで取得を試み、失敗したらデータベースAPIを試す
        try:
            # pages.retrieve APIを使ってページ情報を取得
            page_info = notion.pages.retrieve(page_id=page_id)
            
            # トリッキーな処理：Notionのページのタイトル取得
            # propertiesの中で"title"という名前のプロパティを探す（デフォルト）
            title_property = page_info.get("properties", {}).get("title", {})
            
            # "title"という名前のプロパティがない場合のフォールバック
            if not title_property:
                # 全プロパティをループしてtype="title"のプロパティを探す
                for prop_name, prop_value in page_info.get("properties", {}).items():
                    if prop_value.get("type") == "title":
                        title_property = prop_value
                        break  # 最初に見つかったタイトルプロパティを使用

            # リスト内包表記を使ってタイトルを結合
            # titleプロパティはリッチテキストの配列なので、plain_textを抽出して結合
            current_title = (
                "".join(
                    [t.get("plain_text", "") for t in title_property.get("title", [])]
                )
                or "Untitled"  # タイトルが空の場合は"Untitled"を使用
            )
        except Exception as e:
            # ページAPIが失敗 = データベースの可能性が高い
            is_database = True
            print(f"{'  ' * depth}  🗄️  データベースとして処理: {page_id[:8]}...")

        if is_database:
            # データベースとして処理する場合
            try:
                # databases.retrieve APIでデータベース情報を取得
                db_info = notion.databases.retrieve(database_id=page_id)
                
                # データベースのタイトルを取得（titleフィールドはリッチテキスト配列）
                current_title = (
                    "".join([t.get("plain_text", "") for t in db_info.get("title", [])])
                    or "Database"  # タイトルがない場合のデフォルト値
                )

                # databases.query APIでデータベース内の全ページを取得
                # ページネーション処理を実装して全ページを取得
                db_pages = []  # 全ページを格納するリスト
                has_more = True  # 次のページがあるかどうか
                start_cursor = None  # ページネーション用カーソル
                
                while has_more:
                    # データベースをクエリ（ページネーション対応）
                    if start_cursor:
                        response = notion.databases.query(
                            database_id=page_id,
                            start_cursor=start_cursor
                        )
                    else:
                        response = notion.databases.query(database_id=page_id)
                    
                    # 取得したページを追加
                    db_pages.extend(response.get("results", []))
                    
                    # 次ページの有無を確認
                    has_more = response.get("has_more", False)
                    start_cursor = response.get("next_cursor")
                print(
                    f"{'  ' * depth}  📊 データベース「{current_title}」内の{len(db_pages)}ページを処理"
                )

                # データベース自体の情報を保存
                current_path_titles = parent_titles + [current_title]
                filename = sanitize_filename("-".join(current_path_titles)) + "_DB.md"
                filepath = os.path.join(SAVE_DIR, filename)

                db_content = f"# 🗄️ {current_title}\n\n"
                db_content += f"**Type**: Database\n"
                db_content += f"**ID**: {page_id}\n"
                db_content += f"**Total Pages**: {len(db_pages)}\n\n"
                db_content += "## Properties\n\n"

                # プロパティ情報を追加
                for prop_name, prop_config in db_info.get("properties", {}).items():
                    db_content += f"- **{prop_name}** ({prop_config.get('type')})\n"

                db_content += f"\n## Pages in Database\n\n"

                # データベース内の各ページへのリンクを作成
                for db_page in db_pages:
                    page_title = "Untitled"  # デフォルトのページタイトル
                    
                    # データベース内のページのタイトルを探す
                    # データベースの各ページはpropertiesを持ち、その中にtitleタイプがある
                    for prop_value in db_page.get("properties", {}).values():
                        if prop_value.get("type") == "title":
                            page_title = (
                                "".join(
                                    [
                                        t.get("plain_text", "")
                                        for t in prop_value.get("title", [])
                                    ]
                                )
                                or "Untitled"
                            )
                            break
                    db_content += f"- [{page_title}](notion://{db_page['id']})\n"

                # ファイルに即座に保存（ディレクトリ作成も含む）
                os.makedirs(os.path.dirname(filepath) or SAVE_DIR, exist_ok=True)
                
                # データベース情報もクレンジング（RAG検索精度向上）
                original_length = len(db_content)
                cleaned_content = clean_notion_text(db_content)
                cleaned_length = len(cleaned_content)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)
                print(f"{'  ' * depth}  ✅ データベース保存: {filename}")
                print(f"{'  ' * depth}     🧹 クレンジング: {original_length} → {cleaned_length} 文字 (削減率: {100*(1-cleaned_length/original_length):.1f}%)")

                # メタデータに追加して即座に保存
                metadata["pages"][page_id] = {
                    "title": current_title,
                    "type": "database",
                    "path": filepath,
                    "parent_titles": parent_titles,
                    "page_count": len(db_pages),
                }
                save_metadata()  # メタデータを即座に保存

                # データベース内の各ページを再帰的に処理
                for db_page in db_pages:
                    traverse_and_save(db_page["id"], current_path_titles, depth + 1)

            except Exception as e:
                print(f"{'  ' * depth}  ❌ データベース処理エラー: {e}")
                metadata["error_pages"].append(
                    {"id": page_id, "error": str(e), "type": "database"}
                )
                save_metadata()  # エラー情報も即座に保存
            return

        # ページの場合の処理（データベースではない通常のページ）
        
        # ファイル名生成: 親ページのタイトルを階層的に結合
        # 例: ["Root", "Parent", "Current"] → "Root-Parent-Current.md"
        current_path_titles = parent_titles + [current_title]
        filename = sanitize_filename("-".join(current_path_titles)) + ".md"
        filepath = os.path.join(SAVE_DIR, filename)

        # ページのプロパティ（タグ、日付、作成者など）を取得
        properties = get_page_properties(page_info)

        # Markdownファイルのコンテンツを構築開始
        # まずタイトルをH1として追加
        page_content_md = f"# {current_title}\n\n"

        # プロパティ情報をMarkdownのメタデータセクションとして追加
        if properties:
            page_content_md += "## Properties\n\n"
            
            # 各プロパティをリスト形式で出力
            for prop_name, prop_value in properties.items():
                if prop_value is not None:  # None値はスキップ
                    # リストや辞書はJSON形式に変換（日本語を保持）
                    if isinstance(prop_value, (list, dict)):
                        prop_value = json.dumps(prop_value, ensure_ascii=False)
                    page_content_md += f"- **{prop_name}**: {prop_value}\n"
            
            # プロパティセクションの後に区切り線を追加
            page_content_md += "\n---\n\n"

        # ページの全ブロック（コンテンツ）を取得
        # Notion APIはページネーションを使用してブロックを返す
        all_blocks = []  # 全ブロックを格納するリスト
        has_more = True  # 次のページがあるかどうかのフラグ
        start_cursor = None  # ページネーション用のカーソル

        # トリッキーなページネーション処理
        # Notion APIは1回のリクエストで最大100ブロックしか返さない
        while has_more:
            if start_cursor:
                # 2回目以降のリクエスト: カーソルを指定
                response = notion.blocks.children.list(
                    block_id=page_id, start_cursor=start_cursor
                )
            else:
                # 初回のリクエスト: カーソルなし
                response = notion.blocks.children.list(block_id=page_id)

            # 取得したブロックをリストに追加
            all_blocks.extend(response.get("results", []))
            
            # 次ページの有無を確認
            has_more = response.get("has_more", False)
            
            # 次ページ用のカーソルを取得
            start_cursor = response.get("next_cursor")

        # ブロックを再帰的に処理してMarkdownに変換
        # 返り値: (Markdown文字列, 子ページIDのリスト)
        blocks_md, child_page_ids = process_blocks_recursively(all_blocks, notion)
        page_content_md += blocks_md  # 変換したMarkdownを追加

        # Markdownファイルとして即座に保存
        # トリッキー: os.path.dirname()が空文字列を返す場合に備えてor演算子を使用
        os.makedirs(os.path.dirname(filepath) or SAVE_DIR, exist_ok=True)
        
        # RAG検索精度向上のため、テキストをクレンジング
        # HTMLタグやNotion固有の記法を除去
        original_length = len(page_content_md)
        cleaned_content = clean_notion_text(page_content_md)
        cleaned_length = len(cleaned_content)
        
        # ファイルに書き込み（UTF-8エンコーディングで日本語を保持）
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
        print(f"{'  ' * depth}  ✅ 保存完了: {filename}")
        print(f"{'  ' * depth}     🧹 クレンジング: {original_length} → {cleaned_length} 文字 (削減率: {100*(1-cleaned_length/original_length):.1f}%)")

        # メタデータに追加して即座に保存
        metadata["pages"][page_id] = {
            "title": current_title,
            "type": "page",
            "path": filepath,
            "parent_titles": parent_titles,
            "properties": properties,
            "child_pages": child_page_ids,
        }
        metadata["total_pages"] += 1
        save_metadata()  # メタデータを即座に保存

        # 発見した子ページを再帰的に辿る
        for child_id in child_page_ids:
            traverse_and_save(child_id, current_path_titles, depth + 1)

    except Exception as e:
        print(f"{'  ' * depth}  ❌ エラー発生 (ID: {page_id[:8]}...): {e}")
        metadata["error_pages"].append(
            {"id": page_id, "error": str(e), "parent_titles": parent_titles}
        )
        save_metadata()  # エラー情報も即座に保存


def save_metadata():
    """
    メタデータをJSONファイルとして保存
    
    【保存内容】
    - crawl_date: クロール開始日時
    - pages: 処理済みページの詳細情報（ID、タイトル、ファイルパス等）
    - total_pages: 処理済みページ数
    - error_pages: エラーが発生したページの情報
    - crawl_completed: クロール完了日時
    - total_time_seconds: 処理時間
    
    【重要】
    各ページ処理後に即座に保存することで、
    プログラムが中断されても途中から再開できるようにしている。
    """
    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ メタデータ保存エラー: {e}")


def main():
    """
    メイン実行関数
    
    【処理フロー】
    1. 環境変数チェック: NOTION_TOKENが設定されているか確認
    2. 保存先ディレクトリ作成: Markdownファイルの保存先を準備
    3. 既存メタデータ確認: 前回の処理があるかチェック
    4. クロール開始: ルートページから再帰的に処理
    5. 統計情報出力: 処理結果を表示
    
    【ルートページID】
    ハードコードされたルートページIDからクロールを開始する。
    必要に応じて、コマンドライン引数や環境変数から取得するように
    変更することも可能。
    """
    print("スクリプト開始")
    
    # Notionクライアントを初期化
    global NOTION_TOKEN, notion
    NOTION_TOKEN = os.getenv("NOTION_TOKEN")  # 一時的に直接取得
    
    # 指定されたNotionページID（クロールの起点）
    root_page_id = os.getenv("NOTION_ROOT_PAGE_ID", "03b2ee3346944d159de49f40c952bc21")

    # ========================================
    # 1. 環境変数チェック
    # ========================================
    if not NOTION_TOKEN:
        print("エラー: .envファイルにNOTION_TOKENを設定してください。")
        return
    
    # Notionクライアントを作成
    notion = notion_client.Client(auth=NOTION_TOKEN)
    print(f"Notionクライアント作成完了")

    print("=" * 60)
    print("🚀 Notionクローラー開始")
    print("=" * 60)

    # ========================================
    # 2. 保存先ディレクトリを作成
    # ========================================
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📁 保存先: {SAVE_DIR}")
    print(f"🏠 ルートページID: {root_page_id}")

    # ========================================
    # 3. 既存のメタデータを読み込み（再開可能にする）
    # ========================================
    is_resume = load_existing_metadata()
    if is_resume:
        print("📝 前回の処理を継続します")
    else:
        print("🆕 新規処理を開始します")

    print("-" * 60)

    start_time = time.time()

    # 初期メタデータを保存
    metadata["crawl_date"] = datetime.now().isoformat()
    save_metadata()

    # ========================================
    # 4. ルートページから処理を開始
    # ========================================
    traverse_and_save(root_page_id, [])

    elapsed_time = time.time() - start_time

    # ========================================
    # 5. 最終統計を更新して保存
    # ========================================
    metadata["crawl_completed"] = datetime.now().isoformat()
    metadata["total_time_seconds"] = elapsed_time
    save_metadata()

    print("-" * 60)
    print("📈 統計情報")
    print(f"  - 処理ページ数: {metadata['total_pages']}")
    print(f"  - エラー数: {len(metadata['error_pages'])}")
    print(f"  - 処理時間: {elapsed_time:.2f}秒")
    print("-" * 60)

    if metadata["error_pages"]:
        print("⚠️  エラーが発生したページ:")
        for error in metadata["error_pages"]:
            print(f"  - {error['id']}: {error['error']}")
        print("-" * 60)

    print("✅ すべての処理が完了しました。")
    print(f"📊 メタデータ: {METADATA_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
