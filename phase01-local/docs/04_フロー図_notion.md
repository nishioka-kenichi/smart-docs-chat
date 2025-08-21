# Notionクローラー 処理フロー図

## 1. メインフローチャート

```mermaid
flowchart TD
    Start([開始]) --> LoadEnv[環境変数読み込み<br/>.envファイル]
    LoadEnv --> InitClient[Notion Client初期化<br/>APIトークン設定]
    
    InitClient --> CreateDirs[保存先ディレクトリ作成<br/>./data/documents/notion/]
    CreateDirs --> LoadMeta{既存メタデータ<br/>存在？}
    
    LoadMeta -->|Yes| RestoreMeta[メタデータ復元<br/>visited_page_ids再構築]
    LoadMeta -->|No| InitMeta[新規メタデータ初期化<br/>空のセット作成]
    
    RestoreMeta --> GetRootPage[ルートページ取得<br/>NOTION_ROOT_PAGE_ID]
    InitMeta --> GetRootPage
    
    GetRootPage --> TraverseStart[traverse_and_save開始<br/>再帰処理開始]
    
    TraverseStart --> ProcessPage{ページ処理}
    
    ProcessPage -->|未訪問| FetchPage[ページ/DB取得<br/>Notion API呼び出し]
    ProcessPage -->|訪問済| SkipPage[スキップ<br/>処理済みログ出力]
    
    FetchPage --> ConvertMD[Markdown変換<br/>ブロック処理]
    ConvertMD --> CleanText[テキストクレンジング<br/>ノイズ除去]
    CleanText --> SaveFile[ファイル保存<br/>.md形式]
    SaveFile --> UpdateMeta[メタデータ更新<br/>訪問済み記録]
    
    UpdateMeta --> FindLinks[リンク探索<br/>子ページ・DB検出]
    FindLinks --> RecurseCheck{子要素あり？}
    
    RecurseCheck -->|Yes| ProcessChild[子要素処理<br/>再帰呼び出し]
    ProcessChild --> ProcessPage
    
    RecurseCheck -->|No| NextCheck{次のページ？}
    SkipPage --> NextCheck
    
    NextCheck -->|Yes| ProcessPage
    NextCheck -->|No| CalcStats[統計情報計算<br/>処理時間・ページ数]
    
    CalcStats --> SaveFinalMeta[最終メタデータ保存<br/>metadata.json]
    SaveFinalMeta --> ShowSummary[サマリー表示<br/>処理結果]
    ShowSummary --> End([終了])
```

## 2. ページ処理の詳細フロー

```mermaid
flowchart TD
    PageStart([ページ処理開始]) --> CheckType{オブジェクト<br/>タイプ判定}
    
    CheckType -->|Page| GetPageData[ページデータ取得<br/>プロパティ・タイトル]
    CheckType -->|Database| GetDBData[データベース取得<br/>タイトル・説明]
    
    GetPageData --> GetBlocks[ブロック取得<br/>blocks.children.list]
    GetDBData --> GetDBPages[DBページ取得<br/>databases.query]
    
    GetBlocks --> ProcessBlocks[ブロック処理<br/>再帰的変換]
    GetDBPages --> ProcessDBRows[行データ処理<br/>各ページを処理]
    
    ProcessBlocks --> ExtractLinks[リンク抽出<br/>メンション・子ページ]
    ProcessDBRows --> ExtractLinks
    
    ExtractLinks --> GenerateMD[Markdown生成<br/>階層構造保持]
    GenerateMD --> PageEnd([ページ処理終了])
```

## 3. ブロック変換の詳細

```mermaid
flowchart TD
    BlockStart([ブロック処理]) --> BlockType{ブロック<br/>タイプ}
    
    BlockType -->|paragraph| TextBlock[テキスト抽出<br/>リッチテキスト処理]
    BlockType -->|heading| HeadingBlock[見出し変換<br/># ## ###]
    BlockType -->|bulleted_list| BulletList[箇条書き<br/>- リスト]
    BlockType -->|numbered_list| NumberList[番号付きリスト<br/>1. 2. 3.]
    BlockType -->|toggle| ToggleBlock[トグル<br/>details/summary]
    BlockType -->|code| CodeBlock[コードブロック<br/>```言語```]
    BlockType -->|quote| QuoteBlock[引用<br/>> テキスト]
    BlockType -->|callout| CalloutBlock[コールアウト<br/>💡 Note:]
    BlockType -->|divider| Divider[区切り線<br/>---]
    BlockType -->|image| ImageBlock[画像<br/>![alt](url)]
    BlockType -->|video| VideoBlock[動画<br/>📹 Video]
    BlockType -->|file| FileBlock[ファイル<br/>📎 File]
    BlockType -->|pdf| PDFBlock[PDF<br/>📄 PDF]
    BlockType -->|bookmark| BookmarkBlock[ブックマーク<br/>🔖 リンク]
    BlockType -->|child_page| ChildPage[子ページ<br/>📄 Page Link]
    BlockType -->|child_database| ChildDB[子DB<br/>🗄️ Database]
    BlockType -->|table| TableBlock[テーブル<br/>[Table]]
    BlockType -->|embed| EmbedBlock[埋め込み<br/>🌐 Embed]
    
    TextBlock --> CheckMention{メンション<br/>含む？}
    CheckMention -->|Yes| ExtractPageID[ページID抽出<br/>後で処理]
    CheckMention -->|No| ReturnMD
    
    HeadingBlock --> ReturnMD[Markdown返却]
    BulletList --> ReturnMD
    NumberList --> ReturnMD
    ToggleBlock --> CheckChildren{子ブロック<br/>あり？}
    CodeBlock --> ReturnMD
    QuoteBlock --> ReturnMD
    CalloutBlock --> ReturnMD
    Divider --> ReturnMD
    ImageBlock --> ReturnMD
    VideoBlock --> ReturnMD
    FileBlock --> ReturnMD
    PDFBlock --> ReturnMD
    BookmarkBlock --> ReturnMD
    ChildPage --> ExtractPageID
    ChildDB --> ExtractPageID
    TableBlock --> ReturnMD
    EmbedBlock --> ReturnMD
    
    CheckChildren -->|Yes| RecurseBlocks[再帰処理<br/>子ブロック変換]
    CheckChildren -->|No| ReturnMD
    
    RecurseBlocks --> ReturnMD
    ExtractPageID --> ReturnMD
    
    ReturnMD --> BlockEnd([ブロック処理終了])
```

## 4. テキストクレンジング処理

```mermaid
flowchart LR
    RawText([生テキスト]) --> RemoveHTML[HTMLタグ除去<br/>u, details, summary]
    RemoveHTML --> SimplifyMedia[メディア簡略化<br/>画像・動画・PDF]
    SimplifyMedia --> RemoveNotion[Notion記法除去<br/>notion://リンク]
    RemoveNotion --> NormalizeSpace[空白正規化<br/>連続改行・空白]
    NormalizeSpace --> UnicodeNorm[Unicode正規化<br/>NFKC形式]
    UnicodeNorm --> CleanedText([クレンジング済み<br/>テキスト])
```

## 5. エラー処理とレート制限

```mermaid
flowchart TD
    APICall([API呼び出し]) --> TryCatch{try-catch}
    
    TryCatch -->|Success| ProcessData[データ処理]
    TryCatch -->|Error| CheckError{エラー<br/>タイプ}
    
    CheckError -->|RateLimit| Wait[待機<br/>0.3秒]
    CheckError -->|NotFound| LogSkip[スキップ記録<br/>エラーカウント]
    CheckError -->|Auth| StopProcess[処理停止<br/>認証エラー]
    CheckError -->|Other| LogError[エラー記録<br/>続行]
    
    Wait --> Retry{リトライ<br/>回数？}
    Retry -->|< 3| APICall
    Retry -->|>= 3| LogSkip
    
    ProcessData --> NextItem([次の処理へ])
    LogSkip --> NextItem
    LogError --> NextItem
```

## 6. データ保存形式

```
data/documents/notion/
├── ページタイトル1.md
├── ページタイトル2.md
├── データベース名_DB.md
└── metadata.json
```

### metadata.json構造
```json
{
  "total_pages": 182,
  "total_errors": 0,
  "crawl_start": "2025-01-20T00:00:00",
  "crawl_end": "2025-01-20T00:30:00",
  "visited_page_ids": ["page_id1", "page_id2", ...],
  "root_page_id": "xxxxxxxxxxxx"
}
```

## 7. 主な特徴

### 再帰的探索
- ルートページから開始
- リンク、メンション、子ページを自動検出
- 訪問済みページはスキップ（重複防止）

### クレンジング機能
- HTMLタグ除去
- Notion固有記法の除去
- メディアファイルの簡略化
- 検索精度向上のための最適化

### エラー処理
- API レート制限対策（0.3秒待機）
- 404エラーのスキップ
- 処理継続性の確保

### 中断・再開対応
- メタデータによる進捗管理
- 訪問済みページの記録
- 差分更新が可能

---

最終更新: 2025年1月20日