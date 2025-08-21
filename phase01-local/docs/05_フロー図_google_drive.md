# Google Driveクローラー 処理フロー図

## 1. メインフローチャート

```mermaid
flowchart TD
    Start([開始]) --> LoadEnv[環境変数読み込み<br/>.envファイル]
    LoadEnv --> CheckEnv{環境変数<br/>チェック}
    
    CheckEnv -->|NG| ErrorEnd1[エラー終了<br/>環境変数未設定]
    CheckEnv -->|OK| InitAPI[Google Drive API<br/>初期化]
    
    InitAPI --> CreateDir[保存先ディレクトリ<br/>作成]
    CreateDir --> LoadMeta{既存メタデータ<br/>存在？}
    
    LoadMeta -->|Yes| RestoreMeta[メタデータ復元<br/>visited_item_ids再構築]
    LoadMeta -->|No| InitMeta[新規メタデータ<br/>初期化]
    
    RestoreMeta --> StartCrawl
    InitMeta --> StartCrawl[クロール開始<br/>各ルートフォルダ処理]
    
    StartCrawl --> ForEachRoot{各ルート<br/>フォルダ}
    
    ForEachRoot -->|処理| Traverse[traverse_and_save<br/>再帰処理]
    Traverse --> ForEachRoot
    
    ForEachRoot -->|完了| CalcStats[統計情報計算<br/>処理時間・ファイル数]
    CalcStats --> SaveFinalMeta[最終メタデータ<br/>保存]
    SaveFinalMeta --> ShowSummary[サマリー表示]
    ShowSummary --> End([終了])

    style Start fill:#e1f5e1
    style End fill:#e1f5e1
    style ErrorEnd1 fill:#ffe1e1
```

## 2. traverse_and_save関数の詳細フロー

```mermaid
flowchart TD
    TraverseStart([traverse_and_save開始<br/>item_id, parent_path, depth]) --> CheckVisited{訪問済み？<br/>visited_item_ids}
    
    CheckVisited -->|Yes| SkipItem[スキップ<br/>ログ出力]
    SkipItem --> TraverseEnd([終了])
    
    CheckVisited -->|No| RateLimit[レート制限<br/>0.1秒待機]
    RateLimit --> GetMeta[メタデータ取得<br/>files().get() API]
    
    GetMeta -->|成功| CheckType{アイテム<br/>タイプ判定}
    GetMeta -->|失敗| LogError[エラーログ<br/>metadata更新]
    
    CheckType -->|フォルダ| ProcessFolder[フォルダ処理]
    CheckType -->|ファイル| ProcessFile[ファイル処理]
    CheckType -->|未対応| SkipUnsupported[未対応形式<br/>スキップ]
    
    ProcessFolder --> AddVisited1[visited_item_ids<br/>に追加]
    AddVisited1 --> GetChildren[子アイテム<br/>リスト取得]
    GetChildren --> RecurseChildren{各子アイテム}
    RecurseChildren -->|再帰| TraverseStart
    RecurseChildren -->|完了| TraverseEnd
    
    ProcessFile --> CheckMime{MIME_TYPE<br/>対応？}
    CheckMime -->|No| SkipUnsupported
    CheckMime -->|Yes| AddVisited2[visited_item_ids<br/>に追加]
    
    AddVisited2 --> DownloadContent[コンテンツ<br/>ダウンロード]
    DownloadContent -->|成功| SaveFile[ファイル保存<br/>ローカル]
    DownloadContent -->|失敗| LogError
    
    SaveFile --> UpdateMeta[メタデータ更新<br/>files辞書に追加]
    UpdateMeta --> SaveMeta[metadata.json<br/>保存]
    SaveMeta --> TraverseEnd
    
    LogError --> SaveErrorMeta[エラー情報<br/>保存]
    SaveErrorMeta --> TraverseEnd
    
    SkipUnsupported --> TraverseEnd

    style TraverseStart fill:#e1f5e1
    style TraverseEnd fill:#e1f5e1
    style LogError fill:#ffe1e1
```

## 3. ファイルダウンロード処理の詳細

```mermaid
flowchart TD
    DownStart([download_file_content開始<br/>file_id, mime_type]) --> GetMimeInfo[MIME_TYPE_MAPPING<br/>から設定取得]
    
    GetMimeInfo --> CheckExport{エクスポート<br/>必要？}
    
    CheckExport -->|Yes<br/>Googleドキュメント等| ExportAPI[export_media()<br/>API使用]
    CheckExport -->|No<br/>通常ファイル| GetMediaAPI[get_media()<br/>API使用]
    
    ExportAPI --> CreateIO
    GetMediaAPI --> CreateIO[BytesIOバッファ<br/>作成]
    
    CreateIO --> DownloadChunk[チャンク単位で<br/>ダウンロード]
    
    DownloadChunk --> CheckDone{完了？}
    CheckDone -->|No| NextChunk[次のチャンク<br/>取得]
    NextChunk --> DownloadChunk
    
    CheckDone -->|Yes| CheckPDF{PDFファイル？}
    
    CheckPDF -->|Yes| PDFExtract[pdfplumber<br/>テキスト抽出]
    PDFExtract --> PDFClean[PDFテキスト<br/>クレンジング]
    PDFClean --> ReturnContent[テキスト<br/>返却]
    
    CheckPDF -->|No| DecodeUTF8[UTF-8<br/>デコード試行]
    
    DecodeUTF8 -->|成功| ReturnContent
    DecodeUTF8 -->|失敗| DecodeLatin1[Latin-1<br/>強制デコード]
    
    DecodeLatin1 --> ReturnContent
    ReturnContent --> DownEnd([終了])
    
    ExportAPI -->|403エラー| AccessDenied[アクセス権限なし<br/>ログ出力]
    GetMediaAPI -->|403エラー| AccessDenied
    PDFExtract -->|エラー| PDFError[PDF処理エラー<br/>ログ出力]
    
    AccessDenied --> ReturnNull[None返却]
    PDFError --> ReturnNull
    ReturnNull --> DownEnd

    style DownStart fill:#e1f5e1
    style DownEnd fill:#e1f5e1
    style AccessDenied fill:#ffe1e1
    style PDFError fill:#ffe1e1
```

## 4. PDFテキストクレンジング処理

```mermaid
flowchart LR
    PDFText([PDFテキスト]) --> RemoveMarkers[ページマーカー除去<br/>--- ページ X/Y ---]
    RemoveMarkers --> RemoveControl[制御文字除去<br/>0x00-0x1F]
    RemoveControl --> NormalizeBreaks[改行正規化<br/>3つ以上→2つ]
    NormalizeBreaks --> NormalizeSpaces[空白正規化<br/>連続空白→1つ]
    NormalizeSpaces --> FixGarbled[文字化け修正<br/>□,●等]
    FixGarbled --> UnicodeNorm[Unicode正規化<br/>NFKC形式]
    UnicodeNorm --> CleanedText([クレンジング済み<br/>テキスト])
```

## 5. シーケンス図

```mermaid
sequenceDiagram
    participant User
    participant Main as main()
    participant Env as 環境変数
    participant API as Google Drive API
    participant Traverse as traverse_and_save()
    participant Download as download_file_content()
    participant PDF as pdfplumber
    participant FS as ファイルシステム
    participant Meta as metadata.json

    User->>Main: スクリプト実行
    Main->>Env: .env読み込み
    Env-->>Main: 認証情報パス、フォルダID
    
    Main->>API: サービス初期化
    API-->>Main: 接続成功
    
    Main->>FS: 保存先ディレクトリ作成
    Main->>Meta: 既存メタデータ確認
    
    alt メタデータ存在
        Meta-->>Main: 既存データ返却
        Main->>Main: visited_item_ids復元
    else 新規処理
        Main->>Main: 空のメタデータ初期化
    end
    
    loop 各ルートフォルダ
        Main->>Traverse: 処理開始(folder_id)
        
        Traverse->>Traverse: 訪問済みチェック
        
        alt 未訪問
            Traverse->>API: アイテム情報取得
            API-->>Traverse: メタデータ返却
            
            alt フォルダの場合
                Traverse->>API: 子アイテムリスト取得
                API-->>Traverse: 子アイテムID配列
                
                loop 各子アイテム
                    Traverse->>Traverse: 再帰呼び出し
                end
                
            else ファイルの場合
                Traverse->>Download: コンテンツ取得
                
                alt Googleドキュメント
                    Download->>API: export_media()
                else PDFファイル
                    Download->>API: get_media()
                    API-->>Download: PDFバイナリ
                    Download->>PDF: テキスト抽出
                    PDF-->>Download: 生テキスト
                    Download->>Download: クレンジング処理
                else 通常ファイル
                    Download->>API: get_media()
                end
                
                API-->>Download: データ
                Download-->>Traverse: テキストデータ
                
                Traverse->>FS: ファイル保存
                Traverse->>Meta: メタデータ更新
            end
        end
    end
    
    Main->>Meta: 最終メタデータ保存
    Main->>User: 処理完了・統計表示
```

## 6. エラー処理フロー

```mermaid
flowchart LR
    subgraph エラー種別
        E1[認証エラー]
        E2[403 Forbidden]
        E3[API制限]
        E4[デコードエラー]
        E5[ファイル書き込みエラー]
        E6[PDF処理エラー]
    end
    
    subgraph 対処
        H1[プログラム終了]
        H2[スキップ&ログ記録]
        H3[待機時間挿入]
        H4[別エンコーディング試行]
        H5[エラーメタデータ保存]
    end
    
    E1 --> H1
    E2 --> H2
    E2 --> H5
    E3 --> H3
    E4 --> H4
    E5 --> H2
    E5 --> H5
    E6 --> H2
    E6 --> H5
```

## 7. データ構造

```mermaid
classDiagram
    class Metadata {
        +crawl_date: str
        +files: dict
        +total_files: int
        +error_items: list
        +crawl_completed: str
        +total_time_seconds: float
    }
    
    class FileInfo {
        +name: str
        +type: str
        +path: str
        +properties: dict
    }
    
    class Properties {
        +drive_id: str
        +name: str
        +mime_type: str
        +drive_link: str
        +created_time: str
        +modified_time: str
        +size_bytes: str
        +owner_name: str
        +owner_email: str
    }
    
    class ErrorItem {
        +id: str
        +error: str
        +name: str
    }
    
    Metadata "1" --> "*" FileInfo : contains
    Metadata "1" --> "*" ErrorItem : contains
    FileInfo "1" --> "1" Properties : has
```

## 8. 処理の特徴

### 再開可能性
- `metadata.json`に進捗を逐次保存
- `visited_item_ids`で処理済みアイテムを管理
- 中断後も同じポイントから再開可能

### レート制限対策
- 各API呼び出し前に0.1秒の待機時間
- 過度なリクエストによるAPI制限を回避

### エラー耐性
- 個別ファイルのエラーが全体処理を止めない
- エラー情報を記録し、後で確認可能
- アクセス権限がないファイルは自動スキップ

### 柔軟なファイル形式対応
- Googleドキュメント → テキストエクスポート
- スプレッドシート → CSV形式
- プレゼンテーション → テキスト形式
- PDFファイル → pdfplumberでテキスト抽出 + クレンジング
- 通常ファイル → 直接ダウンロード

### PDF処理の最適化
- pdfplumberによる高精度レイアウト解析
- RAG検索精度向上のためのテキストクレンジング
- ページマーカー、制御文字、文字化けの除去
- Unicode正規化（NFKC形式）で文字の統一

---

最終更新: 2025年1月20日