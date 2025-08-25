"""
チェックポイント管理モジュール

エージェントの状態を保存・復元する機能を提供します。
途中で中断した処理を後で再開したり、
デバッグのために特定の状態からやり直したりできます。

主な機能:
1. エージェント状態の保存（圧縮対応）
2. チェックポイントからの状態復元
3. チェックポイントの管理（一覧、削除、最大保存数制限）
4. メタデータ管理
"""

# 標準ライブラリのインポート
import json  # JSON形式のデータ処理
import pickle  # Pythonオブジェクトのシリアライズ
import gzip  # データ圧縮
import hashlib  # ハッシュ値生成
from pathlib import Path  # ファイルパス操作
from typing import Dict, Any, Optional, List  # 型ヒント用
from datetime import datetime  # タイムスタンプ記録用

# Pydanticモデル（データ検証用）
from pydantic import BaseModel, Field  # 型安全なデータモデル定義

# ローカルモジュールのインポート
from agent_state import AgentState, CheckpointData  # エージェント状態とチェックポイントデータ


class CheckpointManager:
    """
    チェックポイント管理クラス
    
    エージェントの状態をファイルシステムに保存し、
    必要に応じて復元できるように管理します。
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./data/checkpoints",
        max_checkpoints: int = 10,
        enable_compression: bool = True,
        verbose: bool = False
    ):
        """
        チェックポイントマネージャーの初期化
        
        保存ディレクトリを作成し、メタデータを読み込みます。
        
        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ
            max_checkpoints: 最大保存数（古いものから自動削除）
            enable_compression: gzip圧縮を有効にするか
            verbose: 詳細ログ出力の有効/無効
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.enable_compression = enable_compression
        self.verbose = verbose
        
        # ディレクトリ作成
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # メタデータファイル
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        メタデータを読み込み
        
        チェックポイントの一覧と情報をJSONファイルから読み込みます。
        
        Returns:
            メタデータの辞書
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"checkpoints": {}, "order": []}  # 初期状態
    
    def _save_metadata(self):
        """
        メタデータを保存
        
        チェックポイントの一覧と情報をJSONファイルに保存します。
        """
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _generate_checkpoint_id(self, state: AgentState) -> str:
        """
        チェックポイントIDを生成
        
        タイムスタンプと状態のハッシュ値を組み合わせて、
        一意のIDを生成します。
        
        Args:
            state: 保存するエージェント状態
            
        Returns:
            一意のチェックポイントID
        """
        # タイムスタンプを生成（年月日_時分秒）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 状態のハッシュ値を生成（重複防止）
        state_hash = hashlib.md5(
            json.dumps(str(state), sort_keys=True).encode()
        ).hexdigest()[:8]  # 最初の8文字だけ使用
        
        return f"checkpoint_{timestamp}_{state_hash}"
    
    def save_checkpoint(
        self,
        state: AgentState,
        step_name: str,
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        チェックポイントを保存
        
        エージェントの現在の状態をファイルに保存し、
        メタデータを更新します。
        
        Args:
            state: 保存するエージェント状態
            step_name: 現在のステップ名（"reason", "act" など）
            iteration: 現在のイテレーション番号
            metadata: 追加メタデータ（オプション）
            
        Returns:
            生成されたチェックポイントID
        """
        checkpoint_id = self._generate_checkpoint_id(state)
        
        if self.verbose:
            print(f"💾 Saving checkpoint: {checkpoint_id}")
        
        # チェックポイントデータを作成
        checkpoint_data = CheckpointData(
            checkpoint_id=checkpoint_id,
            state=dict(state),  # TypedDictをdictに変換
            step_name=step_name,
            iteration=iteration,
            metadata=metadata or {}
        )
        
        # ファイルパスを決定
        file_extension = ".pkl.gz" if self.enable_compression else ".pkl"
        file_path = self.checkpoint_dir / f"{checkpoint_id}{file_extension}"
        
        try:
            # データを保存
            if self.enable_compression:
                with gzip.open(file_path, "wb") as f:
                    pickle.dump(checkpoint_data.model_dump(), f)
            else:
                with open(file_path, "wb") as f:
                    pickle.dump(checkpoint_data.model_dump(), f)
            
            # メタデータを更新
            self.metadata["checkpoints"][checkpoint_id] = {
                "file_path": str(file_path),
                "timestamp": checkpoint_data.timestamp.isoformat(),
                "step_name": step_name,
                "iteration": iteration,
                "metadata": metadata or {}
            }
            
            # 順序リストに追加
            self.metadata["order"].append(checkpoint_id)
            
            # 古いチェックポイントを削除
            self._cleanup_old_checkpoints()
            
            # メタデータを保存
            self._save_metadata()
            
            if self.verbose:
                print(f"✅ Checkpoint saved: {file_path}")
            
            return checkpoint_id
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """
        チェックポイントを読み込み
        
        指定されたチェックポイントをファイルから読み込み、
        エージェント状態を復元します。
        
        Args:
            checkpoint_id: チェックポイントID
            
        Returns:
            チェックポイントデータ（存在しない場合はNone）
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            if self.verbose:
                print(f"⚠️ Checkpoint not found: {checkpoint_id}")
            return None
        
        checkpoint_info = self.metadata["checkpoints"][checkpoint_id]
        file_path = Path(checkpoint_info["file_path"])
        
        if not file_path.exists():
            if self.verbose:
                print(f"⚠️ Checkpoint file not found: {file_path}")
            return None
        
        try:
            # データを読み込み
            if self.enable_compression:
                with gzip.open(file_path, "rb") as f:
                    data = pickle.load(f)
            else:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
            
            # CheckpointDataオブジェクトとして復元
            checkpoint_data = CheckpointData(**data)
            
            if self.verbose:
                print(f"✅ Checkpoint loaded: {checkpoint_id}")
            
            return checkpoint_data
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load checkpoint: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        利用可能なチェックポイントのリストを取得
        
        保存されているすべてのチェックポイントの
        情報をリストとして返します。
        
        Returns:
            チェックポイント情報のリスト（新しい順）
        """
        checkpoints = []
        
        for checkpoint_id in self.metadata["order"]:
            if checkpoint_id in self.metadata["checkpoints"]:
                info = self.metadata["checkpoints"][checkpoint_id].copy()
                info["checkpoint_id"] = checkpoint_id
                checkpoints.append(info)
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        チェックポイントを削除
        
        指定されたチェックポイントをファイルシステムから削除し、
        メタデータを更新します。
        
        Args:
            checkpoint_id: 削除するチェックポイントID
            
        Returns:
            削除成功の場合True、失敗の場合False
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            if self.verbose:
                print(f"⚠️ Checkpoint not found: {checkpoint_id}")
            return False
        
        try:
            # ファイルを削除
            file_path = Path(self.metadata["checkpoints"][checkpoint_id]["file_path"])
            if file_path.exists():
                file_path.unlink()
            
            # メタデータから削除
            del self.metadata["checkpoints"][checkpoint_id]
            self.metadata["order"].remove(checkpoint_id)
            
            # メタデータを保存
            self._save_metadata()
            
            if self.verbose:
                print(f"✅ Checkpoint deleted: {checkpoint_id}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to delete checkpoint: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """
        古いチェックポイントを削除
        
        最大保存数を超えた場合、古いチェックポイントから
        自動的に削除して、ストレージを管理します。
        """
        if len(self.metadata["order"]) > self.max_checkpoints:
            # 削除するチェックポイント数を計算
            num_to_delete = len(self.metadata["order"]) - self.max_checkpoints
            
            # 古い順に削除（FIFO）
            for checkpoint_id in self.metadata["order"][:num_to_delete]:
                self.delete_checkpoint(checkpoint_id)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        最新のチェックポイントIDを取得
        
        保存されているチェックポイントの中で
        最も新しいもののIDを返します。
        
        Returns:
            最新のチェックポイントID（存在しない場合はNone）
        """
        if self.metadata["order"]:
            return self.metadata["order"][-1]  # リストの最後が最新
        return None
    
    def clear_all_checkpoints(self):
        """
        すべてのチェックポイントを削除
        
        保存されているすべてのチェックポイントを削除し、
        メタデータをリセットします。
        """
        if self.verbose:
            print("🗑️ Clearing all checkpoints...")
        
        # すべてのチェックポイントファイルを削除
        for checkpoint_id in list(self.metadata["order"]):
            self.delete_checkpoint(checkpoint_id)
        
        # メタデータをリセット
        self.metadata = {"checkpoints": {}, "order": []}
        self._save_metadata()
        
        if self.verbose:
            print("✅ All checkpoints cleared")
    
    def get_checkpoint_size(self, checkpoint_id: str) -> Optional[int]:
        """
        チェックポイントのファイルサイズを取得
        
        指定されたチェックポイントのファイルサイズを
        バイト単位で返します。
        
        Args:
            checkpoint_id: チェックポイントID
            
        Returns:
            ファイルサイズ（バイト）、存在しない場合はNone
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            return None
        
        file_path = Path(self.metadata["checkpoints"][checkpoint_id]["file_path"])
        if file_path.exists():
            return file_path.stat().st_size
        
        return None
    
    def get_total_size(self) -> int:
        """
        すべてのチェックポイントの合計サイズを取得
        
        保存されているすべてのチェックポイントの
        ファイルサイズの合計を計算します。
        
        Returns:
            合計サイズ（バイト）
        """
        total_size = 0
        for checkpoint_id in self.metadata["order"]:
            size = self.get_checkpoint_size(checkpoint_id)
            if size:
                total_size += size
        
        return total_size
