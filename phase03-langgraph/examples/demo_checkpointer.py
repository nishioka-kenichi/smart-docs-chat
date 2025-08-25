#!/usr/bin/env python
"""
チェックポイント管理のデモンストレーション

このスクリプトは、LangGraphエージェントの状態を保存・復元する
チェックポイント機能を実際に確認するためのデモンストレーションです。

実行方法:
    python examples/demo_checkpointer.py

デバッグ方法:
    1. VSCodeでこのファイルを開く
    2. ブレークポイントを設定（例：save_checkpoint関数内）
    3. F5キーでデバッグ実行
"""

import sys
import tempfile
import shutil
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from checkpointer import CheckpointManager
from agent_state import create_initial_state, add_reasoning_step


def main():
    """メイン処理"""
    print("="*60)
    print("チェックポイント管理 - 学習モード")
    print("="*60)
    
    # テスト用の一時ディレクトリを作成
    test_dir = Path(tempfile.mkdtemp(prefix="checkpoint_test_"))
    print(f"テスト用ディレクトリ: {test_dir}")
    
    try:
        # 1. CheckpointManagerの初期化
        print("\n1. CheckpointManagerの初期化")
        print("-"*40)
        
        manager = CheckpointManager(
            checkpoint_dir=str(test_dir),
            max_checkpoints=3,
            enable_compression=True,
            verbose=True
        )
        print("✅ 初期化成功")
        print(f"  保存先: {manager.checkpoint_dir}")
        print(f"  最大保存数: {manager.max_checkpoints}")
        print(f"  圧縮: {manager.enable_compression}")
        
        # 2. チェックポイントの保存
        print("\n2. チェックポイントの保存")
        print("-"*40)
        
        # テスト用の状態を作成
        state = create_initial_state("テストクエリ", max_iterations=5)
        state = add_reasoning_step(
            state,
            thought="これはテスト用の推論です",
            action="test_action",
            observation="テスト結果"
        )
        
        # チェックポイントを保存
        checkpoint_id = manager.save_checkpoint(
            state=state,
            step_name="test_step",
            iteration=1,
            metadata={"test": True}
        )
        print(f"保存されたID: {checkpoint_id}")
        
        # ファイルサイズを確認
        size = manager.get_checkpoint_size(checkpoint_id)
        print(f"ファイルサイズ: {size} bytes")
        
        # 3. チェックポイントの読み込み
        print("\n3. チェックポイントの読み込み")
        print("-"*40)
        
        loaded = manager.load_checkpoint(checkpoint_id)
        if loaded:
            print("✅ 読み込み成功")
            print(f"  ステップ名: {loaded.step_name}")
            print(f"  イテレーション: {loaded.iteration}")
            print(f"  保存時刻: {loaded.timestamp}")
            
            # 状態の内容を確認
            if "messages" in loaded.state:
                print(f"  メッセージ数: {len(loaded.state['messages'])}")
            if "reasoning_steps" in loaded.state:
                print(f"  推論ステップ数: {len(loaded.state['reasoning_steps'])}")
        
        # 4. 複数のチェックポイント管理
        print("\n4. 複数のチェックポイント管理")
        print("-"*40)
        
        # 追加のチェックポイントを作成
        checkpoint_ids = [checkpoint_id]
        for i in range(2, 5):
            state["iteration_count"] = i
            cp_id = manager.save_checkpoint(
                state=state,
                step_name=f"step_{i}",
                iteration=i
            )
            checkpoint_ids.append(cp_id)
            print(f"チェックポイント {i} 保存: {cp_id[:20]}...")
        
        # 一覧を取得
        print("\n保存されているチェックポイント:")
        checkpoints = manager.list_checkpoints()
        for cp in checkpoints:
            print(f"  - {cp['checkpoint_id'][:20]}... (iteration: {cp['iteration']})")
        
        print(f"\n注: 最大{manager.max_checkpoints}個のため、古いものは自動削除されました")
        
        # 5. 最新のチェックポイント取得
        print("\n5. 最新のチェックポイント")
        print("-"*40)
        
        latest_id = manager.get_latest_checkpoint()
        if latest_id:
            print(f"最新ID: {latest_id[:20]}...")
            latest = manager.load_checkpoint(latest_id)
            if latest:
                print(f"  イテレーション: {latest.iteration}")
        
        # 6. 統計情報
        print("\n6. 統計情報")
        print("-"*40)
        
        total_size = manager.get_total_size()
        print(f"合計サイズ: {total_size} bytes")
        print(f"チェックポイント数: {len(manager.list_checkpoints())}")
        
        # 7. 個別削除のテスト
        print("\n7. チェックポイントの削除")
        print("-"*40)
        
        if checkpoints:
            target_id = checkpoints[0]["checkpoint_id"]
            success = manager.delete_checkpoint(target_id)
            if success:
                print(f"✅ 削除成功: {target_id[:20]}...")
            print(f"残りのチェックポイント数: {len(manager.list_checkpoints())}")
        
    finally:
        # クリーンアップ
        print("\n8. クリーンアップ")
        print("-"*40)
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"テストディレクトリを削除しました: {test_dir}")
    
    print("\n" + "="*60)
    print("✅ すべてのテストが完了しました")
    print("="*60)


if __name__ == "__main__":
    main()