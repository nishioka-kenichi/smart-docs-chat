#!/usr/bin/env python3
"""
デバッグ支援ユーティリティ
開発時のデバッグを効率化するためのヘルパー関数集
"""

import logging
import sys
import os
from datetime import datetime
from typing import Any, Dict, Optional
import json
import traceback


def setup_debug_logging(log_level=logging.DEBUG, log_dir="./logs"):
    """
    デバッグ用ログ設定を初期化
    
    Args:
        log_level: ログレベル（DEBUG, INFO, WARNING, ERROR）
        log_dir: ログファイルの保存ディレクトリ
    
    Returns:
        設定済みのloggerオブジェクト
    """
    # ログディレクトリ作成
    os.makedirs(log_dir, exist_ok=True)
    
    # ログファイル名（タイムスタンプ付き）
    log_file = f"{log_dir}/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # ログフォーマット設定
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    # ルートロガーの設定
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # コンソールにも出力
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"デバッグログ開始: {log_file}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"作業ディレクトリ: {os.getcwd()}")
    
    return logger


def debug_checkpoint(locals_dict: Dict[str, Any], message: str = "Debug Checkpoint", 
                    exclude_keys: Optional[list] = None):
    """
    ブレークポイントで変数を確認しやすくする
    
    Args:
        locals_dict: locals()の結果を渡す
        message: チェックポイントのメッセージ
        exclude_keys: 表示から除外するキーのリスト
    """
    exclude_keys = exclude_keys or ['__builtins__', '__cached__', '__doc__', '__file__', 
                                    '__loader__', '__name__', '__package__', '__spec__']
    
    print(f"\n{'='*60}")
    print(f"🔍 {message}")
    print(f"{'='*60}")
    print(f"📍 場所: {traceback.extract_stack()[-2].filename}:{traceback.extract_stack()[-2].lineno}")
    print(f"⏰ 時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    for key, value in sorted(locals_dict.items()):
        if key not in exclude_keys and not key.startswith('__'):
            try:
                value_str = repr(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + '...'
                print(f"  {key}: {type(value).__name__} = {value_str}")
            except Exception as e:
                print(f"  {key}: {type(value).__name__} = <表示エラー: {e}>")
    
    print(f"{'='*60}\n")


def inspect_object(obj: Any, name: str = "Object", max_depth: int = 2):
    """
    オブジェクトの詳細情報を表示
    
    Args:
        obj: 調査対象のオブジェクト
        name: オブジェクトの名前（表示用）
        max_depth: 再帰的に調査する深さ
    """
    print(f"\n{'🔍 Object Inspector ':=^60}")
    print(f"Name: {name}")
    print(f"Type: {type(obj).__name__} ({type(obj)})")
    print(f"ID: {id(obj)}")
    print(f"Size: {sys.getsizeof(obj)} bytes")
    
    # 基本情報
    if hasattr(obj, '__doc__') and obj.__doc__:
        print(f"Doc: {obj.__doc__[:100]}...")
    
    # 長さ情報
    if hasattr(obj, '__len__'):
        try:
            print(f"Length: {len(obj)}")
        except:
            pass
    
    # 属性情報
    if hasattr(obj, '__dict__'):
        print("\n📦 Attributes:")
        for attr, value in sorted(obj.__dict__.items()):
            try:
                value_repr = repr(value)[:50]
                print(f"  - {attr}: {type(value).__name__} = {value_repr}")
            except:
                print(f"  - {attr}: {type(value).__name__} = <表示不可>")
    
    # メソッド情報
    methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m, None))]
    if methods:
        print(f"\n🔧 Methods ({len(methods)} total):")
        for method in methods[:10]:  # 最初の10個のみ表示
            print(f"  - {method}()")
        if len(methods) > 10:
            print(f"  ... and {len(methods) - 10} more")
    
    # 特殊なオブジェクトタイプの追加情報
    if isinstance(obj, dict):
        print(f"\n🗝️ Dictionary Keys ({len(obj)}):")
        for key in list(obj.keys())[:10]:
            print(f"  - {repr(key)}")
    elif isinstance(obj, (list, tuple)):
        print(f"\n📝 First 5 items:")
        for i, item in enumerate(obj[:5]):
            print(f"  [{i}]: {type(item).__name__} = {repr(item)[:50]}")
    
    print("=" * 60)


def log_api_call(func_name: str, **kwargs):
    """
    API呼び出しをログに記録
    
    Args:
        func_name: API関数名
        **kwargs: API呼び出しのパラメータ
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🌐 API Call: {func_name}")
    for key, value in kwargs.items():
        logger.debug(f"  - {key}: {repr(value)[:100]}")


def trace_execution(func):
    """
    関数の実行をトレースするデコレータ
    
    使用例:
        @trace_execution
        def my_function(param1, param2):
            return result
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        func_name = func.__name__
        
        # 関数開始
        logger.debug(f"➡️ Starting: {func_name}")
        logger.debug(f"   Args: {repr(args)[:100]}")
        logger.debug(f"   Kwargs: {repr(kwargs)[:100]}")
        
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # 正常終了
            logger.debug(f"✅ Completed: {func_name} ({elapsed:.3f}s)")
            logger.debug(f"   Result: {repr(result)[:100]}")
            return result
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # エラー発生
            logger.error(f"❌ Failed: {func_name} ({elapsed:.3f}s)")
            logger.error(f"   Error: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            raise
    
    return wrapper


def save_debug_snapshot(data: Any, filename: str = None, directory: str = "./debug_snapshots"):
    """
    デバッグ用にデータのスナップショットを保存
    
    Args:
        data: 保存するデータ
        filename: ファイル名（省略時は自動生成）
        directory: 保存先ディレクトリ
    """
    os.makedirs(directory, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"snapshot_{timestamp}.json"
    
    filepath = os.path.join(directory, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            else:
                f.write(str(data))
        
        print(f"📸 Debug snapshot saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Failed to save snapshot: {e}")
        return None


def compare_objects(obj1: Any, obj2: Any, name1: str = "Object1", name2: str = "Object2"):
    """
    2つのオブジェクトを比較して違いを表示
    
    Args:
        obj1, obj2: 比較するオブジェクト
        name1, name2: オブジェクトの名前（表示用）
    """
    print(f"\n{'🔄 Object Comparison ':=^60}")
    print(f"Comparing: {name1} vs {name2}")
    print("=" * 60)
    
    # 型の比較
    if type(obj1) != type(obj2):
        print(f"❌ Type mismatch: {type(obj1).__name__} != {type(obj2).__name__}")
    else:
        print(f"✅ Same type: {type(obj1).__name__}")
    
    # サイズの比較
    size1, size2 = sys.getsizeof(obj1), sys.getsizeof(obj2)
    print(f"Size: {size1} bytes vs {size2} bytes (diff: {size2-size1:+d})")
    
    # 辞書の場合
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        keys1, keys2 = set(obj1.keys()), set(obj2.keys())
        
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        common = keys1 & keys2
        
        if only_in_1:
            print(f"\n🔵 Only in {name1}: {only_in_1}")
        if only_in_2:
            print(f"\n🟢 Only in {name2}: {only_in_2}")
        
        print(f"\n🔍 Common keys ({len(common)}):")
        for key in list(common)[:10]:
            val1, val2 = obj1[key], obj2[key]
            if val1 != val2:
                print(f"  ❌ {key}: {repr(val1)[:30]} != {repr(val2)[:30]}")
            else:
                print(f"  ✅ {key}: same")
    
    # リストの場合
    elif isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        len1, len2 = len(obj1), len(obj2)
        print(f"Length: {len1} vs {len2}")
        
        for i in range(min(len1, len2, 10)):  # 最初の10個を比較
            if obj1[i] != obj2[i]:
                print(f"  ❌ [{i}]: {repr(obj1[i])[:30]} != {repr(obj2[i])[:30]}")
    
    print("=" * 60)


# 使用例とテスト
if __name__ == "__main__":
    # ログ設定
    logger = setup_debug_logging()
    
    # サンプルデータ
    test_dict = {"key1": "value1", "key2": [1, 2, 3]}
    test_list = [1, 2, 3, 4, 5]
    
    # デバッグチェックポイント
    debug_checkpoint(locals(), "テスト実行中")
    
    # オブジェクト検査
    inspect_object(test_dict, "テスト辞書")
    
    # スナップショット保存
    save_debug_snapshot(test_dict, "test_data.json")
    
    # オブジェクト比較
    test_dict2 = {"key1": "value1", "key3": "new"}
    compare_objects(test_dict, test_dict2, "Dict1", "Dict2")
    
    print("✅ デバッグヘルパーのテスト完了")