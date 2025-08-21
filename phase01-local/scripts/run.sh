#!/bin/bash
# 仮想環境を使って実行するヘルパースクリプト

# スクリプトのディレクトリに移動
cd "$(dirname "$0")"

# 仮想環境をアクティベート
source .venv/bin/activate

# 引数で指定されたPythonスクリプトを実行
python "$@"