#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/setup.sh
cd "$(dirname "$0")/.."
round="${1:-1}"
session=tig023r-v2
colab usage
trap 'colab stop -s "$session"; colab usage' EXIT
colab new --gpu L4 -s "$session"
colab upload -s "$session" "items_r${round}.json" /content/items.json
# 1h hard execution limit; verify usage rate before running so total remains <=5 CU.
colab usage
colab exec -s "$session" -f scripts/remote.py --timeout 3600
colab download -s "$session" /content/answers.jsonl "results/local_r${round}.jsonl"
