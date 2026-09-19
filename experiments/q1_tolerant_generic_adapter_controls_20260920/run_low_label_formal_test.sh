#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/107552503710-1
EXP="$ROOT/src/CLEAR-HUG/experiments/q1_tolerant_generic_adapter_controls_20260920"
PY="$ROOT/.venv/bin/python"
GPU="${1:?usage: run_low_label_formal_test.sh GPU SHARD TOTAL_SHARDS}"
SHARD="${2:?usage: run_low_label_formal_test.sh GPU SHARD TOTAL_SHARDS}"
TOTAL="${3:?usage: run_low_label_formal_test.sh GPU SHARD TOTAL_SHARDS}"
"$PY" "$EXP/evaluate_low_label_test.py" --root "$ROOT" --gpu "$GPU" \
  --shard "$SHARD" --total-shards "$TOTAL"
