#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/107552503710-1
EXP="$ROOT/src/CLEAR-HUG/experiments/q1_tolerant_generic_adapter_controls_20260920"
PY="$ROOT/.venv/bin/python"
TASKS=(PTBXL_form PTBXL_super PTBXL_sub PTBXL_rhythm CPSC CSN)
SEEDS=(42 46 55)
FRACTIONS=(0.01 0.1)
METHODS=(parameter-matched-mlp generic-adapter)
GPU="${1:?usage: run_low_label.sh GPU SHARD TOTAL_SHARDS}"
SHARD="${2:?usage: run_low_label.sh GPU SHARD TOTAL_SHARDS}"
TOTAL="${3:?usage: run_low_label.sh GPU SHARD TOTAL_SHARDS}"
index=0
for fraction in "${FRACTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for task in "${TASKS[@]}"; do
      for method in "${METHODS[@]}"; do
        if (( index % TOTAL == SHARD )); then
          "$PY" "$EXP/train_low_label.py" --root "$ROOT" --task "$task" \
            --seed "$seed" --fraction "$fraction" --method "$method" --device "cuda:$GPU"
        fi
        index=$((index + 1))
      done
    done
  done
done
