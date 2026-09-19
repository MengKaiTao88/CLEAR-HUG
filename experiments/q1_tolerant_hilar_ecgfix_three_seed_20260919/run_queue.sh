#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/root/107552503710-1}"
PYTHON="$ROOT/envs/ecg-fix/bin/python"
HERE="$ROOT/src/CLEAR-HUG/experiments/q1_tolerant_hilar_ecgfix_three_seed_20260919"
OUT="$ROOT/results/q1-tolerant-hilar-ecgfix-three-seed-20260919"
mkdir -p "$OUT/logs"
for seed in 46 55; do
  for task in PTBXL_form PTBXL_super PTBXL_sub PTBXL_rhythm CPSC CSN; do
    "$PYTHON" "$HERE/train.py" --root "$ROOT" --seed "$seed" --task "$task" --device cuda:0 \
      > "$OUT/logs/seed-${seed}-${task}.log" 2>&1
  done
done
date -Is > "$OUT/queue.complete"
