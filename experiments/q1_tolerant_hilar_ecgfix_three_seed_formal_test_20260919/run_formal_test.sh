#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/root/107552503710-1}"
HERE="$ROOT/src/CLEAR-HUG/experiments/q1_tolerant_hilar_ecgfix_three_seed_formal_test_20260919"
OUT="$ROOT/results/q1-tolerant-hilar-ecgfix-three-seed-formal-test-20260919"
PYTHON="$ROOT/envs/ecg-fix/bin/python"
mkdir -p "$OUT/logs"
worker() {
  local device="$1"; shift
  for task in "$@"; do
    "$PYTHON" "$HERE/extract_masked_features.py" --root "$ROOT" --task "$task" --device "$device" \
      > "$OUT/logs/${task}-masked.log" 2>&1
    "$PYTHON" "$HERE/extract_local_features.py" --root "$ROOT" --task "$task" --device "$device" --batch-size 64 \
      > "$OUT/logs/${task}-local.log" 2>&1
  done
}
worker cuda:0 PTBXL_form PTBXL_sub CPSC &
pid0=$!
worker cuda:1 PTBXL_super PTBXL_rhythm CSN &
pid1=$!
wait "$pid0" "$pid1"
"$PYTHON" "$HERE/evaluate.py" > "$OUT/evaluate.log" 2>&1
date -Is > "$OUT/complete"
