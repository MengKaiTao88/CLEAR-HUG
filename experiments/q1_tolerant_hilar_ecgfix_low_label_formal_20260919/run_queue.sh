#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/root/107552503710-1}"
HERE="$ROOT/src/CLEAR-HUG/experiments/q1_tolerant_hilar_ecgfix_low_label_formal_20260919"
OUT="$ROOT/results/q1-tolerant-hilar-ecgfix-low-label-formal-20260919"
PYTHON="$ROOT/envs/ecg-fix/bin/python"
mkdir -p "$OUT/logs"
worker() {
  local device="$1"; local fraction="$2"; shift 2
  for seed in 42 46 55; do
    for task in "$@"; do
      "$PYTHON" "$HERE/train.py" --root "$ROOT" --fraction "$fraction" \
        --seed "$seed" --task "$task" --device "$device" \
        > "$OUT/logs/fraction-${fraction}-seed-${seed}-${task}.log" 2>&1
    done
  done
}
# Split each fraction across both GPUs; 1% finishes first, then 10%.
worker cuda:0 0.01 PTBXL_form PTBXL_sub CPSC & p0=$!
worker cuda:1 0.01 PTBXL_super PTBXL_rhythm CSN & p1=$!
wait "$p0" "$p1"
worker cuda:0 0.1 PTBXL_form PTBXL_sub CPSC & p0=$!
worker cuda:1 0.1 PTBXL_super PTBXL_rhythm CSN & p1=$!
wait "$p0" "$p1"
"$PYTHON" "$HERE/report.py" > "$OUT/validation-report.log" 2>&1
date -Is > "$OUT/validation.complete"
