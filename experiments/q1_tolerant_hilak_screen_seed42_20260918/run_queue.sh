#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/root/107552503710-1}"
PYTHON="$ROOT/envs/ecg-fix/bin/python"
HERE="$ROOT/src/CLEAR-HUG/experiments/q1_tolerant_hilak_screen_seed42_20260918"
OUT="$ROOT/results/q1-tolerant-hilak-screen-seed42-20260918"
mkdir -p "$OUT"
for task in superdiagnostic form cpsc2018 csn; do
  "$PYTHON" "$HERE/run_task.py" --root "$ROOT" --task "$task" --device cuda:0 \
    > "$OUT/$task.log" 2>&1
done
"$PYTHON" "$HERE/summarize.py" --root "$ROOT" > "$OUT/summary.json"

