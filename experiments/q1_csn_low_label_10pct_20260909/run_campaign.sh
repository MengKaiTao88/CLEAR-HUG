#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:-/root/107552503710}"
CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_csn_low_label_10pct_20260909"
RESULT="${ROOT}/results/q1-csn-low-label-10pct-20260909"
mkdir -p "${RESULT}"
exec > >(tee -a "${RESULT}/campaign.log") 2>&1
for seed in 43 45 47; do bash "${CODE}/train_val_seed.sh" "${seed}"; done
"${ROOT}/.venv/bin/python" "${CODE}/pretest_gate.py" --root "${RESULT}"
for seed in 43 45 47; do bash "${CODE}/formal_test_seed.sh" "${seed}"; done
"${ROOT}/.venv/bin/python" "${CODE}/summarize.py" --root "${RESULT}"
printf 'COMPLETE %s\n' "$(date -Is)" > "${RESULT}/campaign.status"
