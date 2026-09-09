#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:-/root/107552503710}"
CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_csn_low_label_10pct_20260909"
CAMPAIGN="${CSN_LOW_LABEL_CAMPAIGN:-q1-csn-low-label-10pct-20260909}"
TRAIN_FRACTION="${CSN_LOW_LABEL_FRACTION:-0.1}"
EXPECTED_TRAIN_RECORDS="${CSN_LOW_LABEL_RECORDS:-1654}"
RESULT="${ROOT}/results/${CAMPAIGN}"
mkdir -p "${RESULT}"
exec > >(tee -a "${RESULT}/campaign.log") 2>&1
for seed in 43 45 47; do bash "${CODE}/train_val_seed.sh" "${seed}"; done
"${ROOT}/.venv/bin/python" "${CODE}/pretest_gate.py" --root "${RESULT}" \
  --train-fraction "${TRAIN_FRACTION}" --train-records "${EXPECTED_TRAIN_RECORDS}"
for seed in 43 45 47; do bash "${CODE}/formal_test_seed.sh" "${seed}"; done
"${ROOT}/.venv/bin/python" "${CODE}/summarize.py" --root "${RESULT}" \
  --train-fraction "${TRAIN_FRACTION}"
printf 'COMPLETE %s\n' "$(date -Is)" > "${RESULT}/campaign.status"
