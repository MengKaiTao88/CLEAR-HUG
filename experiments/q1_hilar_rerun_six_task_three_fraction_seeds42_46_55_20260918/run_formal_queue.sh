#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:?}"; GPU="${1:?}"; SEED="${2:?}"
CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_hilar_rerun_six_task_three_fraction_seeds42_46_55_20260918"
for fraction in 1pct 10pct 100pct; do for task in superdiagnostic subdiagnostic form rhythm cpsc2018 csn; do CUDA_VISIBLE_DEVICES="${GPU}" bash "${CODE}/run_formal_unit.sh" "${fraction}" "${task}" "${SEED}"; done; done
