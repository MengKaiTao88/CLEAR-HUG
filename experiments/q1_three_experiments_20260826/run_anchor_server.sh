#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?project root required}"
C="${ROOT}/mvp/q1_three_experiments_20260826"
OUT="${ROOT}/results/q1-three-experiments-20260826/anchor"
LOGDIR="${ROOT}/logs/q1-three-experiments-20260826"
mkdir -p "${OUT}" "${LOGDIR}"
exec > >(tee -a "${LOGDIR}/anchor.log") 2>&1
echo "started $(date -Is)" > "${LOGDIR}/anchor.status"

source "${ROOT}/mvp/activate_mvp.sh"
export PYTHONPATH="${C}:${ROOT}/mvp:${ROOT}/src/CLEAR-HUG:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
PY="${ROOT}/.venv/bin/python"

PTB_QRS="${ROOT}/src/CLEAR-HUG/datasets/ecg_datasets/PTBXL_QRS"
CPSC_QRS="${ROOT}/src/CLEAR-HUG/datasets/ecg_datasets/CPSC2018_QRS/data"

prepare_cache() {
  local task="$1" classes="$2" dataset="$3" baseline="$4" labels_root="$5"
  local cache="${OUT}/cache/${task}"
  if [[ ! -s "${cache}/feature-manifest.json" ]]; then
    "${PY}" "${C}/cache_clear_features.py" \
      --root "${ROOT}" --dataset "${dataset}" \
      --checkpoint "${ROOT}/checkpoints/released_ckpt.pth" \
      --baseline-checkpoint "${baseline}" \
      --labels-root "${labels_root}" \
      --output "${cache}" --classes "${classes}" \
      --batch-size 24 --num-workers 6
  fi
}

prepare_cache superdiagnostic 5 "${PTB_QRS}/superdiagnostic" \
  "${ROOT}/baseline-checkpoints/q1/superdiagnostic-seed42/checkpoint-best.pth" \
  "${PTB_QRS}/superdiagnostic"
prepare_cache rhythm 12 "${PTB_QRS}/rhythm" \
  "${ROOT}/baseline-checkpoints/q1/rhythm-seed42/checkpoint-best.pth" \
  "${PTB_QRS}/rhythm"
prepare_cache cpsc2018 9 "${CPSC_QRS}" \
  "${ROOT}/baseline-checkpoints/q1/cpsc2018-seed42/checkpoint-best.pth" \
  "${ROOT}/src/CLEAR-HUG/datasets/ecg_datasets/CPSC2018/data"

for task_spec in \
  "superdiagnostic 5" \
  "rhythm 12" \
  "cpsc2018 9"; do
  read -r task classes <<< "${task_spec}"
  for seed in 42 43 44; do
    "${PY}" "${C}/train_anchor_ablation_dev.py" \
      --features "${OUT}/cache/${task}" --output "${OUT}/${task}-seed${seed}" \
      --task "${task}" --classes "${classes}" --seed "${seed}" \
      --epochs 10 --batch-size 256 --num-workers 4
  done
done
echo "complete $(date -Is)" > "${LOGDIR}/anchor.status"
