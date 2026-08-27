#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?project root required}"
C="${ROOT}/mvp/q1_three_experiments_20260826"
OUT="${ROOT}/results/q1-three-experiments-20260827/anchor-extension"
LOGDIR="${ROOT}/logs/q1-three-experiments-20260827"
mkdir -p "${OUT}" "${LOGDIR}"
exec > >(tee -a "${LOGDIR}/anchor-extension.log") 2>&1
echo "started $(date -Is) host=$(hostname)" > "${LOGDIR}/anchor-extension.status"

source "${ROOT}/mvp/activate_mvp.sh"
export PYTHONPATH="${C}:${ROOT}/mvp:${ROOT}/src/CLEAR-HUG:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
PY="${ROOT}/.venv/bin/python"

PTB_QRS="${ROOT}/src/CLEAR-HUG/datasets/ecg_datasets/PTBXL_QRS"
DEV="${ROOT}/src/CLEAR-HUG/datasets/ecg_datasets/ANCHORED_RESIDUAL_DEV"
DEV_V2="${ROOT}/src/CLEAR-HUG/datasets/ecg_datasets/ANCHORED_RESIDUAL_DEV_V2"

prepare_cache() {
  local task="$1" classes="$2" dataset="$3" baseline="$4" baseline_model="$5"
  local cache="${OUT}/cache/${task}"
  if [[ ! -s "${cache}/feature-manifest.json" ]]; then
    "${PY}" "${C}/cache_clear_features_flexible_dev.py" \
      --root "${ROOT}" --dataset "${dataset}" \
      --checkpoint "${ROOT}/checkpoints/released_ckpt.pth" \
      --baseline-checkpoint "${baseline}" \
      --baseline-model "${baseline_model}" \
      --output "${cache}" --classes "${classes}" \
      --batch-size 24 --num-workers 6
  fi
}

# PTB-XL Sub retains the frozen HUG baseline available on this node; Form and
# CSN use the corresponding frozen masked-DeepSets development checkpoints.
prepare_cache subdiagnostic 23 "${PTB_QRS}/subdiagnostic" \
  "${ROOT}/results/frozen-multitask-100/subdiagnostic-hug-seed42/checkpoint-best.pth" \
  "CLEAR_HUG_finetune_base"
prepare_cache form 19 "${DEV}/form" \
  "${ROOT}/results/deepsets-anchored-residual-extension/form-seed42/frozen-deepsets/checkpoint-best.pth" \
  "CLEAR_MASKED_DEEPSETS_finetune_base"
prepare_cache csn 38 "${DEV_V2}/csn" \
  "${ROOT}/results/deepsets-anchored-residual-six-task-csn-v2/csn-seed44/frozen-deepsets/checkpoint-best.pth" \
  "CLEAR_MASKED_DEEPSETS_finetune_base"

for task_spec in \
  "subdiagnostic 23" \
  "form 19" \
  "csn 38"; do
  read -r task classes <<< "${task_spec}"
  for seed in 42 43 44; do
    "${PY}" "${C}/train_anchor_ablation_dev.py" \
      --features "${OUT}/cache/${task}" \
      --output "${OUT}/${task}-seed${seed}" \
      --task "${task}" --classes "${classes}" --seed "${seed}" \
      --epochs 10 --batch-size 256 --num-workers 4
  done
done
echo "complete $(date -Is)" > "${LOGDIR}/anchor-extension.status"
