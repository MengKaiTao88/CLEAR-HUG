#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?project root required}"
C="${ROOT}/mvp/q1_three_experiments_20260826"
OUT="${ROOT}/results/q1-three-experiments-20260827/heartllm-aligned"
LOGDIR="${ROOT}/logs/q1-three-experiments-20260827"
FEATURES="${OUT}/features"
mkdir -p "${OUT}" "${LOGDIR}"
exec > >(tee -a "${LOGDIR}/heartllm-aligned.log") 2>&1
echo "started $(date -Is) host=$(hostname)" > "${LOGDIR}/heartllm-aligned.status"

source "${ROOT}/mvp/activate_mvp.sh"
export PYTHONPATH="${C}:${ROOT}/mvp:${ROOT}/src/CLEAR-HUG:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
PY="${ROOT}/.venv/bin/python"
DATASET="${ROOT}/src/CLEAR-HUG/datasets/ecg_datasets/PTBXL_QRS/superdiagnostic"
CHECKPOINT="${ROOT}/src/HeartLLM/ecg_tokenizer/result_tokenzier/best.pt"

if [[ ! -s "${FEATURES}/feature-manifest.json" ]]; then
  "${PY}" "${C}/extract_heartllm_aligned_features.py" \
    --root "${ROOT}" --dataset "${DATASET}" --checkpoint "${CHECKPOINT}" \
    --output "${FEATURES}" --classes 5 --batch-size 16 --segment-batch 2048
fi

for seed in 42 43 44; do
  "${PY}" "${C}/train_heartllm_aligned_paired_dev.py" \
    --features "${FEATURES}" \
    --output "${OUT}/ptbxl-superdiagnostic-seed${seed}" \
    --task "ptbxl-superdiagnostic" --classes 5 --seed "${seed}" \
    --epochs 10 --batch-size 256 --num-workers 4 --anchor-lambda 0.1
done
echo "complete $(date -Is)" > "${LOGDIR}/heartllm-aligned.status"
