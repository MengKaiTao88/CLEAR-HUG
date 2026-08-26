#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?project root required}"
C="${ROOT}/mvp/q1_three_experiments_20260826"
OUT="${ROOT}/results/q1-three-experiments-20260826/second-encoder"
LOGDIR="${ROOT}/logs/q1-three-experiments-20260826"
mkdir -p "${OUT}" "${LOGDIR}"
exec > >(tee -a "${LOGDIR}/second-encoder.log") 2>&1
echo "started $(date -Is)" > "${LOGDIR}/second-encoder.status"

source "${ROOT}/mvp/activate_mvp.sh"
export PYTHONPATH="${C}:${ROOT}/mvp:${ROOT}/src/CLEAR-HUG:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
PY="${ROOT}/.venv/bin/python"

PTB="${ROOT}/data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
FEATURES="${OUT}/heartllm-features"
if [[ ! -s "${FEATURES}/feature-manifest.json" ]]; then
  "${PY}" "${C}/extract_heartllm_features.py" \
    --root "${ROOT}" --ptbxl "${PTB}" --records "${PTB}" \
    --checkpoint "${ROOT}/src/HeartLLM/ecg_tokenizer/result_tokenzier/best.pt" \
    --output "${FEATURES}" --batch-size 32
fi

for seed in 42 43 44; do
  "${PY}" "${C}/train_second_encoder_dev.py" \
    --features "${FEATURES}" --output "${OUT}/ptbxl-superdiagnostic-seed${seed}" \
    --task "ptbxl-superdiagnostic" --classes 5 --seed "${seed}" \
    --epochs 10 --batch-size 128 --num-workers 4
done
echo "complete $(date -Is)" > "${LOGDIR}/second-encoder.status"
