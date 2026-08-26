#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?project root required}"
C="${ROOT}/mvp/q1_three_experiments_20260826"
OUT="${ROOT}/results/q1-three-experiments-20260826/second-encoder-paired"
LOGDIR="${ROOT}/logs/q1-three-experiments-20260826"
FEATURES="${ROOT}/results/q1-three-experiments-20260826/second-encoder/heartllm-features"
mkdir -p "${OUT}" "${LOGDIR}"
exec > >(tee -a "${LOGDIR}/second-encoder-paired.log") 2>&1
echo "started $(date -Is)" > "${LOGDIR}/second-encoder-paired.status"

source "${ROOT}/mvp/activate_mvp.sh"
export PYTHONPATH="${C}:${ROOT}/mvp:${ROOT}/src/CLEAR-HUG:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
PY="${ROOT}/.venv/bin/python"

if [[ ! -s "${FEATURES}/feature-manifest.json" ]]; then
  echo "missing frozen HeartLLM feature manifest: ${FEATURES}" >&2
  exit 2
fi

for seed in 42 43 44; do
  "${PY}" "${C}/train_second_encoder_paired_dev.py" \
    --features "${FEATURES}" \
    --output "${OUT}/ptbxl-superdiagnostic-seed${seed}" \
    --task "ptbxl-superdiagnostic" --classes 5 --seed "${seed}" \
    --epochs 10 --batch-size 256 --num-workers 4 --anchor-lambda 0.1
done
echo "complete $(date -Is)" > "${LOGDIR}/second-encoder-paired.status"
