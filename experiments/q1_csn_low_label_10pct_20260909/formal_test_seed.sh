#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:-/root/107552503710}"
SEED="${1:?seed required}"
case "${SEED}" in 43|45|47) ;; *) exit 2 ;; esac
CAMPAIGN="${CSN_LOW_LABEL_CAMPAIGN:-q1-csn-low-label-10pct-20260909}"
TRAIN_FRACTION="${CSN_LOW_LABEL_FRACTION:-0.1}"
CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_csn_low_label_10pct_20260909"
INPUT="${ROOT}/campaign-inputs/q1-clear-deepsets-hilar-10seed-20260903/ecg_datasets-resolved-v2/CSN_QRS/data"
RUN="${ROOT}/results/${CAMPAIGN}/csn-seed${SEED}"
test -s "${ROOT}/results/${CAMPAIGN}/pretest-gate.json"
test -s "${RUN}/train-val-complete.json"
if [[ -s "${RUN}/formal-test/complete.json" ]]; then
  echo "formal test already complete: ${RUN}"
  exit 0
fi
source "${ROOT}/mvp/activate_mvp.sh"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
mkdir -p "${RUN}/formal-test"
if [[ ! -s "${RUN}/features/test/feature-manifest.json" ]]; then
  "${ROOT}/.venv/bin/python" "${CODE}/cache_low_label_features.py" \
    --root "${ROOT}" --dataset "${INPUT}" --checkpoint "${RUN}/hila/checkpoint-best.pth" \
    --output "${RUN}/features" --classes 38 --seed "${SEED}" --splits test
fi
if [[ ! -s "${RUN}/formal-test/paired/formal-test-result.json" ]]; then
  "${ROOT}/.venv/bin/python" "${ROOT}/mvp/evaluate_direct_residual_formal.py" \
    --features "${RUN}/features/test" \
    --classifier "${RUN}/hilar-training/parameter-matched-direct/checkpoint-best.pth" \
    --output "${RUN}/formal-test/paired" --classes 38 --task csn --seed "${SEED}"
fi
if [[ ! -s "${RUN}/formal-test/hug/formal-test-result.json" ]]; then
  "${ROOT}/.venv/bin/python" "${ROOT}/src/CLEAR-HUG/experiments/q1_clear_deepsets_hilar_10seed_20260903/evaluate_hug_formal.py" \
    --root "${ROOT}" --dataset "${INPUT}" --checkpoint "${RUN}/clear-hug/checkpoint-best.pth" \
    --released-checkpoint "${ROOT}/checkpoints/released_ckpt.pth" \
    --strict-labels "${RUN}/features/test/labels.npy" --output "${RUN}/formal-test/hug" \
    --task csn --seed "${SEED}" --classes 38
fi
"${ROOT}/.venv/bin/python" - "${RUN}" "${TRAIN_FRACTION}" <<'PY'
import json, os, sys
from pathlib import Path
r=Path(sys.argv[1]); fraction=float(sys.argv[2]); paired=json.loads((r/'formal-test/paired/formal-test-result.json').read_text()); hug=json.loads((r/'formal-test/hug/formal-test-result.json').read_text())
payload={'status':'complete','task':'csn','seed':paired['seed'],'train_fraction':fraction,
 'test_used_for_selection':False,'hug':hug['metrics'],'hila':paired['baseline'],'hilar':paired['direct']}
q=r/'formal-test/complete.json.incoming'; q.write_text(json.dumps(payload,indent=2)+'\n'); os.replace(q,r/'formal-test/complete.json')
PY
