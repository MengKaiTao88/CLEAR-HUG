#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:?CLEAR_HUG_ROOT is required}"
TASK="${1:?task required}"; SEED="${2:?seed required}"
CAMPAIGN="q1-clear-deepsets-hilar-10seed-20260903"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_ROOT="${CAMPAIGN_INPUT_ROOT:-${ROOT}/campaign-inputs/${CAMPAIGN}/ecg_datasets}"
case "${SEED}" in 42|43|44|45|46|47|48|49|50|51) ;; *) exit 2 ;; esac
case "${TASK}" in
 superdiagnostic) CLASSES=5; DATASET=PTBXL_QRS/superdiagnostic ;;
 subdiagnostic) CLASSES=23; DATASET=PTBXL_QRS/subdiagnostic ;;
 form) CLASSES=19; DATASET=PTBXL_QRS/form ;;
 rhythm) CLASSES=12; DATASET=PTBXL_QRS/rhythm ;;
 cpsc2018) CLASSES=9; DATASET=CPSC2018_QRS/data ;;
 csn) CLASSES=38; DATASET=CSN_QRS/data ;;
 *) exit 2 ;;
esac
source "${ROOT}/mvp/activate_mvp.sh"
RUN="${ROOT}/results/${CAMPAIGN}/${TASK}-seed${SEED}"
GATE="${ROOT}/results/${CAMPAIGN}/pretest-gate.json"
test -s "${RUN}/train-val-complete.json"; test -s "${GATE}"
python - "${GATE}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p['status']=='passed' and p['formal_test_authorized'] is True and p['train_val_units']==60
PY
if [[ -e "${RUN}/formal-test" ]]; then echo "refusing existing formal output" >&2; exit 20; fi
python "${ROOT}/mvp/cache_frozen_deepsets_features.py" --root "${ROOT}" \
 --dataset "${INPUT_ROOT}/${DATASET}" --checkpoint "${RUN}/deepsets/checkpoint-best.pth" \
 --output "${RUN}/features" --classes "${CLASSES}" --seed "${SEED}" --splits test
mkdir -p "${RUN}/formal-test"
python "${ROOT}/mvp/evaluate_direct_residual_formal.py" --features "${RUN}/features/test" \
 --classifier "${RUN}/hilar-training/parameter-matched-direct/checkpoint-best.pth" \
 --output "${RUN}/formal-test/paired" --classes "${CLASSES}" --task "${TASK}" --seed "${SEED}"
python "${CODE_DIR}/evaluate_hug_formal.py" --root "${ROOT}" --dataset "${INPUT_ROOT}/${DATASET}" \
 --checkpoint "${RUN}/clear-hug/checkpoint-best.pth" --released-checkpoint "${ROOT}/checkpoints/released_ckpt.pth" \
 --strict-labels "${RUN}/features/test/labels.npy" --output "${RUN}/formal-test/hug" \
 --task "${TASK}" --seed "${SEED}" --classes "${CLASSES}"
python - "${RUN}" <<'PY'
import json,os,sys
from pathlib import Path
r=Path(sys.argv[1]); paired=json.load(open(r/'formal-test/paired/formal-test-result.json')); hug=json.load(open(r/'formal-test/hug/formal-test-result.json'))
p={"status":"complete","task":paired['task'],"seed":paired['seed'],"test_used_for_selection":False,"hug":hug['metrics'],"deepsets":paired['baseline'],"hilar":paired['direct']}
q=r/'formal-test/complete.json.incoming'; q.write_text(json.dumps(p,indent=2)+'\n'); os.replace(q,r/'formal-test/complete.json')
PY
echo "FORMAL_COMPLETE task=${TASK} seed=${SEED} time=$(date -Is)"
