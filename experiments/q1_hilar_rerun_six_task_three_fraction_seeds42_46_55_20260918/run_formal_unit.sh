#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:?}"; FKEY="${1:?}"; TASK="${2:?}"; SEED="${3:?}"
CAMPAIGN=q1-hilar-rerun-six-task-three-fraction-seeds42-46-55-20260918
CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_hilar_rerun_six_task_three_fraction_seeds42_46_55_20260918"
INPUT_ROOT="${ROOT}/campaign-inputs/q1-clear-deepsets-hilar-10seed-20260903/ecg_datasets-resolved-v2"
case "${FKEY}" in 1pct) FRACTION=0.01 ;; 10pct) FRACTION=0.10 ;; 100pct) FRACTION=1.0 ;; *) exit 2 ;; esac
case "${SEED}" in 42|46|55) ;; *) exit 2 ;; esac
case "${TASK}" in
 superdiagnostic) CLASSES=5; DATASET=PTBXL_QRS/superdiagnostic ;; subdiagnostic) CLASSES=23; DATASET=PTBXL_QRS/subdiagnostic ;;
 form) CLASSES=19; DATASET=PTBXL_QRS/form ;; rhythm) CLASSES=12; DATASET=PTBXL_QRS/rhythm ;;
 cpsc2018) CLASSES=9; DATASET=CPSC2018_QRS/data ;; csn) CLASSES=38; DATASET=CSN_QRS/data ;; *) exit 2 ;;
esac
RUN="${ROOT}/results/${CAMPAIGN}/${FKEY}/${TASK}-seed${SEED}"; GATE="${ROOT}/results/${CAMPAIGN}/global-pretest-gate.json"; QRS="${INPUT_ROOT}/${DATASET}"
test -s "${RUN}/train-val-complete.json"; test -s "${GATE}"
python - "${GATE}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));assert p['status']=='passed' and p['formal_test_authorized'] is True and p['train_val_units']==54
PY
[[ -s "${RUN}/formal-test/complete.json" ]] && exit 0
source "${ROOT}/mvp/activate_mvp.sh"; export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=1
if [[ ! -s "${RUN}/features/test/labels.npy" ]]; then
 python "${CODE}/cache_paired_features.py" --root "${ROOT}" --dataset "${QRS}" --checkpoint "${RUN}/hila/checkpoint-best.pth" \
  --output "${RUN}/features" --classes "${CLASSES}" --seed "${SEED}" --splits test --train-split-ratio "${FRACTION}" --sampling-method random
fi
if [[ -e "${RUN}/formal-test/paired" && ! -s "${RUN}/formal-test/paired/formal-test-result.json" ]]; then echo 'refusing partial formal output' >&2; exit 20; fi
if [[ ! -s "${RUN}/formal-test/paired/formal-test-result.json" ]]; then
 python "${ROOT}/mvp/evaluate_direct_residual_formal.py" --features "${RUN}/features/test" \
  --classifier "${RUN}/hilar-training/parameter-matched-direct/checkpoint-best.pth" --output "${RUN}/formal-test/paired" \
  --classes "${CLASSES}" --task "${TASK}" --seed "${SEED}"
fi
mkdir -p "${RUN}/formal-test"
python - "${RUN}" "${FKEY}" "${FRACTION}" <<'PY'
import json,os,sys
from pathlib import Path
r,fkey,fraction=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]); paired=json.load(open(r/'formal-test/paired/formal-test-result.json'))
p={'status':'complete','fraction_key':fkey,'train_fraction':fraction,'task':paired['task'],'seed':paired['seed'],'test_used_for_selection':False,'hila':paired['baseline'],'hilar':paired['direct'],'delta_auroc_pp':paired['delta_auroc_pp'],'delta_auprc_pp':paired['delta_auprc_pp']}
q=r/'formal-test/complete.json.incoming';q.write_text(json.dumps(p,indent=2)+'\n');os.replace(q,r/'formal-test/complete.json')
PY
