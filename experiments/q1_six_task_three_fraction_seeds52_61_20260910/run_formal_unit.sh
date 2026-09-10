#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:?}"; FKEY="${1:?}"; TASK="${2:?}"; SEED="${3:?}"
CAMPAIGN=q1-six-task-three-fraction-seeds52-61-20260910; CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_six_task_three_fraction_seeds52_61_20260910"
INPUT_ROOT="${ROOT}/campaign-inputs/q1-clear-deepsets-hilar-10seed-20260903/ecg_datasets-resolved-v2"
case "${FKEY}" in 100pct) FRACTION=1.0 ;; 10pct) FRACTION=0.10 ;; 1pct) FRACTION=0.01 ;; *) exit 2 ;; esac
case "${SEED}" in 52|53|54|55|56|57|58|59|60|61) ;; *) exit 2 ;; esac
case "${TASK}" in
 superdiagnostic) CLASSES=5; DATASET=PTBXL_QRS/superdiagnostic ;;
 subdiagnostic) CLASSES=23; DATASET=PTBXL_QRS/subdiagnostic ;;
 form) CLASSES=19; DATASET=PTBXL_QRS/form ;;
 rhythm) CLASSES=12; DATASET=PTBXL_QRS/rhythm ;;
 cpsc2018) CLASSES=9; DATASET=CPSC2018_QRS/data ;;
 csn) CLASSES=38; DATASET=CSN_QRS/data ;;
 *) exit 2 ;;
esac
source "${ROOT}/mvp/activate_mvp.sh"; export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
RUN="${ROOT}/results/${CAMPAIGN}/${FKEY}/${TASK}-seed${SEED}"; GATE="${ROOT}/results/${CAMPAIGN}/global-pretest-gate.json"; QRS="${INPUT_ROOT}/${DATASET}"
test -s "${RUN}/train-val-complete.json"; test -s "${GATE}"
python - "${GATE}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p['status']=='passed' and p['formal_test_authorized'] is True and p['train_val_units']==180
PY
[[ -s "${RUN}/formal-test/complete.json" ]] && exit 0
mkdir -p "${RUN}/formal-test"
if [[ ! -s "${RUN}/features/test/labels.npy" ]]; then
 python "${ROOT}/src/CLEAR-HUG/experiments/q1_six_task_low_label_10seed_20260909/cache_low_label_features.py" --root "${ROOT}" --dataset "${QRS}" --checkpoint "${RUN}/hila/checkpoint-best.pth" --output "${RUN}/features" --classes "${CLASSES}" --seed "${SEED}" --splits test --train-split-ratio "${FRACTION}" --sampling-method random
fi
if [[ ! -s "${RUN}/formal-test/paired/formal-test-result.json" ]]; then
 python "${ROOT}/mvp/evaluate_direct_residual_formal.py" --features "${RUN}/features/test" --classifier "${RUN}/hilar-training/parameter-matched-direct/checkpoint-best.pth" --output "${RUN}/formal-test/paired" --classes "${CLASSES}" --task "${TASK}" --seed "${SEED}"
fi
if [[ ! -s "${RUN}/formal-test/hug/formal-test-result.json" ]]; then
 python "${ROOT}/src/CLEAR-HUG/experiments/q1_clear_deepsets_hilar_10seed_20260903/evaluate_hug_formal.py" --root "${ROOT}" --dataset "${QRS}" --checkpoint "${RUN}/clear-hug/checkpoint-best.pth" --released-checkpoint "${ROOT}/checkpoints/released_ckpt.pth" --strict-labels "${RUN}/features/test/labels.npy" --output "${RUN}/formal-test/hug" --task "${TASK}" --seed "${SEED}" --classes "${CLASSES}"
fi
python - "${RUN}" "${FKEY}" "${FRACTION}" <<'PY'
import json,os,sys
from pathlib import Path
r,fkey,fraction=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]); paired=json.load(open(r/'formal-test/paired/formal-test-result.json')); hug=json.load(open(r/'formal-test/hug/formal-test-result.json'))
p={'status':'complete','fraction_key':fkey,'train_fraction':fraction,'task':paired['task'],'seed':paired['seed'],'test_used_for_selection':False,'hug':hug['metrics'],'hila':paired['baseline'],'hilar':paired['direct']}
q=r/'formal-test/complete.json.incoming';q.write_text(json.dumps(p,indent=2)+'\n');os.replace(q,r/'formal-test/complete.json')
PY
