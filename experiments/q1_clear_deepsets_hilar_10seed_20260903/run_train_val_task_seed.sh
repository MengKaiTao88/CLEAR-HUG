#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLEAR_HUG_ROOT:?CLEAR_HUG_ROOT is required}"
TASK="${1:?task required}"
SEED="${2:?seed required}"
CAMPAIGN="q1-clear-deepsets-hilar-10seed-20260903"
CODE_COMMIT="${CAMPAIGN_CODE_COMMIT:?CAMPAIGN_CODE_COMMIT is required}"
INPUT_ROOT="${CAMPAIGN_INPUT_ROOT:-${ROOT}/campaign-inputs/${CAMPAIGN}/ecg_datasets}"

case "${SEED}" in 42|43|44|45|46|47|48|49|50|51) ;; *) echo "unsupported seed ${SEED}" >&2; exit 2 ;; esac
case "${TASK}" in
  superdiagnostic) CLASSES=5; DATASET=PTBXL_QRS/superdiagnostic; TASK_ID=1 ;;
  subdiagnostic) CLASSES=23; DATASET=PTBXL_QRS/subdiagnostic; TASK_ID=2 ;;
  form) CLASSES=19; DATASET=PTBXL_QRS/form; TASK_ID=3 ;;
  rhythm) CLASSES=12; DATASET=PTBXL_QRS/rhythm; TASK_ID=4 ;;
  cpsc2018) CLASSES=9; DATASET=CPSC2018_QRS/data; TASK_ID=5 ;;
  csn) CLASSES=38; DATASET=CSN_QRS/data; TASK_ID=6 ;;
  *) echo "unsupported task ${TASK}" >&2; exit 2 ;;
esac

source "${ROOT}/mvp/activate_mvp.sh"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
RUN="${ROOT}/results/${CAMPAIGN}/${TASK}-seed${SEED}"
QRS="${INPUT_ROOT}/${DATASET}"
HUG="${RUN}/clear-hug"
DEEPSETS="${RUN}/deepsets"
FEATURES="${RUN}/features"
HILAR_TRAIN="${RUN}/hilar-training"
HILAR="${HILAR_TRAIN}/parameter-matched-direct"
STATUS="${RUN}/status.json"
LOG="${RUN}/train-val.log"

if [[ -e "${RUN}" ]]; then
  if [[ -s "${RUN}/train-val-complete.json" ]]; then exit 0; fi
  echo "refusing partial run directory ${RUN}" >&2
  exit 20
fi
mkdir -p "${RUN}"
exec > >(tee -a "${LOG}") 2>&1

write_status() {
  python - "${STATUS}" "$1" "${2:-}" "${TASK}" "${SEED}" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1]); q=p.with_suffix(".json.incoming")
q.write_text(json.dumps({"state":sys.argv[2],"stage":sys.argv[3] or None,"task":sys.argv[4],"seed":int(sys.argv[5]),"updated_at":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
os.replace(q,p)
PY
}
stage=initializing
trap 'code=$?; if [[ $code -ne 0 ]]; then write_status failed "${stage}:exit=${code}"; fi' EXIT
write_status running "${stage}"

test -d "${QRS}"
for split in train val test; do
  test -s "${QRS}/${split}_data.npy"
  test -s "${QRS}/${split}_labels.npy"
done
python - "${ROOT}/checkpoints/released_ckpt.pth" <<'PY'
import hashlib,sys
p=sys.argv[1]; h=hashlib.sha256(open(p,'rb').read()).hexdigest()
expected="15e456964c5f819aa882522a203946b515cc1034e217dfcbc474e5e50bf1719a"
assert h == expected, (h, expected)
PY
python - <<'PY'
import torch
assert torch.cuda.is_available()
print("GPU", torch.cuda.get_device_name(0), flush=True)
PY

stage=hug-training
write_status running "${stage}"
mkdir -p "${HUG}-tb"
cd "${ROOT}/src/CLEAR-HUG"
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port="$((21000 + SEED * 10 + TASK_ID))" run_class_finetuning.py \
  --dataset_dir "${QRS}" --output_dir "${HUG}" --log_dir "${HUG}-tb" \
  --model CLEAR_HUG_finetune_base --trainable moe --split_ratio 1.0 \
  --finetune "${ROOT}/checkpoints/released_ckpt.pth" --sampling_method random \
  --weight_decay 0.05 --batch_size 256 --lr 5e-3 --update_freq 1 \
  --warmup_epochs 10 --epochs 100 --layer_decay 0.9 --save_ckpt_freq 100 \
  --seed "${SEED}" --is_binary --nb_classes "${CLASSES}" --world_size 1 \
  --atten_mask --cls_token_num 12 --mask_ratio 0 --num_workers 10 \
  --screen_only --screen_select_metric roc_auc --screen_patience 0
test -s "${HUG}/checkpoint-best.pth"
python - "${HUG}" "${TASK}" "${SEED}" <<'PY'
import json,os,sys
from pathlib import Path
out,task,seed=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3])
rows=[json.loads(x) for x in (out/'log.txt').read_text().splitlines() if x.strip()]
assert len(rows)==100, len(rows)
best=max(rows,key=lambda r:float(r['val_roc_auc']))
p={"task":task,"seed":seed,"epochs":100,"selection_split":"validation","selection_metric":"macro_auroc","best_epoch":int(best['epoch']),"best_validation_auroc":float(best['val_roc_auc']),"best_validation_auprc":float(best['val_pr_auc']),"test_used_for_selection":False}
q=out/'training-complete.json.incoming'; q.write_text(json.dumps(p,indent=2)+'\n'); os.replace(q,out/'training-complete.json')
PY

stage=deepsets-training
write_status running "${stage}"
bash "${ROOT}/mvp/run_deepsets_dev_baseline.sh" "${TASK}" "${QRS}" "${DEEPSETS}" "${SEED}" "$((31000 + SEED * 10 + TASK_ID))" "${CLASSES}"
test -s "${DEEPSETS}/checkpoint-best.pth"

stage=cache-train-val
write_status running "${stage}"
python "${ROOT}/mvp/cache_frozen_deepsets_features.py" --root "${ROOT}" --dataset "${QRS}" \
  --checkpoint "${DEEPSETS}/checkpoint-best.pth" --output "${FEATURES}" \
  --classes "${CLASSES}" --seed "${SEED}" --splits train val

stage=hilar-training
write_status running "${stage}"
python "${ROOT}/mvp/train_deepsets_anchored_residual.py" --features "${FEATURES}" \
  --output "${HILAR_TRAIN}" --task "${TASK}" --classes "${CLASSES}" --seed "${SEED}" \
  --selection-metric auroc --variants parameter-matched-direct
test -s "${HILAR}/checkpoint-best.pth"

stage=pretest-audit
write_status running "${stage}"
python - "${RUN}" "${TASK}" "${SEED}" "${CLASSES}" "${CODE_COMMIT}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
run,task,seed,classes,commit=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
hug=run/'clear-hug/checkpoint-best.pth'; deep=run/'deepsets/checkpoint-best.pth'; hilar=run/'hilar-training/parameter-matched-direct/checkpoint-best.pth'
feature=json.loads((run/'features/feature-manifest.json').read_text())
assert Path(feature['checkpoint'])==deep and int(feature['checkpoint_seed'])==seed
residual=json.loads((run/'hilar-training/parameter-matched-direct/complete.json').read_text())
assert residual['selection_metric']=='auroc' and float(residual['anchor'])==0.0
p={"schema_version":1,"status":"complete","task":task,"seed":seed,"classes":classes,"code_commit":commit,"selection_metric":"macro_auroc","pairing":"DeepSets_s -> frozen logits_s/features_s -> zero-init no-anchor HiLAR_s","anchor_lambda":0.0,"test_used_for_selection":False,"hug_checkpoint":str(hug),"hug_checkpoint_sha256":sha(hug),"deepsets_checkpoint":str(deep),"deepsets_checkpoint_sha256":sha(deep),"hilar_checkpoint":str(hilar),"hilar_checkpoint_sha256":sha(hilar),"feature_manifest":str(run/'features/feature-manifest.json'),"feature_checkpoint_exact":True}
q=run/'train-val-complete.json.incoming'; q.write_text(json.dumps(p,indent=2)+'\n'); os.replace(q,run/'train-val-complete.json')
PY
stage=complete
write_status complete "${stage}"
echo "TRAIN_VAL_COMPLETE task=${TASK} seed=${SEED} time=$(date -Is)"
