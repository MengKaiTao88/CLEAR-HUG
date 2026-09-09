#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:?CLEAR_HUG_ROOT required}"
source "${ROOT}/mvp/activate_mvp.sh"
FRACTION_KEY="${1:?fraction key required}"
TASK="${2:?task required}"
SEED="${3:?seed required}"
CAMPAIGN=q1-six-task-low-label-10seed-20260909
CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_six_task_low_label_10seed_20260909"
INPUT_ROOT="${ROOT}/campaign-inputs/q1-clear-deepsets-hilar-10seed-20260903/ecg_datasets-resolved-v2"
case "${FRACTION_KEY}" in 1pct) FRACTION=0.01 ;; 10pct) FRACTION=0.10 ;; *) exit 2 ;; esac
case "${SEED}" in 42|43|44|45|46|47|48|49|50|51) ;; *) exit 2 ;; esac
case "${TASK}" in
 superdiagnostic) CLASSES=5; DATASET=PTBXL_QRS/superdiagnostic; TOTAL=17084; ID=1 ;;
 subdiagnostic) CLASSES=23; DATASET=PTBXL_QRS/subdiagnostic; TOTAL=17084; ID=2 ;;
 form) CLASSES=19; DATASET=PTBXL_QRS/form; TOTAL=7197; ID=3 ;;
 rhythm) CLASSES=12; DATASET=PTBXL_QRS/rhythm; TOTAL=16832; ID=4 ;;
 cpsc2018) CLASSES=9; DATASET=CPSC2018_QRS/data; TOTAL=4950; ID=5 ;;
 csn) CLASSES=38; DATASET=CSN_QRS/data; TOTAL=16546; ID=6 ;;
 *) exit 2 ;;
esac
case "${FRACTION_KEY}" in
 1pct) EXPECTED=$((TOTAL / 100)) ;;
 10pct) EXPECTED=$((TOTAL / 10)) ;;
esac
RUN="${ROOT}/results/${CAMPAIGN}/${FRACTION_KEY}/${TASK}-seed${SEED}"
QRS="${INPUT_ROOT}/${DATASET}"; HUG="${RUN}/clear-hug"; HILA="${RUN}/hila"
FEATURES="${RUN}/features"; HILAR_ROOT="${RUN}/hilar-training"; HILAR="${HILAR_ROOT}/parameter-matched-direct"
STATUS="${RUN}/status.json"; LOG="${RUN}/train-val.log"
if [[ -e "${RUN}" ]]; then
 [[ -s "${RUN}/train-val-complete.json" ]] && exit 0
 if [[ "${ALLOW_SAFE_RESUME:-0}" != 1 ]]; then
  echo "refusing partial unit ${RUN}" >&2; exit 20
 fi
 test -s "${RUN}/status.json"
 test -s "${RUN}/subset-manifest.json"
 python - "${RUN}/status.json" "${RUN}/subset-manifest.json" "${FRACTION_KEY}" "${TASK}" "${SEED}" "${FRACTION}" "${EXPECTED}" <<'PY'
import json,sys
status=json.load(open(sys.argv[1])); subset=json.load(open(sys.argv[2]))
assert status['fraction']==sys.argv[3] and status['task']==sys.argv[4] and status['seed']==int(sys.argv[5])
assert subset['seed']==int(sys.argv[5]) and subset['fraction']==float(sys.argv[6]) and subset['selected_records']==int(sys.argv[7])
PY
fi
mkdir -p "${RUN}"; exec > >(tee -a "${LOG}") 2>&1
write_status(){ python - "${STATUS}" "$1" "${2:-}" "${FRACTION_KEY}" "${TASK}" "${SEED}" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]); q=p.with_suffix('.json.incoming')
q.write_text(json.dumps({'state':sys.argv[2],'stage':sys.argv[3] or None,'fraction':sys.argv[4],'task':sys.argv[5],'seed':int(sys.argv[6]),'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n'); os.replace(q,p)
PY
}
stage=initializing; trap 'c=$?; [[ $c -eq 0 ]] || write_status failed "${stage}:exit=${c}"' EXIT
write_status running "${stage}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
for split in train val test; do test -s "${QRS}/${split}_data.npy"; test -s "${QRS}/${split}_labels.npy"; done
python - "${QRS}" "${SEED}" "${FRACTION}" "${RUN}/subset-manifest.json" <<'PY'
import hashlib,json,random,sys
from pathlib import Path
import numpy as np
root,seed,fraction,out=Path(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),Path(sys.argv[4])
y=np.load(root/'train_labels.npy',mmap_mode='r'); idx=np.asarray(random.Random(seed).sample(range(len(y)),int(len(y)*fraction)),dtype=np.int64)
p={'seed':seed,'fraction':fraction,'source_records':len(y),'selected_records':len(idx),'indices_sha256':hashlib.sha256(idx.tobytes()).hexdigest(),'selected_labels_sha256':hashlib.sha256(np.asarray(y[idx],dtype=np.float32).tobytes()).hexdigest()}
out.write_text(json.dumps(p,indent=2)+'\n')
PY
stage=hug-training; write_status running "${stage}"; mkdir -p "${HUG}-tb"; cd "${ROOT}/src/CLEAR-HUG"
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port="$((21000+SEED*10+ID))" run_class_finetuning.py \
 --dataset_dir "${QRS}" --output_dir "${HUG}" --log_dir "${HUG}-tb" --model CLEAR_HUG_finetune_base --trainable moe --split_ratio "${FRACTION}" \
 --finetune "${ROOT}/checkpoints/released_ckpt.pth" --sampling_method random --weight_decay 0.05 --batch_size 256 --lr 5e-3 --update_freq 1 \
 --warmup_epochs 10 --epochs 100 --layer_decay 0.9 --save_ckpt_freq 100 --seed "${SEED}" --is_binary --nb_classes "${CLASSES}" --world_size 1 \
 --atten_mask --cls_token_num 12 --mask_ratio 0 --num_workers 10 --screen_only --screen_select_metric roc_auc --screen_patience 0
test -s "${HUG}/checkpoint-best.pth"
python - "${HUG}" "${TASK}" "${SEED}" "${FRACTION}" <<'PY'
import json,os,sys
from pathlib import Path
out,task,seed,fraction=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),float(sys.argv[4]); rows=[json.loads(x) for x in (out/'log.txt').read_text().splitlines() if x.strip()]; assert len(rows)==100
b=max(rows,key=lambda x:float(x['val_roc_auc'])); p={'task':task,'seed':seed,'epochs':100,'train_fraction':fraction,'selection_metric':'macro_auroc','best_epoch':int(b['epoch']),'best_validation_auroc':float(b['val_roc_auc']),'best_validation_auprc':float(b['val_pr_auc']),'test_used_for_selection':False}
q=out/'training-complete.json.incoming'; q.write_text(json.dumps(p,indent=2)+'\n'); os.replace(q,out/'training-complete.json')
PY
stage=hila-training; write_status running "${stage}"; mkdir -p "${HILA}-tb"; cd "${ROOT}/src/CLEAR-HUG"
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port="$((31000+SEED*10+ID))" "${ROOT}/mvp/run_fair_baseline_finetuning.py" \
 --dataset_dir "${QRS}" --output_dir "${HILA}" --log_dir "${HILA}-tb" --model CLEAR_MASKED_DEEPSETS_finetune_base \
 --finetune "${ROOT}/checkpoints/released_ckpt.pth" --trainable adapter --split_ratio "${FRACTION}" --sampling_method random --weight_decay 0.05 \
 --batch_size 256 --lr 5e-3 --update_freq 1 --warmup_epochs 10 --epochs 100 --layer_decay 0.9 --save_ckpt_freq 100 --seed "${SEED}" \
 --is_binary --nb_classes "${CLASSES}" --world_size 1 --atten_mask --cls_token_num 12 --mask_ratio 0 --num_workers 10 --screen_only --screen_select_metric roc_auc --screen_patience 0
test -s "${HILA}/checkpoint-best.pth"
python - "${HILA}" "${TASK}" "${SEED}" "${FRACTION}" <<'PY'
import json,os,sys
from pathlib import Path
out,task,seed,fraction=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),float(sys.argv[4]); rows=[json.loads(x) for x in (out/'log.txt').read_text().splitlines() if x.strip()]; assert len(rows)==100
b=max(rows,key=lambda x:float(x['val_roc_auc'])); p={'task':task,'seed':seed,'epochs':100,'train_fraction':fraction,'selection_metric':'macro_auroc','best_epoch':int(b['epoch']),'best_validation_auroc':float(b['val_roc_auc']),'best_validation_auprc':float(b['val_pr_auc']),'test_used_for_selection':False}
q=out/'baseline-complete.json.incoming'; q.write_text(json.dumps(p,indent=2)+'\n'); os.replace(q,out/'baseline-complete.json')
PY
stage=cache-train-val; write_status running "${stage}"
python "${CODE}/cache_low_label_features.py" --root "${ROOT}" --dataset "${QRS}" --checkpoint "${HILA}/checkpoint-best.pth" --output "${FEATURES}" \
 --classes "${CLASSES}" --seed "${SEED}" --splits train val --train-split-ratio "${FRACTION}" --sampling-method random
stage=hilar-training; write_status running "${stage}"
python "${ROOT}/mvp/train_deepsets_anchored_residual.py" --features "${FEATURES}" --output "${HILAR_ROOT}" --task "${TASK}" --classes "${CLASSES}" \
 --seed "${SEED}" --selection-metric auroc --variants parameter-matched-direct
stage=audit; write_status running "${stage}"
python - "${RUN}" "${FRACTION_KEY}" "${TASK}" "${SEED}" "${CLASSES}" "${FRACTION}" "${EXPECTED}" "${CAMPAIGN_CODE_COMMIT:?}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
run,fkey,task,seed,classes,fraction,expected,commit=Path(sys.argv[1]),sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6]),int(sys.argv[7]),sys.argv[8]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
hug=run/'clear-hug/checkpoint-best.pth'; hila=run/'hila/checkpoint-best.pth'; hilar=run/'hilar-training/parameter-matched-direct/checkpoint-best.pth'; feature=json.loads((run/'features/feature-manifest.json').read_text()); residual=json.loads((run/'hilar-training/parameter-matched-direct/complete.json').read_text()); subset=json.loads((run/'subset-manifest.json').read_text())
assert feature['checkpoint']==str(hila) and feature['checkpoint_seed']==seed and feature['train_split_ratio']==fraction
assert feature['splits']['train']['records']==expected and subset['selected_records']==expected
assert residual['selection_metric']=='auroc' and float(residual['anchor'])==0.0
p={'schema_version':1,'status':'complete','fraction_key':fkey,'train_fraction':fraction,'train_records':expected,'task':task,'seed':seed,'classes':classes,'code_commit':commit,'selection_metric':'macro_auroc','test_used_for_selection':False,'pairing':'HILA_s -> frozen logits_s/features_s -> zero-init no-anchor HiLAR_s','hug_checkpoint':str(hug),'hug_sha256':sha(hug),'hila_checkpoint':str(hila),'hila_sha256':sha(hila),'hilar_checkpoint':str(hilar),'hilar_sha256':sha(hilar),'feature_checkpoint_exact':True,'subset_manifest':str(run/'subset-manifest.json')}
q=run/'train-val-complete.json.incoming'; q.write_text(json.dumps(p,indent=2)+'\n'); os.replace(q,run/'train-val-complete.json')
PY
stage=complete; write_status complete "${stage}"
