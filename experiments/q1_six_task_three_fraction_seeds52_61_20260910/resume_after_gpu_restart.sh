#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:?CLEAR_HUG_ROOT required}"
FKEY="${1:?fraction required}"; TASK="${2:?task required}"; SEED="${3:?seed required}"
START="${4:?start stage must be hila or cache}"
CAMPAIGN=q1-six-task-three-fraction-seeds52-61-20260910
CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_six_task_three_fraction_seeds52_61_20260910"
INPUT_ROOT="${ROOT}/campaign-inputs/q1-clear-deepsets-hilar-10seed-20260903/ecg_datasets-resolved-v2"
case "${FKEY}" in 100pct) FRACTION=1.0 ;; 10pct) FRACTION=0.10 ;; 1pct) FRACTION=0.01 ;; *) exit 2 ;; esac
case "${TASK}" in
 superdiagnostic) CLASSES=5; DATASET=PTBXL_QRS/superdiagnostic; TOTAL=17084; ID=1 ;;
 subdiagnostic) CLASSES=23; DATASET=PTBXL_QRS/subdiagnostic; TOTAL=17084; ID=2 ;;
 form) CLASSES=19; DATASET=PTBXL_QRS/form; TOTAL=7197; ID=3 ;;
 rhythm) CLASSES=12; DATASET=PTBXL_QRS/rhythm; TOTAL=16832; ID=4 ;;
 cpsc2018) CLASSES=9; DATASET=CPSC2018_QRS/data; TOTAL=4950; ID=5 ;;
 csn) CLASSES=38; DATASET=CSN_QRS/data; TOTAL=16546; ID=6 ;;
 *) exit 2 ;;
esac
case "${FKEY}" in 100pct) EXPECTED=${TOTAL} ;; 10pct) EXPECTED=$((TOTAL/10)) ;; 1pct) EXPECTED=$((TOTAL/100)) ;; esac
RUN="${ROOT}/results/${CAMPAIGN}/${FKEY}/${TASK}-seed${SEED}"
QRS="${INPUT_ROOT}/${DATASET}"; HILA="${RUN}/hila"; FEATURES="${RUN}/features"
HILAR_ROOT="${RUN}/hilar-training"; STATUS="${RUN}/status.json"
source "${ROOT}/mvp/activate_mvp.sh"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
test -s "${RUN}/clear-hug/training-complete.json"
test -s "${RUN}/clear-hug/checkpoint-best.pth"
test -s "${RUN}/subset-manifest.json"
cp -n "${STATUS}" "${RUN}/failure-status-gpu-restart-20260911.json"
write_status(){ python - "${STATUS}" "$1" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); q=p.with_suffix('.json.incoming')
x=json.loads(p.read_text()) if p.exists() else {}
x.update({'state':'running','stage':sys.argv[2]})
q.write_text(json.dumps(x,indent=2)+'\n'); os.replace(q,p)
PY
}
if [[ "${START}" == hila ]]; then
 test ! -s "${HILA}/baseline-complete.json"
 write_status hila-training
 mkdir -p "${HILA}-tb"; cd "${ROOT}/src/CLEAR-HUG"
 python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port="$((31000+SEED*10+ID))" "${ROOT}/mvp/run_fair_baseline_finetuning.py" \
  --dataset_dir "${QRS}" --output_dir "${HILA}" --log_dir "${HILA}-tb" --model CLEAR_MASKED_DEEPSETS_finetune_base \
  --finetune "${ROOT}/checkpoints/released_ckpt.pth" --trainable adapter --split_ratio "${FRACTION}" --sampling_method random --weight_decay 0.05 \
  --batch_size 256 --lr 5e-3 --update_freq 1 --warmup_epochs 10 --epochs 100 --layer_decay 0.9 --save_ckpt_freq 100 --seed "${SEED}" \
  --is_binary --nb_classes "${CLASSES}" --world_size 1 --atten_mask --cls_token_num 12 --mask_ratio 0 --num_workers 10 --screen_only --screen_select_metric roc_auc --screen_patience 0
 python - "${HILA}" "${TASK}" "${SEED}" "${FRACTION}" <<'PY'
import json,os,sys
from pathlib import Path
out,task,seed,fraction=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),float(sys.argv[4]); rows=[json.loads(x) for x in (out/'log.txt').read_text().splitlines() if x.strip()]; assert len(rows)==100
b=max(rows,key=lambda x:float(x['val_roc_auc'])); p={'task':task,'seed':seed,'epochs':100,'train_fraction':fraction,'selection_metric':'macro_auroc','best_epoch':int(b['epoch']),'best_validation_auroc':float(b['val_roc_auc']),'best_validation_auprc':float(b['val_pr_auc']),'test_used_for_selection':False}
q=out/'baseline-complete.json.incoming'; q.write_text(json.dumps(p,indent=2)+'\n'); os.replace(q,out/'baseline-complete.json')
PY
elif [[ "${START}" != cache ]]; then exit 2; fi
test -s "${HILA}/baseline-complete.json"; test -s "${HILA}/checkpoint-best.pth"
write_status cache-train-val
python "${ROOT}/src/CLEAR-HUG/experiments/q1_six_task_low_label_10seed_20260909/cache_low_label_features.py" --root "${ROOT}" --dataset "${QRS}" --checkpoint "${HILA}/checkpoint-best.pth" --output "${FEATURES}" --classes "${CLASSES}" --seed "${SEED}" --splits train val --train-split-ratio "${FRACTION}" --sampling-method random
write_status hilar-training
python "${ROOT}/mvp/train_deepsets_anchored_residual.py" --features "${FEATURES}" --output "${HILAR_ROOT}" --task "${TASK}" --classes "${CLASSES}" --seed "${SEED}" --selection-metric auroc --variants parameter-matched-direct
python - "${RUN}" "${FKEY}" "${TASK}" "${SEED}" "${CLASSES}" "${FRACTION}" "${EXPECTED}" "${CAMPAIGN_CODE_COMMIT:?}" <<'PY'
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
p={'schema_version':1,'status':'complete','fraction_key':fkey,'train_fraction':fraction,'train_records':expected,'task':task,'seed':seed,'classes':classes,'code_commit':commit,'selection_metric':'macro_auroc','test_used_for_selection':False,'pairing':'HILA_s -> frozen logits_s/features_s -> zero-init no-anchor HiLAR_s','hug_checkpoint':str(hug),'hug_sha256':sha(hug),'hila_checkpoint':str(hila),'hila_sha256':sha(hila),'hilar_checkpoint':str(hilar),'hilar_sha256':sha(hilar),'feature_checkpoint_exact':True,'subset_manifest':str(run/'subset-manifest.json'),'recovery':'continued after container GPU restart without replacing completed checkpoints'}
q=run/'train-val-complete.json.incoming'; q.write_text(json.dumps(p,indent=2)+'\n'); os.replace(q,run/'train-val-complete.json')
s=run/'status.json'; q=s.with_suffix('.json.incoming'); q.write_text(json.dumps({'state':'complete','stage':'complete','fraction':fkey,'task':task,'seed':seed},indent=2)+'\n'); os.replace(q,s)
PY
