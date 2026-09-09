#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLEAR_HUG_ROOT:-/root/107552503710}"
SEED="${1:?seed required}"
case "${SEED}" in 43|45|47) ;; *) echo "unsupported seed ${SEED}" >&2; exit 2 ;; esac

CAMPAIGN="${CSN_LOW_LABEL_CAMPAIGN:-q1-csn-low-label-10pct-20260909}"
TRAIN_FRACTION="${CSN_LOW_LABEL_FRACTION:-0.1}"
EXPECTED_TRAIN_RECORDS="${CSN_LOW_LABEL_RECORDS:-1654}"
CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_csn_low_label_10pct_20260909"
INPUT="${ROOT}/campaign-inputs/q1-clear-deepsets-hilar-10seed-20260903/ecg_datasets-resolved-v2/CSN_QRS/data"
RUN="${ROOT}/results/${CAMPAIGN}/csn-seed${SEED}"
HUG="${RUN}/clear-hug"
HILA="${RUN}/hila"
FEATURES="${RUN}/features"
HILAR_ROOT="${RUN}/hilar-training"
HILAR="${HILAR_ROOT}/parameter-matched-direct"
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
  "${ROOT}/.venv/bin/python" - "${STATUS}" "$1" "${2:-}" "${SEED}" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1]); q=p.with_suffix('.json.incoming')
q.write_text(json.dumps({'state':sys.argv[2], 'stage':sys.argv[3] or None, 'task':'csn', 'seed':int(sys.argv[4]), 'updated_at':datetime.now(timezone.utc).isoformat()}, indent=2)+'\n')
os.replace(q,p)
PY
}
stage=initializing
trap 'code=$?; if [[ $code -ne 0 ]]; then write_status failed "${stage}:exit=${code}"; fi' EXIT
write_status running "${stage}"

source "${ROOT}/mvp/activate_mvp.sh"
export CLEAR_HUG_ROOT="${ROOT}" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
test -d "${INPUT}"
for split in train val test; do
  test -s "${INPUT}/${split}_data.npy"
  test -s "${INPUT}/${split}_labels.npy"
done
"${ROOT}/.venv/bin/python" - "${INPUT}" "${SEED}" "${RUN}/subset-manifest.json" "${TRAIN_FRACTION}" <<'PY'
import hashlib, json, random, sys
from pathlib import Path
import numpy as np
root, seed, output, fraction = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
labels = np.load(root/'train_labels.npy', mmap_mode='r')
indices = np.asarray(
    random.Random(seed).sample(range(len(labels)), int(len(labels) * fraction)),
    dtype=np.int64,
)
payload = {
    'seed': seed,
    'sampling_method': 'random',
    'train_fraction': fraction,
    'source_train_records': len(labels),
    'selected_train_records': len(indices),
    'indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
    'selected_labels_sha256': hashlib.sha256(
        np.asarray(labels[indices], dtype=np.float32).tobytes()
    ).hexdigest(),
}
output.write_text(json.dumps(payload, indent=2)+'\n')
PY

stage=hug-training
write_status running "${stage}"
mkdir -p "${HUG}-tb"
cd "${ROOT}/src/CLEAR-HUG"
"${ROOT}/.venv/bin/python" -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
  --master_port="$((51000 + SEED))" run_class_finetuning.py \
  --dataset_dir "${INPUT}" --output_dir "${HUG}" --log_dir "${HUG}-tb" \
  --model CLEAR_HUG_finetune_base --trainable moe --split_ratio "${TRAIN_FRACTION}" \
  --finetune "${ROOT}/checkpoints/released_ckpt.pth" --sampling_method random \
  --weight_decay 0.05 --batch_size 256 --lr 5e-3 --update_freq 1 \
  --warmup_epochs 10 --epochs 100 --layer_decay 0.9 --save_ckpt_freq 100 \
  --seed "${SEED}" --is_binary --nb_classes 38 --world_size 1 --atten_mask \
  --cls_token_num 12 --mask_ratio 0 --num_workers 10 --screen_only \
  --screen_select_metric roc_auc --screen_patience 0
test -s "${HUG}/checkpoint-best.pth"
"${ROOT}/.venv/bin/python" - "${HUG}" "${SEED}" "${TRAIN_FRACTION}" <<'PY'
import json, os, sys
from pathlib import Path
out, seed, fraction = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
rows=[json.loads(x) for x in (out/'log.txt').read_text().splitlines() if x.strip()]
assert len(rows)==100, len(rows)
best=max(rows,key=lambda row:float(row['val_roc_auc']))
payload={'task':'csn','seed':seed,'epochs':100,'train_fraction':fraction,
 'sampling_method':'random','selection_split':'validation','selection_metric':'macro_auroc',
 'best_epoch':int(best['epoch']),'best_validation_auroc':float(best['val_roc_auc']),
 'best_validation_auprc':float(best['val_pr_auc']),'test_used_for_selection':False}
q=out/'training-complete.json.incoming'; q.write_text(json.dumps(payload,indent=2)+'\n'); os.replace(q,out/'training-complete.json')
PY

stage=hila-training
write_status running "${stage}"
mkdir -p "${HILA}-tb"
cd "${ROOT}/src/CLEAR-HUG"
"${ROOT}/.venv/bin/python" -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
  --master_port="$((52000 + SEED))" "${ROOT}/mvp/run_fair_baseline_finetuning.py" \
  --dataset_dir "${INPUT}" --output_dir "${HILA}" --log_dir "${HILA}-tb" \
  --model CLEAR_MASKED_DEEPSETS_finetune_base --finetune "${ROOT}/checkpoints/released_ckpt.pth" \
  --trainable adapter --split_ratio "${TRAIN_FRACTION}" --sampling_method random --weight_decay 0.05 \
  --batch_size 256 --lr 5e-3 --update_freq 1 --warmup_epochs 10 --epochs 100 \
  --layer_decay 0.9 --save_ckpt_freq 100 --seed "${SEED}" --is_binary \
  --nb_classes 38 --world_size 1 --atten_mask --cls_token_num 12 --mask_ratio 0 \
  --num_workers 10 --screen_only --screen_select_metric roc_auc --screen_patience 0
test -s "${HILA}/checkpoint-best.pth"
"${ROOT}/.venv/bin/python" - "${HILA}" "${SEED}" "${TRAIN_FRACTION}" <<'PY'
import json, os, sys
from pathlib import Path
out, seed, fraction = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
rows=[json.loads(x) for x in (out/'log.txt').read_text().splitlines() if x.strip()]
assert len(rows)==100, len(rows)
best=max(rows,key=lambda row:float(row['val_roc_auc']))
payload={'task':'csn','seed':seed,'epochs':100,'split_ratio':fraction,'sampling_method':'random',
 'selection_split':'validation','selection_metric':'macro_auroc','best_epoch':int(best['epoch']),
 'best_validation_auroc':float(best['val_roc_auc']),'best_validation_auprc':float(best['val_pr_auc']),
 'test_used_for_selection':False}
q=out/'baseline-complete.json.incoming'; q.write_text(json.dumps(payload,indent=2)+'\n'); os.replace(q,out/'baseline-complete.json')
PY

stage=cache-train-val
write_status running "${stage}"
"${ROOT}/.venv/bin/python" "${CODE}/cache_low_label_features.py" \
  --root "${ROOT}" --dataset "${INPUT}" --checkpoint "${HILA}/checkpoint-best.pth" \
  --output "${FEATURES}" --classes 38 --seed "${SEED}" --splits train val \
  --train-split-ratio "${TRAIN_FRACTION}" --sampling-method random

stage=hilar-training
write_status running "${stage}"
"${ROOT}/.venv/bin/python" "${ROOT}/mvp/train_deepsets_anchored_residual.py" \
  --features "${FEATURES}" --output "${HILAR_ROOT}" --task csn --classes 38 \
  --seed "${SEED}" --selection-metric auroc --variants parameter-matched-direct

stage=audit
write_status running "${stage}"
"${ROOT}/.venv/bin/python" - "${RUN}" "${SEED}" "${TRAIN_FRACTION}" "${EXPECTED_TRAIN_RECORDS}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
run, seed, fraction, expected_records = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()
hug=run/'clear-hug/checkpoint-best.pth'; hila=run/'hila/checkpoint-best.pth'
hilar=run/'hilar-training/parameter-matched-direct/checkpoint-best.pth'
for path in (hug,hila,hilar):
    if not path.is_file(): raise RuntimeError(f'missing checkpoint {path}')
hila_meta=json.loads((run/'hila/baseline-complete.json').read_text())
features=json.loads((run/'features/feature-manifest.json').read_text())
residual=json.loads((run/'hilar-training/parameter-matched-direct/complete.json').read_text())
assert hila_meta['seed']==seed and hila_meta['split_ratio']==fraction
assert features['checkpoint']==str(hila) and features['checkpoint_seed']==seed
assert features['train_split_ratio']==fraction and features['splits']['train']['records']==expected_records
assert residual['selection_metric']=='auroc' and float(residual['anchor'])==0.0
payload={'schema_version':1,'status':'complete','task':'csn','seed':seed,'train_fraction':fraction,
 'sampling_method':'random','train_records':expected_records,'validation_is_full':True,'test_is_full':True,
 'selection_metric':'macro_auroc','test_used_for_selection':False,
 'pairing':'HILA_s -> frozen logits_s/local features_s -> zero-init no-anchor HiLAR_s',
 'hug_checkpoint':str(hug),'hug_sha256':sha(hug),'hila_checkpoint':str(hila),
 'hila_sha256':sha(hila),'hilar_checkpoint':str(hilar),'hilar_sha256':sha(hilar)}
q=run/'train-val-complete.json.incoming'; q.write_text(json.dumps(payload,indent=2)+'\n'); os.replace(q,run/'train-val-complete.json')
PY
stage=complete
write_status complete "${stage}"
