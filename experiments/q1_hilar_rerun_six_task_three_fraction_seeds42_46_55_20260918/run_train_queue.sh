#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:?}"; source "${ROOT}/mvp/activate_mvp.sh"; GPU="${1:?gpu required}"; SEED="${2:?seed required}"
CAMPAIGN=q1-hilar-rerun-six-task-three-fraction-seeds42-46-55-20260918
CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_hilar_rerun_six_task_three_fraction_seeds42_46_55_20260918"
case "${GPU}" in 0|1|2) ;; *) exit 2 ;; esac; case "${SEED}" in 42|46|55) ;; *) exit 2 ;; esac
OUT="${ROOT}/results/${CAMPAIGN}"; mkdir -p "${OUT}"; STATUS="${OUT}/gpu${GPU}-seed${SEED}-queue-status.json"; LOG="${OUT}/gpu${GPU}-seed${SEED}-queue.log"
exec > >(tee -a "${LOG}") 2>&1
write_status(){ python - "${STATUS}" "$1" "${2:-}" "${GPU}" "${SEED}" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]);q=p.with_suffix('.json.incoming');q.write_text(json.dumps({'state':sys.argv[2],'spec':sys.argv[3] or None,'gpu':int(sys.argv[4]),'seed':int(sys.argv[5]),'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');os.replace(q,p)
PY
}
current=initializing; trap 'c=$?; [[ $c -eq 0 ]] || write_status failed "${current}:exit=${c}"' EXIT
for fraction in 1pct 10pct 100pct; do
 for task in superdiagnostic subdiagnostic form rhythm cpsc2018 csn; do
  current="${fraction}:${task}:${SEED}"; write_status running "${current}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash "${CODE}/run_train_val_unit.sh" "${fraction}" "${task}" "${SEED}"
 done
done
write_status complete ""
