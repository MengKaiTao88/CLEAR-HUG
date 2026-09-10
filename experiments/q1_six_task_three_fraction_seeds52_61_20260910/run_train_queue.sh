#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:?}"; NODE="${1:?node required}"; shift; [[ $# -eq 60 ]]
source "${ROOT}/mvp/activate_mvp.sh"
CAMPAIGN=q1-six-task-three-fraction-seeds52-61-20260910; CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_six_task_three_fraction_seeds52_61_20260910"
OUT="${ROOT}/results/${CAMPAIGN}"; mkdir -p "${OUT}"; STATUS="${OUT}/${NODE}-queue-status.json"; LOG="${OUT}/${NODE}-queue.log"
exec > >(tee -a "${LOG}") 2>&1
write_status(){ python - "${STATUS}" "$1" "${2:-}" "${NODE}" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]);q=p.with_suffix('.json.incoming');q.write_text(json.dumps({'state':sys.argv[2],'spec':sys.argv[3] or None,'node':sys.argv[4],'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');os.replace(q,p)
PY
}
current=initializing; trap 'c=$?; [[ $c -eq 0 ]] || write_status failed "${current}:exit=${c}"' EXIT
for spec in "$@"; do current="${spec}"; write_status running "${spec}"; IFS=: read -r fraction task seed <<<"${spec}"; bash "${CODE}/run_train_val_unit.sh" "${fraction}" "${task}" "${seed}"; done
write_status complete ""
