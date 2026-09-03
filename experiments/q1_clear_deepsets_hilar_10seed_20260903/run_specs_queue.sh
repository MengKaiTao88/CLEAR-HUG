#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:?CLEAR_HUG_ROOT is required}"
QUEUE="${1:?queue name required}"; shift
[[ $# -gt 0 ]]
source "${ROOT}/mvp/activate_mvp.sh"
CAMPAIGN="q1-clear-deepsets-hilar-10seed-20260903"
OUT="${ROOT}/results/${CAMPAIGN}"
mkdir -p "${OUT}"
STATUS="${OUT}/${QUEUE}-queue-status.json"
LOG="${OUT}/${QUEUE}-queue.log"
exec > >(tee -a "${LOG}") 2>&1
write_status() {
 python - "${STATUS}" "$1" "${2:-}" "${QUEUE}" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]); q=p.with_suffix('.json.incoming')
q.write_text(json.dumps({"state":sys.argv[2],"spec":sys.argv[3] or None,"queue":sys.argv[4],"updated_at":datetime.now(timezone.utc).isoformat()},indent=2)+'\n'); os.replace(q,p)
PY
}
current=initializing
trap 'code=$?; if [[ $code -ne 0 ]]; then write_status failed "${current}:exit=${code}"; fi' EXIT
for spec in "$@"; do
 current="${spec}"; write_status running "${spec}"
 task="${spec%%:*}"; seed="${spec##*:}"
 bash "$(dirname "$0")/run_train_val_task_seed.sh" "${task}" "${seed}"
done
write_status complete ""
echo "QUEUE_COMPLETE queue=${QUEUE} time=$(date -Is)"
