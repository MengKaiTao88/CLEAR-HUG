#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLEAR_HUG_ROOT:?}"; source "${ROOT}/mvp/activate_mvp.sh"
CAMPAIGN=q1-hilar-rerun-six-task-three-fraction-seeds42-46-55-20260918
CODE="${ROOT}/src/CLEAR-HUG/experiments/q1_hilar_rerun_six_task_three_fraction_seeds42_46_55_20260918"
OUT="${ROOT}/results/${CAMPAIGN}"; STATUS="${OUT}/supervisor-status.json"
write_status(){ python - "${STATUS}" "$1" "${2:-}" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]);q=p.with_suffix('.json.incoming');q.write_text(json.dumps({'state':sys.argv[2],'stage':sys.argv[3] or None,'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');os.replace(q,p)
PY
}
stage=waiting-train-val; trap 'c=$?; [[ $c -eq 0 ]] || write_status failed "${stage}:exit=${c}"' EXIT
write_status running "${stage}"
while true; do
 failed=$(python - "${OUT}" <<'PY'
import json,sys
from pathlib import Path
print(sum(json.loads(p.read_text()).get('state')=='failed' for p in Path(sys.argv[1]).glob('gpu*-queue-status.json')))
PY
)
 [[ "${failed}" -eq 0 ]] || { echo "training queue failure" >&2; exit 30; }
 complete=$(find "${OUT}" -name train-val-complete.json | wc -l)
 [[ "${complete}" -eq 54 ]] && break
 sleep 300
done
stage=global-gate; write_status running "${stage}"
python "${CODE}/build_global_gate.py" --results "${OUT}" --output "${OUT}/global-pretest-gate.json"
stage=formal-test; write_status running "${stage}"
CUDA_VISIBLE_DEVICES=0 bash "${CODE}/run_formal_queue.sh" 0 42 > "${OUT}/formal-gpu0-seed42.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 bash "${CODE}/run_formal_queue.sh" 1 46 > "${OUT}/formal-gpu1-seed46.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=2 bash "${CODE}/run_formal_queue.sh" 2 55 > "${OUT}/formal-gpu2-seed55.log" 2>&1 & p2=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"
[[ $(find "${OUT}" -name formal-test -type d -prune -exec test -s '{}/complete.json' \; -print | wc -l) -eq 54 ]]
stage=analysis; write_status running "${stage}"
python "${CODE}/analyze_results.py" --results "${OUT}" --output "${OUT}/summary.json"
write_status complete complete
