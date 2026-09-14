#!/usr/bin/env python3
"""Read-only status monitor for the official ST-MEM fine-tuning campaign."""
from __future__ import annotations

import json
from deploy_and_start import connect, run
from official_protocol import CAMPAIGN, NODES


def main():
    reports = []
    for node in NODES:
        client = connect(node); root = NODES[node][3]; result = f"{root}/results/{CAMPAIGN}"
        command = f'''python3 - <<'PY'
import json,pathlib
r=pathlib.Path("{result}")
done=list(r.glob("*pct/*-seed*/stmem/training-complete.json"))
status={{}}
p=r/"{node}-train-queue-status.json"
if p.exists(): status=json.loads(p.read_text())
by_fraction={{fraction:len(list(r.glob(f"{{fraction}}/*-seed*/stmem/training-complete.json"))) for fraction in ("100pct","10pct","1pct")}}
current_log=[]
spec=status.get("spec")
if spec:
    fraction,task,seed=spec.split(":")
    log=r/fraction/f"{{task}}-seed{{seed}}"/"stmem"/"log.txt"
    if log.exists(): current_log=log.read_text().splitlines()[-2:]
print(json.dumps({{"completed":len(done),"by_fraction":by_fraction,"status":status,"current_log":current_log}}))
PY
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -eo etime,pid,args | grep '[t]rain_official_finetune.py' | tail -1
tail -n 5 {result}/{node}-train.log 2>/dev/null || true'''
        try: reports.append({"node": node, "output": run(client, command)})
        finally: client.close()
    print(json.dumps(reports, indent=2))


if __name__ == "__main__": main()
