#!/usr/bin/env python3
"""Pause only the official ST-MEM fine-tuning queues, preserving artifacts."""
from __future__ import annotations

import json

from deploy_and_start import connect, run
from official_protocol import CAMPAIGN, NODES


def main() -> None:
    reports = []
    for node in NODES:
        client = connect(node)
        root = NODES[node][3]
        campaign_root = f"{root}/results/{CAMPAIGN}"
        command = rf'''{root}/.venv/bin/python - <<'PY'
import json, os, pathlib, signal, time
campaign = "{CAMPAIGN}"
allowed = ("run_official_finetune.py", "train_official_finetune.py")
targets = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        raw = (entry / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    argv = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
    executable = pathlib.Path(argv[0]).name if argv else ""
    is_campaign_python = executable.startswith("python") and any(
        any(arg.endswith(name) for name in allowed) for arg in argv
    )
    if argv and campaign in " ".join(argv) and is_campaign_python:
        targets.append((int(entry.name), argv))
# Stop queue parents first so terminated workers cannot advance the queue.
targets.sort(key=lambda item: 0 if any("run_official_finetune.py" in arg for arg in item[1]) else 1)
for pid, _ in targets:
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
forced = []
for pid, _ in targets:
    try:
        os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
        forced.append(pid)
    except ProcessLookupError:
        pass
print(json.dumps({{"targets": [pid for pid, _ in targets], "forced": forced}}))
PY
python3 - <<'PY'
import datetime, json, pathlib
p=pathlib.Path("{campaign_root}/{node}-train-queue-status.json")
old=json.loads(p.read_text()) if p.exists() else {{}}
old.update({{"state":"paused","error":None,"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()}})
incoming=p.with_name(p.name+".incoming"); incoming.write_text(json.dumps(old,indent=2,sort_keys=True)+"\n"); incoming.replace(p)
print(p.read_text())
PY
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -eo pid,args | grep -E '[r]un_official_finetune.py|[t]rain_official_finetune.py' || true'''
        try:
            reports.append({"node": node, "output": run(client, command)})
        finally:
            client.close()
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
