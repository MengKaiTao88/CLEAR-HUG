#!/usr/bin/env python3
"""Read-only three-node campaign progress monitor."""
from __future__ import annotations

import json
from inspect_remote import NODES, env_file

import paramiko

from protocol import CAMPAIGN


def inspect(node_info):
    node, env_name, host, port, root = node_info
    cfg = env_file(env_name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, port=port, username=cfg.get("CODEX_REMOTE_USER", "root"),
                       password=cfg["CODEX_REMOTE_PASSWORD"], timeout=20)
        campaign = f"{root}/results/{CAMPAIGN}"
        command = f'''python3 - <<'PY'
import json, pathlib
r=pathlib.Path("{campaign}")
models=("heartlang","stmem")
complete=[p for m in models for p in r.glob(f"*pct/*-seed*/{{m}}/training-complete.json")]
formal=[p for m in models for p in r.glob(f"*pct/*-seed*/{{m}}/formal-test/formal-test-result.json")]
queues=[]
for p in sorted(r.glob("*-queue-status.json")):
    try: queues.append(json.loads(p.read_text()))
    except Exception: pass
print(json.dumps({{"training":len(complete),"formal":len(formal),"queues":queues}}))
PY
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
for log in {campaign}/{node}-formal.log {campaign}/{node}-train.log {campaign}/{node}-canary.log; do
  if test -f "$log"; then echo LOG=$log; tail -n 8 "$log"; fi
done'''
        _, stdout, stderr = client.exec_command(command)
        output = stdout.read().decode(errors="replace")
        error = stderr.read().decode(errors="replace")
        return {"node": node, "output": output, "error": error}
    except Exception as exc:
        return {"node": node, "connect_error": f"{type(exc).__name__}: {exc}"}
    finally:
        client.close()


def main():
    reports = [inspect(node) for node in NODES]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
