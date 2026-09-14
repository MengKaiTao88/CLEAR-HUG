#!/usr/bin/env python3
"""Deploy and start the official ST-MEM downstream queues."""
from __future__ import annotations

import json
import shlex
import time

from deploy_and_start import CODE_DIRNAME, connect, remote_sha, upload_code
from official_protocol import CAMPAIGN, NODES, NODE_TASKS, SOURCE_CAMPAIGN, STMEM_SHA256, TASKS


def run(client, command):
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode(errors="replace"); error = stderr.read().decode(errors="replace")
    if stdout.channel.recv_exit_status(): raise RuntimeError(error or output)
    return output


def main():
    clients = {node: connect(node) for node in NODES}
    try:
        for node, client in clients.items():
            root = NODES[node][3]; upload_code(client, root)
            checkpoint = f"{root}/model_weights/ST-MEM/st_mem_vit_base_encoder.pth"
            if remote_sha(client, checkpoint) != STMEM_SHA256:
                raise RuntimeError(f"checkpoint mismatch on {node}")
            for task in NODE_TASKS[node]:
                raw_rel = TASKS[task][2]
                raw = f"{root}/campaign-inputs/{SOURCE_CAMPAIGN}/raw/{raw_rel}"
                check = run(client, f"test -f {shlex.quote(raw + '/train_data.npy')} && test -f {shlex.quote(raw + '/val_data.npy')} && echo ok").strip()
                if check != "ok": raise RuntimeError(f"missing source arrays {node}:{task}")
            python = f"{root}/.venv/bin/python"
            script = f"{root}/src/CLEAR-HUG/experiments/{CODE_DIRNAME}/run_official_finetune.py"
            result = f"{root}/results/{CAMPAIGN}"
            preload = "env LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 " if node == "10110" else ""
            marker = f"{result}/{node}-train-queue-status.json"
            command = (f"mkdir -p {shlex.quote(result)}; if ! test -f {shlex.quote(marker)}; then "
                       f"nohup setsid {preload}{python} {script} --root {root} --node {node} "
                       f"> {result}/{node}-train.log 2>&1 < /dev/null & echo $! > {result}/{node}-train.pid; fi")
            _, stdout, _ = client.exec_command(command); time.sleep(1); stdout.channel.close()
            print(json.dumps({"node": node, "status": "started_or_existing", "tasks": NODE_TASKS[node]}), flush=True)
    finally:
        for client in clients.values(): client.close()


if __name__ == "__main__": main()
