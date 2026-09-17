#!/usr/bin/env python3
"""Deploy and queue the custom ST-MEM seed-123 comparison on server 204."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path, PurePosixPath

import paramiko


ROOT = "/root/107552503710-1"
EXPERIMENT = "q1_stmem_seed123_comparison_20260917"
CAMPAIGN = "q1-stmem-seed123-six-task-three-fraction-seeds42-46-55-20260917"
CHECKPOINT_SHA256 = "ad7e8527002213dda15e51c583108bf12aa01e1571935009e98b8e7f5ac09df9"
HERE = Path(__file__).resolve().parent; WORKSPACE = Path(__file__).resolve().parents[4]


def cfg():
    values = {}
    for line in (WORKSPACE / ".env.remote.secondary").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1); values[key] = value
    return values


def mkdirs(sftp, path: str):
    current = PurePosixPath("/")
    for part in PurePosixPath(path).parts[1:]:
        current /= part
        try: sftp.stat(str(current))
        except FileNotFoundError: sftp.mkdir(str(current))


def run(client, command: str, check=True):
    _, stdout, stderr = client.exec_command(command); out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace"); code = stdout.channel.recv_exit_status()
    if check and code: raise RuntimeError(f"remote failed ({code}): {err}\n{out}")
    return code, out, err


def main():
    values = cfg(); client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(values["CODEX_REMOTE_HOST"], port=int(values["CODEX_REMOTE_PORT"]),
                   username=values["CODEX_REMOTE_USER"], password=values["CODEX_REMOTE_PASSWORD"])
    try:
        destination = f"{ROOT}/src/CLEAR-HUG/experiments/{EXPERIMENT}"; sftp = client.open_sftp()
        mkdirs(sftp, destination)
        for name in ("prepare_features.py", "run_probe.py", "queue_runner.py", "deploy_start.py", "README.md"):
            remote = f"{destination}/{name}"; sftp.put(str(HERE / name), remote + ".incoming")
            sftp.chmod(remote + ".incoming", 0o555 if name.endswith(".py") else 0o444)
            sftp.posix_rename(remote + ".incoming", remote)
        sftp.close(); python = f"{ROOT}/envs/ecg-fix/bin/python"
        run(client, f"test \"$(sha256sum {ROOT}/model_weights/ST-MEM-seed123/pretrained_checkpoint.pt | cut -d' ' -f1)\" = {CHECKPOINT_SHA256}")
        run(client, f"{python} -m py_compile {destination}/*.py")
        campaign = f"{ROOT}/results/{CAMPAIGN}"; run(client, f"mkdir -p {campaign}")
        _, active, _ = run(client, f"pgrep -af '{destination}/queue_runner.py'", check=False)
        if not active.strip():
            run(client, f"nohup setsid {python} {destination}/queue_runner.py --root {ROOT} > {campaign}/queue.log 2>&1 < /dev/null & echo $! > {campaign}/queue.pid")
        time.sleep(2); _, output, _ = run(client, f"cat {campaign}/queue-status.json; cat {campaign}/queue.pid", check=False)
        print(output)
    finally: client.close()


if __name__ == "__main__": main()
