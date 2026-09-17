#!/usr/bin/env python3
"""Deploy and safely start the ECGFounder/ECG-JEPA comparison on server 204."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path, PurePosixPath

import paramiko


ROOT = "/root/107552503710-1"
EXPERIMENT = "q1_ecgfounder_ecgjepa_comparison_20260917"
CAMPAIGN = "q1-ecgfounder-ecgjepa-six-task-three-fraction-seeds42-46-55-20260917"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(__file__).resolve().parents[4]


def config() -> dict[str, str]:
    values = {}
    for raw in (WORKSPACE / ".env.remote.secondary").read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
            key, value = raw.split("=", 1); values[key.strip()] = value.strip()
    return values


def connect() -> paramiko.SSHClient:
    cfg = config(); client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(cfg["CODEX_REMOTE_HOST"], port=int(cfg["CODEX_REMOTE_PORT"]),
                   username=cfg.get("CODEX_REMOTE_USER", "root"), password=cfg["CODEX_REMOTE_PASSWORD"],
                   timeout=30, banner_timeout=30, auth_timeout=30)
    return client


def run(client, command: str, check: bool = True):
    _, stdout, stderr = client.exec_command(command); out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace"); code = stdout.channel.recv_exit_status()
    if check and code: raise RuntimeError(f"remote command failed ({code}): {err}\n{out}")
    return code, out, err


def mkdirs(sftp, path: str) -> None:
    current = PurePosixPath("/")
    for part in PurePosixPath(path).parts[1:]:
        current /= part
        try: sftp.stat(str(current))
        except FileNotFoundError: sftp.mkdir(str(current))


def main() -> None:
    client = connect()
    try:
        destination = f"{ROOT}/src/CLEAR-HUG/experiments/{EXPERIMENT}"
        sftp = client.open_sftp(); mkdirs(sftp, destination)
        for name in ("prepare_features.py", "run_probe.py", "queue_runner.py", "deploy_start.py", "README.md"):
            remote = f"{destination}/{name}"; sftp.put(str(HERE / name), remote + ".incoming")
            sftp.chmod(remote + ".incoming", 0o555 if name.endswith(".py") else 0o444)
            sftp.posix_rename(remote + ".incoming", remote)
        sftp.close(); python = f"{ROOT}/envs/ecg-fix/bin/python"
        run(client, f"{python} -m py_compile {destination}/*.py")
        _, cuda, _ = run(client, f"{python} - <<'PY'\nimport torch\nprint(torch.cuda.is_available(), torch.cuda.device_count())\nPY")
        if cuda.strip() != "True 3": raise RuntimeError(f"expected three CUDA GPUs, got {cuda.strip()}")
        # Deployment identities and loader shapes must pass immediately before launch.
        verify = f"{ROOT}/src/CLEAR-HUG/experiments/q1_ecgfounder_ecgjepa_official_deploy_20260917/verify_deployment.py"
        run(client, f"CUDA_VISIBLE_DEVICES=0 {python} {verify} --root {ROOT} >/tmp/q1-founder-jepa-verify.log")
        campaign = f"{ROOT}/results/{CAMPAIGN}"; run(client, f"mkdir -p {campaign}")
        _, active, _ = run(client, f"pgrep -af '{destination}/queue_runner.py'", check=False)
        if active.strip(): print("already-running\n" + active); return
        command = (f"nohup setsid {python} {destination}/queue_runner.py --root {ROOT} "
                   f"> {campaign}/queue.log 2>&1 < /dev/null & echo $! > {campaign}/queue.pid")
        run(client, command); time.sleep(3)
        _, status, _ = run(client, f"cat {campaign}/queue.pid; ps -ef | grep '{EXPERIMENT}' | grep -v grep; "
                                    f"cat {campaign}/queue-status.json 2>/dev/null || true", check=False)
        print(status)
    finally: client.close()


if __name__ == "__main__": main()
