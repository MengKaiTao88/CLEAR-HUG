#!/usr/bin/env python3
"""Deploy and start the PyTorch linear-probe campaign on server 204."""
from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path, PurePosixPath

import paramiko


ROOT = "/root/107552503710-1"
EXPERIMENT = "q1_ecgfix_clocs_pytorch_linear_20260917"
CAMPAIGN = "q1-ecgfix-clocs-pytorch-linear-six-task-three-fraction-seeds42-46-55-20260917"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(__file__).resolve().parents[4]


def env() -> dict[str, str]:
    values = {}
    for raw in (WORKSPACE / ".env.remote.secondary").read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
            key, value = raw.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def connect() -> paramiko.SSHClient:
    cfg = env()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(cfg["CODEX_REMOTE_HOST"], port=int(cfg["CODEX_REMOTE_PORT"]),
                   username=cfg.get("CODEX_REMOTE_USER", "root"), password=cfg["CODEX_REMOTE_PASSWORD"],
                   timeout=30, banner_timeout=30, auth_timeout=30)
    return client


def run(client: paramiko.SSHClient, command: str, check: bool = True) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode(errors="replace")
    error = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if check and code:
        raise RuntimeError(f"remote command failed ({code}): {error}\n{output}")
    return code, output, error


def mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    current = PurePosixPath("/")
    for part in PurePosixPath(path).parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current))


def main() -> None:
    client = connect()
    try:
        destination = f"{ROOT}/src/CLEAR-HUG/experiments/{EXPERIMENT}"
        sftp = client.open_sftp()
        mkdirs(sftp, destination)
        for name in ("run_probe.py", "queue_runner.py", "deploy_start.py", "README.md"):
            remote = f"{destination}/{name}"
            sftp.put(str(HERE / name), remote + ".incoming")
            sftp.chmod(remote + ".incoming", 0o555 if name.endswith(".py") else 0o444)
            sftp.posix_rename(remote + ".incoming", remote)
        sftp.close()
        python = f"{ROOT}/envs/ecg-fix/bin/python"
        run(client, f"{python} -m py_compile {destination}/run_probe.py {destination}/queue_runner.py")
        code, output, error = run(client, f"{python} - <<'PY'\nimport torch\nprint(torch.cuda.is_available(), torch.cuda.device_count())\nPY", check=False)
        print("CUDA_PREFLIGHT", output.strip(), error.strip())
        campaign = f"{ROOT}/results/{CAMPAIGN}"
        run(client, f"mkdir -p {campaign}")
        if output.strip() != "True 3":
            run(client, f"printf '%s\\n' '{{\"state\":\"waiting_for_gpu\",\"required_gpus\":3}}' > {campaign}/queue-status.json")
            print("DEPLOYED_WAITING_FOR_3_GPUS")
            return
        command = (f"if test -s {campaign}/queue.pid && kill -0 $(cat {campaign}/queue.pid) 2>/dev/null; "
                   f"then echo already-running; else nohup setsid {python} {destination}/queue_runner.py "
                   f"--root {ROOT} > {campaign}/queue.log 2>&1 < /dev/null & echo $! > {campaign}/queue.pid; "
                   f"echo started; fi")
        _, started, _ = run(client, command)
        print(started.strip())
        time.sleep(2)
        _, verification, _ = run(client, f"cat {campaign}/queue-status.json; ps -ef | grep q1_ecgfix_clocs_pytorch_linear_20260917 | grep -v grep", check=False)
        print(verification)
    finally:
        client.close()


if __name__ == "__main__":
    main()
