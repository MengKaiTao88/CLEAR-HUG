#!/usr/bin/env python3
"""Deploy and enqueue the CLOCS campaign on the 204 three-GPU server."""
from __future__ import annotations

import json
import shlex
import time
from pathlib import Path, PurePosixPath

import paramiko


ROOT = "/root/107552503710-1"
EXPERIMENT = "q1_ecgfix_clocs_six_task_three_seed_20260916"
CAMPAIGN = "q1-ecgfix-clocs-six-task-three-fraction-seeds42-46-55-20260916"
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


def run(client: paramiko.SSHClient, command: str) -> str:
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode(errors="replace")
    error = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"remote command failed ({code}): {error}\n{output}")
    return output


def mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    current = PurePosixPath("/")
    for part in PurePosixPath(path).parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current))


def upload(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    mkdirs(sftp, str(PurePosixPath(remote).parent))
    incoming = remote + ".incoming"
    sftp.put(str(local), incoming)
    sftp.chmod(incoming, 0o555 if local.suffix == ".py" else 0o444)
    sftp.posix_rename(incoming, remote)


def main() -> None:
    client = connect()
    try:
        destination = f"{ROOT}/src/CLEAR-HUG/experiments/{EXPERIMENT}"
        sftp = client.open_sftp()
        for name in (
            "run_campaign.py",
            "prepare_csn_headers.py",
            "queue_runner.py",
            "deploy_queue.py",
            "README.md",
        ):
            upload(sftp, HERE / name, f"{destination}/{name}")
        sftp.close()
        output = f"{ROOT}/results/{CAMPAIGN}"
        pidfile = f"{output}/queue.pid"
        logfile = f"{output}/queue.log"
        python = f"{ROOT}/envs/ecg-fix/bin/python"
        runner = f"{destination}/queue_runner.py"
        run(client, f"{shlex.quote(python)} -m py_compile "
                    f"{shlex.quote(destination + '/run_campaign.py')} "
                    f"{shlex.quote(destination + '/prepare_csn_headers.py')} "
                    f"{shlex.quote(destination + '/queue_runner.py')}")
        run(client, f"mkdir -p {shlex.quote(output)}")
        command = (
            f"if test -s {shlex.quote(pidfile)} && kill -0 \"$(cat {shlex.quote(pidfile)})\" 2>/dev/null; then "
            f"echo already-running; else nohup setsid {shlex.quote(python)} {shlex.quote(runner)} "
            f"--root {shlex.quote(ROOT)} > {shlex.quote(logfile)} 2>&1 < /dev/null & "
            f"echo $! > {shlex.quote(pidfile)}; echo started; fi"
        )
        start = run(client, command).strip()
        time.sleep(2)
        verification = run(client, f"printf 'START=%s\\nPID=%s\\n' {shlex.quote(start)} \"$(cat {pidfile})\"; "
                                   f"kill -0 \"$(cat {pidfile})\"; cat {output}/queue-status.json; "
                                   "printf '\\nGPU_PROCESSES\\n'; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
        print(verification)
    finally:
        client.close()


if __name__ == "__main__":
    main()
