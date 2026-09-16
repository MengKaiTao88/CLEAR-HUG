#!/usr/bin/env python3
"""Deploy official splits and start one balanced MERL queue per GPU."""
from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path, PurePosixPath

import paramiko


CAMPAIGN = "q1-merl-five-task-three-seed-20260916"
ROOT = "/root/107552503710"
EXPERIMENT = "q1_merl_five_task_three_seed_20260916"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(__file__).resolve().parents[4]
OFFICIAL = WORKSPACE / "external_models_staging" / "MERL-ICML2024" / "finetune" / "data_split"


def env() -> dict[str, str]:
    values = {}
    for raw in (WORKSPACE / ".env.remote").read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
            key, value = raw.split("=", 1); values[key.strip()] = value.strip()
    return values


def connect() -> paramiko.SSHClient:
    config = env(); client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(config["CODEX_REMOTE_HOST"], port=int(config["CODEX_REMOTE_PORT"]),
                   username=config.get("CODEX_REMOTE_USER", "root"), password=config["CODEX_REMOTE_PASSWORD"],
                   timeout=30, banner_timeout=30, auth_timeout=30)
    return client


def run(client: paramiko.SSHClient, command: str) -> str:
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode(errors="replace"); error = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"remote command failed ({code}): {error}\n{output}")
    return output


def mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    current = PurePosixPath("/")
    for part in PurePosixPath(path).parts[1:]:
        current /= part
        try: sftp.stat(str(current))
        except FileNotFoundError: sftp.mkdir(str(current))


def upload(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    mkdirs(sftp, str(PurePosixPath(remote).parent)); incoming = remote + ".incoming"
    sftp.put(str(local), incoming); sftp.chmod(incoming, 0o444); sftp.posix_rename(incoming, remote)


def main() -> None:
    client = connect()
    try:
        gpu_count = int(run(client, "nvidia-smi -L | wc -l").strip())
        if gpu_count != 3:
            raise RuntimeError(f"expected exactly 3 GPUs, found {gpu_count}")
        destination = f"{ROOT}/src/CLEAR-HUG/experiments/{EXPERIMENT}"
        sftp = client.open_sftp()
        for local in (HERE / "run_worker.py", HERE / "deploy_and_start.py", HERE / "README.md"):
            upload(sftp, local, f"{destination}/{local.name}")
        for relative in (
            "ptbxl/super_class/ptbxl_super_class_train.csv", "ptbxl/super_class/ptbxl_super_class_val.csv", "ptbxl/super_class/ptbxl_super_class_test.csv",
            "ptbxl/sub_class/ptbxl_sub_class_train.csv", "ptbxl/sub_class/ptbxl_sub_class_val.csv", "ptbxl/sub_class/ptbxl_sub_class_test.csv",
            "ptbxl/form/ptbxl_form_train.csv", "ptbxl/form/ptbxl_form_val.csv", "ptbxl/form/ptbxl_form_test.csv",
            "ptbxl/rhythm/ptbxl_rhythm_train.csv", "ptbxl/rhythm/ptbxl_rhythm_val.csv", "ptbxl/rhythm/ptbxl_rhythm_test.csv",
            "chapman/chapman_train.csv", "chapman/chapman_val.csv", "chapman/chapman_test.csv",
        ):
            upload(sftp, OFFICIAL / relative, f"{destination}/data_split/{relative}")
        sftp.close()
        result = f"{ROOT}/results/{CAMPAIGN}"
        python = f"{ROOT}/.venv/bin/python"; script = f"{destination}/run_worker.py"
        run(client, f"mkdir -p {shlex.quote(result)}")
        for gpu in range(3):
            pidfile = f"{result}/worker-{gpu}.pid"; logfile = f"{result}/worker-{gpu}.log"
            command = (
                f"if ! test -s {shlex.quote(pidfile)} || ! kill -0 \"$(cat {shlex.quote(pidfile)})\" 2>/dev/null; then "
                f"nohup setsid env CUDA_VISIBLE_DEVICES={gpu} LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 "
                f"{shlex.quote(python)} {shlex.quote(script)} --root {shlex.quote(ROOT)} --gpu {gpu} "
                f"> {shlex.quote(logfile)} 2>&1 < /dev/null & echo $! > {shlex.quote(pidfile)}; fi"
            )
            _, stdout, _ = client.exec_command(command); time.sleep(1); stdout.channel.close()
        print(run(client, f"for g in 0 1 2; do echo GPU=$g PID=$(cat {result}/worker-$g.pid); done; "
                          "nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
