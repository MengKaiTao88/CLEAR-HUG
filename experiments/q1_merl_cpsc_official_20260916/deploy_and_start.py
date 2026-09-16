#!/usr/bin/env python3
"""Deploy the isolated official MERL CPSC campaign to node 10110."""
from __future__ import annotations

import hashlib
import os
import shlex
import time
from pathlib import Path, PurePosixPath

import paramiko


CAMPAIGN = "q1-merl-cpsc-official-20260916"
ROOT = "/root/107552503710"
EXPERIMENT = "q1_merl_cpsc_official_20260916"
ENCODER_SHA256 = "38ba669c2cc319670c4172d8c292f123e86b8e7106b1a68bb7e10dd89f09daf5"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(__file__).resolve().parents[4]
WEIGHT = WORKSPACE / "模型权重" / "MERL" / "res18_best_encoder.pth"
OFFICIAL = WORKSPACE / "external_models_staging" / "MERL-ICML2024"


def environment() -> dict[str, str]:
    values = {}
    for raw in (WORKSPACE / ".env.remote").read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
            key, value = raw.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def connect() -> paramiko.SSHClient:
    config = environment()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        config["CODEX_REMOTE_HOST"],
        port=int(config["CODEX_REMOTE_PORT"]),
        username=config.get("CODEX_REMOTE_USER", "root"),
        password=config["CODEX_REMOTE_PASSWORD"],
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def upload_atomic(sftp: paramiko.SFTPClient, local: Path, remote: str, mode: int = 0o444) -> None:
    mkdirs(sftp, str(PurePosixPath(remote).parent))
    incoming = remote + ".incoming"
    sftp.put(str(local), incoming)
    sftp.chmod(incoming, mode)
    sftp.posix_rename(incoming, remote)


def main() -> None:
    if digest(WEIGHT) != ENCODER_SHA256:
        raise RuntimeError("local official MERL encoder hash mismatch")
    client = connect()
    try:
        if "NVIDIA" not in run(client, "nvidia-smi --query-gpu=name --format=csv,noheader"):
            raise RuntimeError("GPU unavailable")
        sftp = client.open_sftp()
        destination = f"{ROOT}/src/CLEAR-HUG/experiments/{EXPERIMENT}"
        for local in (HERE / "run_merl_cpsc.py", HERE / "deploy_and_start.py", HERE / "README.md"):
            upload_atomic(sftp, local, f"{destination}/{local.name}", 0o444)
        for split in ("train", "val", "test"):
            local = OFFICIAL / "finetune" / "data_split" / "icbeb" / f"icbeb_{split}.csv"
            upload_atomic(sftp, local, f"{destination}/data_split/icbeb_{split}.csv", 0o444)
        remote_weight = f"{ROOT}/model_weights/MERL/res18_best_encoder.pth"
        mkdirs(sftp, str(PurePosixPath(remote_weight).parent))
        sftp.close()
        existing = run(client, f"if test -f {shlex.quote(remote_weight)}; then sha256sum {shlex.quote(remote_weight)} | cut -d' ' -f1; fi").strip()
        if existing and existing != ENCODER_SHA256:
            raise RuntimeError("refusing to overwrite mismatched remote MERL encoder")
        if not existing:
            sftp = client.open_sftp()
            upload_atomic(sftp, WEIGHT, remote_weight, 0o444)
            sftp.close()
        verified = run(client, f"sha256sum {shlex.quote(remote_weight)} | cut -d' ' -f1").strip()
        if verified != ENCODER_SHA256:
            raise RuntimeError("remote MERL encoder verification failed")
        result = f"{ROOT}/results/{CAMPAIGN}"
        python = f"{ROOT}/.venv/bin/python"
        script = f"{destination}/run_merl_cpsc.py"
        pattern = f"{script} --root {ROOT}"
        pidfile = result + "/run.pid"
        command = (
            f"mkdir -p {shlex.quote(result)}; "
            f"if ! test -s {shlex.quote(pidfile)} || ! kill -0 \"$(cat {shlex.quote(pidfile)})\" 2>/dev/null; then "
            f"nohup setsid env LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 "
            f"{shlex.quote(python)} {shlex.quote(script)} --root {shlex.quote(ROOT)} "
            f"> {shlex.quote(result + '/run.log')} 2>&1 < /dev/null & "
            f"echo $! > {shlex.quote(pidfile)}; fi"
        )
        _, stdout, _ = client.exec_command(command)
        time.sleep(2)
        stdout.channel.close()
        print(run(client, f"pgrep -af {shlex.quote(pattern)} || true; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; tail -30 {shlex.quote(result + '/run.log')} 2>/dev/null || true"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
