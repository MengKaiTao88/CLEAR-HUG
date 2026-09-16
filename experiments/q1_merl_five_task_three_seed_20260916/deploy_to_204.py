#!/usr/bin/env python3
"""Migrate missing immutable inputs from 172 and launch three workers on 204."""
from __future__ import annotations

import hashlib
import shlex
import stat
import time
from pathlib import Path, PurePosixPath

import paramiko


CAMPAIGN = "q1-merl-five-task-three-seed-20260916"
SOURCE_ROOT = "/root/107552503710"
ROOT = "/root/107552503710-1"
EXPERIMENT = "q1_merl_five_task_three_seed_20260916"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(__file__).resolve().parents[4]
OFFICIAL = WORKSPACE / "external_models_staging" / "MERL-ICML2024" / "finetune" / "data_split"
ENCODER_SHA256 = "38ba669c2cc319670c4172d8c292f123e86b8e7106b1a68bb7e10dd89f09daf5"


def env(name: str) -> dict[str, str]:
    values = {}
    for raw in (WORKSPACE / name).read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
            key, value = raw.split("=", 1); values[key.strip()] = value.strip()
    return values


def connect(name: str) -> paramiko.SSHClient:
    config = env(name); client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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


def remote_sha(client: paramiko.SSHClient, path: str) -> str | None:
    value = run(client, f"if test -f {shlex.quote(path)}; then sha256sum {shlex.quote(path)} | cut -d' ' -f1; fi")
    return value.strip() or None


def relay(source: paramiko.SSHClient, destination: paramiko.SSHClient,
          source_path: str, destination_path: str, expected_sha: str) -> None:
    if remote_sha(destination, destination_path) == expected_sha:
        return
    if remote_sha(destination, destination_path) is not None:
        raise RuntimeError(f"refusing mismatched destination file {destination_path}")
    source_sftp = source.open_sftp(); destination_sftp = destination.open_sftp()
    mkdirs(destination_sftp, str(PurePosixPath(destination_path).parent))
    incoming = destination_path + ".incoming"
    size = source_sftp.stat(source_path).st_size
    try:
        try: transferred = destination_sftp.stat(incoming).st_size
        except FileNotFoundError: transferred = 0
        if transferred > size:
            destination_sftp.remove(incoming); transferred = 0
        # SFTP write acknowledgements are extremely slow across the two public
        # SSH endpoints.  Pump an authenticated stdout/stdin stream instead;
        # the existing .incoming byte count remains the resume boundary.
        start = transferred + 1
        source_command = f"tail -c +{start} {shlex.quote(source_path)}"
        destination_command = f"cat >> {shlex.quote(incoming)}"
        _, reader, source_error = source.exec_command(source_command)
        writer, destination_output, destination_error = destination.exec_command(destination_command)
        while True:
            block = reader.read(8 * 1024 * 1024)
            if not block: break
            writer.write(block); transferred += len(block)
            if transferred % (256 * 1024 * 1024) < len(block):
                print(f"relay {PurePosixPath(source_path).name}: {transferred}/{size}", flush=True)
        writer.channel.shutdown_write()
        source_code = reader.channel.recv_exit_status()
        destination_code = destination_output.channel.recv_exit_status()
        if source_code or destination_code:
            raise RuntimeError(
                f"stream relay failed source={source_code} destination={destination_code}: "
                f"{source_error.read().decode(errors='replace')} "
                f"{destination_error.read().decode(errors='replace')}"
            )
        destination_sftp.chmod(incoming, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        destination_sftp.posix_rename(incoming, destination_path)
    finally:
        source_sftp.close(); destination_sftp.close()
    if remote_sha(destination, destination_path) != expected_sha:
        raise RuntimeError(f"hash verification failed for {destination_path}")


def main() -> None:
    source = connect(".env.remote"); destination = connect(".env.remote.secondary")
    try:
        if int(run(destination, "nvidia-smi -L | wc -l").strip()) != 3:
            raise RuntimeError("204 does not expose exactly three GPUs")
        archives = (
            ("merl-ptbxl-records500.tar.gz", "PTB"),
            ("merl-csn-wfdbrecords.tar.gz", "CSN"),
        )
        for archive, _ in archives:
            source_path = f"{SOURCE_ROOT}/deployment_logs/{archive}"
            digest = remote_sha(source, source_path)
            if not digest:
                raise RuntimeError(f"source archive is not ready: {archive}")
            relay(source, destination, source_path, f"{ROOT}/deployment_logs/{archive}", digest)
        relay(source, destination, f"{SOURCE_ROOT}/model_weights/MERL/res18_best_encoder.pth",
              f"{ROOT}/model_weights/MERL/res18_best_encoder.pth", ENCODER_SHA256)

        ptb_parent = f"{ROOT}/data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
        csn_parent = f"{ROOT}/src/CLEAR-HUG/datasets/dataset_preprocess/CSN"
        run(destination,
            f"if ! test -d {ptb_parent}/records500; then rm -rf {ptb_parent}/.merl-records500-incoming; "
            f"mkdir -p {ptb_parent}/.merl-records500-incoming; tar -xzf {ROOT}/deployment_logs/merl-ptbxl-records500.tar.gz "
            f"-C {ptb_parent}/.merl-records500-incoming; mv {ptb_parent}/.merl-records500-incoming/records500 {ptb_parent}/records500; "
            f"rmdir {ptb_parent}/.merl-records500-incoming; fi; "
            f"if ! test -d {csn_parent}/WFDBRecords; then rm -rf {csn_parent}/.merl-wfdb-incoming; "
            f"mkdir -p {csn_parent}/.merl-wfdb-incoming; tar -xzf {ROOT}/deployment_logs/merl-csn-wfdbrecords.tar.gz "
            f"-C {csn_parent}/.merl-wfdb-incoming; mv {csn_parent}/.merl-wfdb-incoming/WFDBRecords {csn_parent}/WFDBRecords; "
            f"rmdir {csn_parent}/.merl-wfdb-incoming; fi")
        counts = run(destination,
                     f"find {ptb_parent}/records500 -type f | wc -l; find {csn_parent}/WFDBRecords -type f | wc -l").splitlines()
        if counts != ["43598", "23026"]:
            raise RuntimeError(f"migrated raw file counts mismatch: {counts}")

        destination_path = f"{ROOT}/src/CLEAR-HUG/experiments/{EXPERIMENT}"
        sftp = destination.open_sftp()
        for local in (HERE / "run_worker.py", HERE / "deploy_to_204.py", HERE / "README.md"):
            upload(sftp, local, f"{destination_path}/{local.name}")
        for relative in (
            "ptbxl/super_class/ptbxl_super_class_train.csv", "ptbxl/super_class/ptbxl_super_class_val.csv", "ptbxl/super_class/ptbxl_super_class_test.csv",
            "ptbxl/sub_class/ptbxl_sub_class_train.csv", "ptbxl/sub_class/ptbxl_sub_class_val.csv", "ptbxl/sub_class/ptbxl_sub_class_test.csv",
            "ptbxl/form/ptbxl_form_train.csv", "ptbxl/form/ptbxl_form_val.csv", "ptbxl/form/ptbxl_form_test.csv",
            "ptbxl/rhythm/ptbxl_rhythm_train.csv", "ptbxl/rhythm/ptbxl_rhythm_val.csv", "ptbxl/rhythm/ptbxl_rhythm_test.csv",
            "chapman/chapman_train.csv", "chapman/chapman_val.csv", "chapman/chapman_test.csv",
        ):
            upload(sftp, OFFICIAL / relative, f"{destination_path}/data_split/{relative}")
        sftp.close()

        result = f"{ROOT}/results/{CAMPAIGN}"; python = f"{ROOT}/.venv/bin/python"
        script = f"{destination_path}/run_worker.py"; run(destination, f"mkdir -p {result}")
        for gpu in range(3):
            pidfile = f"{result}/worker-{gpu}.pid"; logfile = f"{result}/worker-{gpu}.log"
            command = (
                f"if ! test -s {pidfile} || ! kill -0 \"$(cat {pidfile})\" 2>/dev/null; then "
                f"nohup setsid env CUDA_VISIBLE_DEVICES={gpu} {python} {script} --root {ROOT} --gpu {gpu} "
                f"> {logfile} 2>&1 < /dev/null & echo $! > {pidfile}; fi"
            )
            _, stdout, _ = destination.exec_command(command); time.sleep(1); stdout.channel.close()
        print(run(destination, f"for g in 0 1 2; do echo GPU=$g PID=$(cat {result}/worker-$g.pid); done; "
                              "nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader"))
    finally:
        source.close(); destination.close()


if __name__ == "__main__":
    main()
