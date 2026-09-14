#!/usr/bin/env python3
"""Deploy immutable inputs and official encoders, then start node canaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import sys
from pathlib import Path, PurePosixPath

import paramiko

from protocol import CAMPAIGN, HEARTLANG_SHA256, NODE_TASKS, NODES, STMEM_SHA256, TASKS


CODE_DIRNAME = "q1_heartlang_six_task_three_fraction_seeds42_46_55_20260913"
LOCAL_CODE = Path(__file__).resolve().parent
WORKSPACE = Path(__file__).resolve().parents[4]
SOURCE_NODE = "10110"


def env_file(name: str) -> dict[str, str]:
    values = {}
    for raw in (WORKSPACE / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def connect(node: str) -> paramiko.SSHClient:
    env_name, host, port, _ = NODES[node]
    cfg = env_file(env_name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=cfg.get("CODEX_REMOTE_USER", "root"),
                   password=cfg["CODEX_REMOTE_PASSWORD"], timeout=30,
                   banner_timeout=30, auth_timeout=30)
    return client


def run(client: paramiko.SSHClient, command: str) -> str:
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode(errors="replace")
    error = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"remote command failed ({code}): {command}\n{error}\n{output}")
    return output


def remote_sha(client: paramiko.SSHClient, path: str) -> str | None:
    output = run(client, f"if test -f {shlex.quote(path)}; then sha256sum {shlex.quote(path)} | cut -d' ' -f1; fi")
    return output.strip() or None


def mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    current = PurePosixPath("/")
    for part in PurePosixPath(path).parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current))


def upload_code(client: paramiko.SSHClient, root: str) -> None:
    sftp = client.open_sftp()
    destination = f"{root}/src/CLEAR-HUG/experiments/{CODE_DIRNAME}"
    mkdirs(sftp, destination)
    for local in sorted(LOCAL_CODE.glob("*.py")) + sorted(LOCAL_CODE.glob("*.md")):
        remote = f"{destination}/{local.name}"
        incoming = remote + ".incoming"
        sftp.put(str(local), incoming)
        sftp.chmod(incoming, 0o644)
        sftp.posix_rename(incoming, remote)
    sftp.close()


def relay_file(source: paramiko.SSHClient, destination: paramiko.SSHClient,
               source_path: str, destination_path: str, expected_sha: str | None = None) -> str:
    source_sftp = source.open_sftp()
    destination_sftp = destination.open_sftp()
    mkdirs(destination_sftp, str(PurePosixPath(destination_path).parent))
    incoming = destination_path + ".incoming"
    try:
        size = source_sftp.stat(source_path).st_size
        digest = hashlib.sha256()
        transferred = 0
        with source_sftp.open(source_path, "rb") as reader, destination_sftp.open(incoming, "wb") as writer:
            while True:
                block = reader.read(8 * 1024 * 1024)
                if not block:
                    break
                writer.write(block)
                digest.update(block)
                transferred += len(block)
                if transferred % (512 * 1024 * 1024) < len(block):
                    print(f"relay {source_path}: {transferred}/{size}", flush=True)
        actual = digest.hexdigest()
        if expected_sha and actual != expected_sha:
            raise RuntimeError(f"source hash mismatch for {source_path}: {actual}")
        destination_sftp.chmod(incoming, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        destination_sftp.posix_rename(incoming, destination_path)
        return actual
    finally:
        source_sftp.close()
        destination_sftp.close()


def ensure_relay(source: paramiko.SSHClient, destination: paramiko.SSHClient,
                 source_path: str, destination_path: str, expected_sha: str) -> None:
    if remote_sha(destination, destination_path) == expected_sha:
        return
    if remote_sha(destination, destination_path) is not None:
        raise RuntimeError(f"refusing mismatched existing file {destination_path}")
    actual = relay_file(source, destination, source_path, destination_path, expected_sha)
    if remote_sha(destination, destination_path) != actual:
        raise RuntimeError(f"destination hash mismatch for {destination_path}")


def deploy_models(source: paramiko.SSHClient, destination: paramiko.SSHClient, root: str) -> None:
    source_root = NODES[SOURCE_NODE][3]
    archives = (
        ("HeartLang-clean.tar.gz", "819751dfe74cbc367d0290b11a0305e119fb8e5992eaf7dc901d8ffb04371391", "HeartLang"),
        ("ST-MEM-clean.tar.gz", "c3749c1ca37ce005f9737299fffd7471b46f7d2613ef2dfe1c2171c4c6fd515e", "ST-MEM"),
    )
    for archive, digest, directory in archives:
        target = f"{root}/deployment_logs/{archive}"
        ensure_relay(source, destination, f"{source_root}/deployment_logs/{archive}", target, digest)
        run(destination, f"mkdir -p {shlex.quote(root + '/external_models')} && "
                         f"if ! test -d {shlex.quote(root + '/external_models/' + directory)}; then "
                         f"tar -xzf {shlex.quote(target)} -C {shlex.quote(root + '/external_models')}; fi")
    weights = (
        ("HeartLang/checkpoint-200.pth", HEARTLANG_SHA256),
        ("ST-MEM/st_mem_vit_base_encoder.pth", STMEM_SHA256),
    )
    for relative, digest in weights:
        ensure_relay(source, destination, f"{source_root}/model_weights/{relative}",
                     f"{root}/model_weights/{relative}", digest)


def list_source_files(sftp: paramiko.SFTPClient, source_dir: str):
    return sorted(item.filename for item in sftp.listdir_attr(source_dir)
                  if stat.S_ISREG(item.st_mode) and not item.filename.startswith("."))


def deploy_raw_task(source: paramiko.SSHClient, destination: paramiko.SSHClient,
                    node: str, task: str) -> dict[str, str]:
    source_root = NODES[SOURCE_NODE][3]
    destination_root = NODES[node][3]
    _, _, raw_rel, _ = TASKS[task]
    canonical = f"{source_root}/src/CLEAR-HUG/datasets/ecg_datasets/{raw_rel}"
    local_source = f"{destination_root}/src/CLEAR-HUG/datasets/ecg_datasets/{raw_rel}"
    snapshot = f"{destination_root}/campaign-inputs/{CAMPAIGN}/raw/{raw_rel}"
    source_sftp = source.open_sftp()
    files = list_source_files(source_sftp, canonical)
    source_sftp.close()
    hashes = {}
    run(destination, f"mkdir -p {shlex.quote(snapshot)}")
    for filename in files:
        canonical_path = f"{canonical}/{filename}"
        target_path = f"{snapshot}/{filename}"
        expected = remote_sha(source, canonical_path)
        if expected is None:
            raise RuntimeError(f"missing canonical input {canonical_path}")
        hashes[filename] = expected
        existing = remote_sha(destination, target_path)
        if existing == expected:
            continue
        if existing is not None:
            raise RuntimeError(f"refusing mismatched snapshot file {target_path}")
        if remote_sha(destination, f"{local_source}/{filename}") == expected:
            run(destination, f"cp --reflink=auto {shlex.quote(local_source + '/' + filename)} "
                             f"{shlex.quote(target_path + '.incoming')} && chmod 444 "
                             f"{shlex.quote(target_path + '.incoming')} && mv "
                             f"{shlex.quote(target_path + '.incoming')} {shlex.quote(target_path)}")
        else:
            relay_file(source, destination, canonical_path, target_path, expected)
        if remote_sha(destination, target_path) != expected:
            raise RuntimeError(f"snapshot verification failed {target_path}")
    return hashes


def smoke(client: paramiko.SSHClient, root: str) -> None:
    python = f"{root}/.venv/bin/python"
    code = (
        "import sys,torch; "
        f"sys.path.insert(0,'{root}/external_models/HeartLang'); import modeling_pretrain; "
        f"sys.path.insert(0,'{root}/external_models/ST-MEM'); from models.encoder.st_mem_vit import st_mem_vit_base; "
        "print(torch.__version__,torch.cuda.is_available(),modeling_pretrain.HeartLang(pretrained=False).__class__.__name__,"
        "st_mem_vit_base(12,None,2250,75).__class__.__name__)"
    )
    print(run(client, f'{python} -c "{code}"').strip(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-canaries", action="store_true")
    args = parser.parse_args()
    clients = {node: connect(node) for node in NODES}
    source = clients[SOURCE_NODE]
    manifest = {"campaign": CAMPAIGN, "nodes": {}}
    try:
        for node, client in clients.items():
            root = NODES[node][3]
            print(f"deploy code/models {node}", flush=True)
            upload_code(client, root)
            if node != SOURCE_NODE:
                deploy_models(source, client, root)
            hashes = {}
            for task in NODE_TASKS[node]:
                print(f"snapshot {node}:{task}", flush=True)
                hashes[task] = deploy_raw_task(source, client, node, task)
            smoke(client, root)
            manifest["nodes"][node] = {"root": root, "tasks": list(NODE_TASKS[node]), "raw_sha256": hashes}
        local_manifest = LOCAL_CODE / "deployment-manifest.json"
        local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.start_canaries:
            for node, client in clients.items():
                root = NODES[node][3]
                python = f"{root}/.venv/bin/python"
                code = f"{root}/src/CLEAR-HUG/experiments/{CODE_DIRNAME}/run_node.py"
                result = f"{root}/results/{CAMPAIGN}"
                preload = "env LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 " if node == "10110" else ""
                command = (f"mkdir -p {shlex.quote(result)} && "
                           f"nohup {preload}{python} {code} --root {root} --node {node} --mode canary "
                           f"> {result}/{node}-canary.log 2>&1 < /dev/null & echo $!")
                print(f"START {node} pid={run(client, command).strip()}", flush=True)
    finally:
        for client in clients.values():
            client.close()


if __name__ == "__main__":
    main()
