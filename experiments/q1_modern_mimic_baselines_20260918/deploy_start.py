#!/usr/bin/env python3
"""Resumably deploy and start the three-model campaign on server 204."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path, PurePosixPath

import paramiko


ROOT = "/root/107552503710-1"
EXPERIMENT = "q1_modern_mimic_baselines_20260918"
CAMPAIGN = "q1-modern-mimic-baselines-six-task-three-fraction-seeds42-46-55-20260918"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(__file__).resolve().parents[4]
WEIGHTS = WORKSPACE / "transfer/q1-modern-ecg-baselines-20260918/weights"
TOLERANT_SOURCE = WORKSPACE / "external_models_staging/TolerantECG/src/models/ecg_encoder/convnext.py"
EXPECTED_SIZES = {
    "dbeta_best.pt": 3_956_750_922,
    "dbeta_config.json": 1_687,
    "ked.pt": 1_341_207_980,
    "TolerantECG_encoder.pth": 107_317_824,
}
EXPECTED_MD5 = {"ked.pt": "f5d5711b4fd52f41ac2e3c9eb10e4650"}


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


def run(client: paramiko.SSHClient, command: str, check: bool = True):
    _, stdout, stderr = client.exec_command(command)
    output = stdout.read().decode(errors="replace"); error = stderr.read().decode(errors="replace")
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


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def upload_resumable(sftp: paramiko.SFTPClient, local: Path, remote: str, mode: int = 0o444) -> None:
    incoming = remote + ".incoming"
    expected = local.stat().st_size
    try:
        existing = sftp.stat(remote)
        if existing.st_size == expected:
            return
    except FileNotFoundError:
        pass
    try:
        offset = sftp.stat(incoming).st_size
    except FileNotFoundError:
        offset = 0
    if offset > expected:
        sftp.remove(incoming); offset = 0
    with local.open("rb") as source:
        source.seek(offset)
        with sftp.open(incoming, "ab" if offset else "wb") as target:
            while block := source.read(8 * 1024 * 1024):
                target.write(block)
    if sftp.stat(incoming).st_size != expected:
        raise RuntimeError(f"incomplete upload for {local}")
    sftp.chmod(incoming, mode); sftp.posix_rename(incoming, remote)


def verify_local_inputs() -> dict:
    result = {}
    for name, size in EXPECTED_SIZES.items():
        path = WEIGHTS / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"missing/incomplete local weight {path}: expected {size} bytes")
        result[name] = {"bytes": size, "sha256": digest(path, "sha256")}
        if name in EXPECTED_MD5 and digest(path, "md5") != EXPECTED_MD5[name]:
            raise RuntimeError(f"official checksum mismatch for {path}")
    if not TOLERANT_SOURCE.is_file():
        raise FileNotFoundError(TOLERANT_SOURCE)
    return result


def main() -> None:
    hashes = verify_local_inputs(); client = connect()
    try:
        destination = f"{ROOT}/src/CLEAR-HUG/experiments/{EXPERIMENT}"
        remote_weights = f"{ROOT}/model_weights/modern-mimic-baselines"
        sftp = client.open_sftp(); mkdirs(sftp, destination); mkdirs(sftp, destination + "/vendor")
        mkdirs(sftp, remote_weights)
        for name in ("prepare_embeddings.py", "run_probe.py", "worker.py", "queue_runner.py",
                     "deploy_start.py", "README.md"):
            upload_resumable(sftp, HERE / name, f"{destination}/{name}",
                             0o555 if name.endswith(".py") else 0o444)
        upload_resumable(sftp, TOLERANT_SOURCE, destination + "/vendor/tolerant_convnext.py")
        for name in EXPECTED_SIZES:
            upload_resumable(sftp, WEIGHTS / name, f"{remote_weights}/{name}")
        sftp.close()
        python = f"{ROOT}/envs/ecg-fix/bin/python"
        run(client, f"{python} -m py_compile {destination}/*.py {destination}/vendor/*.py")
        for name, value in hashes.items():
            _, remote_hash, _ = run(client, f"sha256sum {remote_weights}/{name}")
            if remote_hash.split()[0] != value["sha256"]:
                raise RuntimeError(f"remote SHA256 mismatch for {name}")
        campaign = f"{ROOT}/results/{CAMPAIGN}"; run(client, f"mkdir -p {campaign}")
        manifest = {
            "state": "deployed", "root": ROOT, "campaign": CAMPAIGN,
            "models": ["KED", "D_BETA", "TolerantECG"], "weights": hashes,
            "tolerant_source_commit": "f38ad823da11da4336d8a057cc5ff2c6f1693da8",
            "tolerant_source_sha256": digest(TOLERANT_SOURCE, "sha256"),
        }
        incoming = HERE / "deployment-manifest.local.json"
        incoming.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sftp = client.open_sftp(); upload_resumable(sftp, incoming, campaign + "/deployment-manifest.json")
        sftp.close(); incoming.unlink(missing_ok=True)
        _, cuda, _ = run(client, f"{python} - <<'PY'\nimport torch\nprint(torch.cuda.is_available(), torch.cuda.device_count())\nPY")
        if cuda.strip() != "True 3":
            run(client, f"printf '%s\\n' '{{\"state\":\"waiting_for_gpu\",\"required_gpus\":3}}' > {campaign}/queue-status.json")
            print("DEPLOYED_WAITING_FOR_3_GPUS"); return
        _, active, _ = run(client, f"pgrep -af '{destination}/queue_runner.py'", check=False)
        if active.strip():
            print("already-running\n" + active); return
        command = (f"nohup setsid {python} {destination}/queue_runner.py --root {ROOT} "
                   f"> {campaign}/queue.log 2>&1 < /dev/null & echo $! > {campaign}/queue.pid")
        run(client, command); time.sleep(5)
        _, state, _ = run(client, f"cat {campaign}/queue-status.json 2>/dev/null || true; "
                                    f"ps -ef | grep '{EXPERIMENT}' | grep -v grep; "
                                    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader",
                          check=False)
        print(state)
    finally:
        client.close()


if __name__ == "__main__":
    main()
