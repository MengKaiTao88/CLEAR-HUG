#!/usr/bin/env python3
"""Read-only three-node progress monitor for the frozen low-label campaign."""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path

import paramiko


WORKSPACE = Path(__file__).resolve().parents[4]
CAMPAIGN = "q1-six-task-low-label-10seed-20260909"
NODES = (
    ("10110", ".env.remote", "10.109.118.172", 10110, "/root/107552503710"),
    ("10092", ".env.remote.secondary", "10.109.118.204", 10092, "/root/107552503710-1"),
    ("10103", ".env.remote.tertiary", "10.109.118.204", 10103, "/root/107552503710-2"),
)


def env_file(name: str) -> dict[str, str]:
    values = {}
    for raw in (WORKSPACE / name).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_json(sftp: paramiko.SFTPClient, path: str) -> dict:
    with sftp.open(path, "r") as handle:
        return json.loads(handle.read().decode())


def inspect(node: tuple[str, str, str, int, str]) -> dict:
    name, env_name, host, port, root = node
    config = env_file(env_name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=config.get("CODEX_REMOTE_USER", "root"),
                   password=config["CODEX_REMOTE_PASSWORD"], timeout=20,
                   banner_timeout=20, auth_timeout=20)
    out = f"{root}/results/{CAMPAIGN}"
    command = (
        f"find {out}/1pct -mindepth 2 -maxdepth 2 -name train-val-complete.json -type f 2>/dev/null | wc -l; "
        f"find {out}/10pct -mindepth 2 -maxdepth 2 -name train-val-complete.json -type f 2>/dev/null | wc -l; "
        f"find {out} -type d -name '*.failed-*' -prune -o -name training-complete.json -type f -print 2>/dev/null | wc -l; "
        f"find {out} -type d -name '*.failed-*' -prune -o -name baseline-complete.json -type f -print 2>/dev/null | wc -l; "
        f"find {out} -type d -name '*.failed-*' -prune -o -path '*/hilar-training/parameter-matched-direct/complete.json' -type f -print 2>/dev/null | wc -l; "
        f"find {out} -path '*/formal-test/complete.json' -type f 2>/dev/null | wc -l; "
        "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader"
    )
    _, stdout, stderr = client.exec_command(command)
    lines = stdout.read().decode().strip().splitlines()
    error = stderr.read().decode().strip()
    if error or len(lines) != 7:
        raise RuntimeError(f"{name}: {error or lines}")
    sftp = client.open_sftp()
    try:
        queue = read_json(sftp, f"{out}/{name}-queue-status.json")
        if queue.get("spec"):
            fraction, task, seed = queue["spec"].split(":")
            run = f"{out}/{fraction}/{task}-seed{seed}"
            status = read_json(sftp, f"{run}/status.json")
            model = "clear-hug" if status["stage"] == "hug-training" else "hila"
            try:
                with sftp.open(f"{run}/{model}/log.txt", "r") as handle:
                    rows = [json.loads(x) for x in handle.read().decode().splitlines() if x.strip()]
            except FileNotFoundError:
                rows = []
            best = max(rows, key=lambda row: float(row["val_roc_auc"])) if rows else None
            with sftp.open(f"{run}/train-val.log", "r") as handle:
                current_traceback = "Traceback (most recent call last)" in handle.read().decode()
        else:
            status = {"state": queue["state"], "stage": None}
            model = None
            rows = []
            best = None
            current_traceback = False
    finally:
        sftp.close()
        client.close()
    return {
        "node": name, "queue": queue, "status": status,
        "units_1pct": int(lines[0]), "units_10pct": int(lines[1]),
        "hug_checkpoints": int(lines[2]), "hila_checkpoints": int(lines[3]),
        "hilar_checkpoints": int(lines[4]), "formal_tests": int(lines[5]),
        "gpu": lines[6], "model": model,
        "current_epoch": int(rows[-1]["epoch"]) if rows else None,
        "best_epoch": int(best["epoch"]) if best else None,
        "best_auroc": float(best["val_roc_auc"]) if best else None,
        "best_auprc": float(best["val_pr_auc"]) if best else None,
        "current_traceback": current_traceback,
    }


def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(inspect, node): node[0] for node in NODES}
        rows = []
        for future, node in futures.items():
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"node": node, "error": f"{type(exc).__name__}: {exc}"})
    healthy = [row for row in rows if "error" not in row]
    result = {
        "nodes": rows,
        "totals": {
            "reporting_nodes": len(healthy),
            "units_1pct": sum(x["units_1pct"] for x in healthy),
            "units_10pct": sum(x["units_10pct"] for x in healthy),
            "hug_checkpoints": sum(x["hug_checkpoints"] for x in healthy),
            "hila_checkpoints": sum(x["hila_checkpoints"] for x in healthy),
            "hilar_checkpoints": sum(x["hilar_checkpoints"] for x in healthy),
            "formal_tests": sum(x["formal_tests"] for x in healthy),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
