#!/usr/bin/env python3
"""Wait for MERL on 204, then run the queued ECG-FIX CLOCS campaign."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


CAMPAIGN = "q1-ecgfix-clocs-six-task-three-fraction-seeds42-46-55-20260916"
MERL_CAMPAIGN = "q1-merl-five-task-three-seed-20260916"
SEEDS = (42, 46, 55)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def merl_gate(root: Path) -> tuple[str, dict]:
    output = root / "results" / MERL_CAMPAIGN
    workers = [read_json(output / f"worker-{gpu}-status.json") for gpu in range(3)]
    states = [value.get("state", "missing") if value else "missing" for value in workers]
    complete_files = list(output.glob("*/seed*/*pct/complete.json"))
    details = {"worker_states": states, "valid_complete_files": len(complete_files), "required": 45}
    if "failed" in states:
        return "blocked", details
    if states == ["complete", "complete", "complete"] and len(complete_files) == 45:
        return "open", details
    return "waiting", details


def run_checked(command: list[str], log_path: Path, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\nCOMMAND " + " ".join(command) + "\n")
        stream.flush()
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if completed.returncode:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "results" / CAMPAIGN
    status_path = output / "queue-status.json"
    runner = root / "src/CLEAR-HUG/experiments/q1_ecgfix_clocs_six_task_three_seed_20260916/run_campaign.py"
    python = root / "envs/ecg-fix/bin/python"
    output.mkdir(parents=True, exist_ok=True)

    while True:
        state, details = merl_gate(root)
        atomic_json(status_path, {"state": state, "reason": "waiting for MERL 45/45 gate", **details})
        if state == "blocked":
            return
        if state == "open":
            break
        time.sleep(args.poll_seconds)

    try:
        if not (output / "prepare-complete.json").exists():
            atomic_json(status_path, {"state": "preparing", **details})
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            run_checked([str(python), str(runner), "--root", str(root), "--phase", "prepare"],
                        output / "prepare.log", env)

        atomic_json(status_path, {"state": "evaluating", "completed_seeds": 0,
                                  "total_seeds": len(SEEDS), "total_units": 54})
        processes = []
        for seed in SEEDS:
            if (output / f"seed-{seed}-complete.json").exists():
                continue
            log = (output / f"seed-{seed}.log").open("a", encoding="utf-8")
            command = [str(python), str(runner), "--root", str(root), "--phase", "eval", "--seed", str(seed)]
            processes.append((seed, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
        failures = []
        for seed, process, log in processes:
            code = process.wait()
            log.close()
            if code:
                failures.append({"seed": seed, "exit_code": code})
        completed = sum((output / f"seed-{seed}-complete.json").exists() for seed in SEEDS)
        if failures or completed != len(SEEDS):
            raise RuntimeError(f"CLOCS seed evaluation failure: {failures}; completed={completed}/3")
        complete = {"state": "complete", "completed_seeds": completed, "total_seeds": 3,
                    "completed_units": 54, "total_units": 54}
        atomic_json(output / "campaign-complete.json", complete)
        atomic_json(status_path, complete)
    except Exception as error:
        atomic_json(status_path, {"state": "failed", "error": repr(error),
                                  "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
