#!/usr/bin/env python3
"""Run one PyTorch linear-probe seed on each of three GPUs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from run_probe import CAMPAIGN, SEEDS, atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "results" / CAMPAIGN
    output.mkdir(parents=True, exist_ok=True)
    status = output / "queue-status.json"
    runner = Path(__file__).with_name("run_probe.py")
    if not __import__("torch").cuda.is_available() or __import__("torch").cuda.device_count() < 3:
        atomic_json(status, {"state": "waiting_for_gpu", "required_gpus": 3,
                             "visible_gpus": __import__("torch").cuda.device_count()})
        raise RuntimeError("three visible CUDA GPUs are required")
    processes = []
    try:
        atomic_json(status, {"state": "running", "completed_units": 0, "total_units": 54})
        for gpu, seed in enumerate(SEEDS):
            if (output / f"seed-{seed}-complete.json").is_file():
                continue
            log = (output / f"seed-{seed}.log").open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [sys.executable, str(runner), "--root", str(root), "--seed", str(seed),
                       "--device", "cuda:0"]
            processes.append((seed, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                       env=env), log))
        failures = []
        for seed, process, log in processes:
            code = process.wait()
            log.close()
            if code:
                failures.append({"seed": seed, "exit_code": code})
        completed = sum((output / f"seed-{seed}-complete.json").is_file() for seed in SEEDS)
        if failures or completed != 3:
            raise RuntimeError(f"seed failures={failures}; completed={completed}/3")
        value = {"state": "complete", "completed_units": 54, "total_units": 54}
        atomic_json(output / "campaign-complete.json", value)
        atomic_json(status, value)
    except Exception as error:
        atomic_json(status, {"state": "failed", "error": repr(error),
                             "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
