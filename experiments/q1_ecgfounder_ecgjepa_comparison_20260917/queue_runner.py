#!/usr/bin/env python3
"""Orchestrate three-GPU feature extraction followed by three seeded probes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from prepare_features import CAMPAIGN


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def launch(command: list[str], log: Path, gpu: int) -> tuple[subprocess.Popen, object]:
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("a", encoding="utf-8")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                               start_new_session=True)
    return process, stream


def run_group(campaign: Path, specs: list[tuple[list[str], Path, int]], phase: str) -> None:
    processes = [launch(*spec) for spec in specs]
    while True:
        states = [process.poll() for process, _ in processes]
        atomic_json(campaign / "queue-status.json", {"state": "running", "phase": phase,
            "workers": [{"pid": process.pid, "gpu": specs[index][2], "returncode": states[index]}
                        for index, (process, _) in enumerate(processes)]})
        if all(state is not None for state in states): break
        time.sleep(10)
    for _, stream in processes: stream.close()
    failures = [state for state in states if state != 0]
    if failures: raise RuntimeError(f"{phase} workers failed: {states}")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    here = Path(__file__).resolve().parent; campaign = root / "results" / CAMPAIGN
    python = sys.executable
    prepare_specs = [
        ([python, str(here / "prepare_features.py"), "--root", str(root), "--model", "ECGFounder",
          "--bases", "ptbxl", "cpsc2018", "csn", "--device", "cuda:0", "--batch-size", "64"],
         campaign / "prepare-founder.log", 0),
        ([python, str(here / "prepare_features.py"), "--root", str(root), "--model", "ECG-JEPA",
          "--bases", "ptbxl", "--device", "cuda:0", "--batch-size", "16"],
         campaign / "prepare-jepa-ptbxl.log", 1),
        ([python, str(here / "prepare_features.py"), "--root", str(root), "--model", "ECG-JEPA",
          "--bases", "cpsc2018", "csn", "--device", "cuda:0", "--batch-size", "16"],
         campaign / "prepare-jepa-cpsc-csn.log", 2),
    ]
    run_group(campaign, prepare_specs, "feature_extraction")
    probe_specs = []
    for gpu, seed in enumerate((42, 46, 55)):
        probe_specs.append(([python, str(here / "run_probe.py"), "--root", str(root),
                             "--seed", str(seed), "--device", "cuda:0"],
                            campaign / f"seed-{seed}.log", gpu))
    run_group(campaign, probe_specs, "linear_probes")
    atomic_json(campaign / "queue-complete.json", {"state": "complete", "completed_units": 108,
                "total_units": 108, "models": ["ECGFounder", "ECG-JEPA"]})
    atomic_json(campaign / "queue-status.json", {"state": "complete", "phase": "complete",
                "completed_units": 108, "total_units": 108})


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"FATAL: {error!r}", file=sys.stderr, flush=True)
        raise
