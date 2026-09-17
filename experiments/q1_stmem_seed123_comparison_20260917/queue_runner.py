#!/usr/bin/env python3
"""Wait for the active comparison, then run ST-MEM extraction and probes on 3 GPUs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from prepare_features import CAMPAIGN


BLOCKING_CAMPAIGN = "q1-ecgfounder-ecgjepa-six-task-three-fraction-seeds42-46-55-20260917"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def launch(command: list[str], log: Path, gpu: int):
    stream = log.open("a", encoding="utf-8"); env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                               start_new_session=True)
    return process, stream


def group(campaign: Path, specs, phase: str) -> None:
    workers = [launch(*spec) for spec in specs]
    while True:
        returncodes = [process.poll() for process, _ in workers]
        atomic_json(campaign / "queue-status.json", {"state": "running", "phase": phase,
            "workers": [{"pid": process.pid, "gpu": specs[i][2], "returncode": returncodes[i]}
                        for i, (process, _) in enumerate(workers)]})
        if all(value is not None for value in returncodes): break
        time.sleep(10)
    for _, stream in workers: stream.close()
    if any(value != 0 for value in returncodes): raise RuntimeError(f"{phase} failed: {returncodes}")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve(); here = Path(__file__).resolve().parent
    campaign = root / "results" / CAMPAIGN; campaign.mkdir(parents=True, exist_ok=True)
    blocker = root / "results" / BLOCKING_CAMPAIGN / "queue-status.json"
    while True:
        try: state = json.loads(blocker.read_text()).get("state")
        except Exception: state = "unknown"
        if state == "complete": break
        atomic_json(campaign / "queue-status.json", {"state": "waiting", "phase": "waiting_for_gpus",
                    "blocking_campaign": BLOCKING_CAMPAIGN, "blocking_state": state})
        time.sleep(30)
    python = sys.executable
    extract_specs = [
        ([python, str(here / "prepare_features.py"), "--root", str(root), "--base", base,
          "--device", "cuda:0", "--batch-size", "32"], campaign / f"prepare-{base}.log", gpu)
        for gpu, base in enumerate(("ptbxl", "cpsc2018", "csn"))]
    group(campaign, extract_specs, "feature_extraction")
    probe_specs = [
        ([python, str(here / "run_probe.py"), "--root", str(root), "--seed", str(seed),
          "--device", "cuda:0"], campaign / f"seed-{seed}.log", gpu)
        for gpu, seed in enumerate((42, 46, 55))]
    group(campaign, probe_specs, "linear_probes")
    value = {"state": "complete", "completed_units": 54, "total_units": 54}
    atomic_json(campaign / "queue-complete.json", value); atomic_json(campaign / "queue-status.json", value)


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"FATAL: {error!r}", file=sys.stderr, flush=True); raise
