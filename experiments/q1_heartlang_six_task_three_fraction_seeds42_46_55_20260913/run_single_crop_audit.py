#!/usr/bin/env python3
"""Run the gated CPSC seed-42 single-crop diagnostic on one GPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from audit_stmem_cpsc_seed42 import AUDIT
from common import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args()
    code = args.root / "src/CLEAR-HUG/experiments/q1_heartlang_six_task_three_fraction_seeds42_46_55_20260913"
    output = args.root / "results" / AUDIT; status = output / "queue-status.json"
    commands = [
        ("identity-freeze-split-audit", [sys.executable, code / "audit_stmem_cpsc_seed42.py", "--root", args.root, "--stage", "audit"]),
        ("extract-train", [sys.executable, code / "audit_stmem_cpsc_seed42.py", "--root", args.root, "--stage", "extract", "--split", "train"]),
        ("extract-val", [sys.executable, code / "audit_stmem_cpsc_seed42.py", "--root", args.root, "--stage", "extract", "--split", "val"]),
        ("linear-probe", [sys.executable, code / "train_probe.py", "--features", output / "features", "--output", output / "checkpoint", "--model", "stmem", "--fraction", "100pct", "--task", "cpsc2018", "--seed", "42"]),
        ("extract-test", [sys.executable, code / "audit_stmem_cpsc_seed42.py", "--root", args.root, "--stage", "extract", "--split", "test"]),
        ("independent-evaluate", [sys.executable, code / "audit_stmem_cpsc_seed42.py", "--root", args.root, "--stage", "evaluate"]),
    ]
    try:
        for stage, command in commands:
            atomic_json(status, {"state": "running", "stage": stage})
            print("RUN", stage, flush=True); subprocess.run(list(map(str, command)), check=True)
        atomic_json(status, {"state": "complete", "stage": "complete"})
    except Exception as exc:
        atomic_json(status, {"state": "failed", "stage": stage, "error": repr(exc)}); raise


if __name__ == "__main__":
    main()
