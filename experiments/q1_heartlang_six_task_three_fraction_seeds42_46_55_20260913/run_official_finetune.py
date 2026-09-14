#!/usr/bin/env python3
"""One-GPU queue for official ST-MEM fine-tuning."""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import atomic_json
from official_protocol import CAMPAIGN, NODES, specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node", choices=tuple(NODES), required=True)
    args = parser.parse_args()
    code = args.root / "src/CLEAR-HUG/experiments/q1_heartlang_six_task_three_fraction_seeds42_46_55_20260913"
    output = args.root / "results" / CAMPAIGN
    status_path = output / f"{args.node}-train-queue-status.json"

    def status(state, spec=None, error=None):
        atomic_json(status_path, {"state": state, "spec": spec, "error": error,
                                  "updated_at": datetime.now(timezone.utc).isoformat()})
    try:
        for fraction, task, seed in specs(args.node):
            spec = f"{fraction}:{task}:{seed}"
            status("running", spec)
            unit = output / fraction / f"{task}-seed{seed}" / "stmem"
            subprocess.run([sys.executable, code / "train_official_finetune.py",
                            "--root", args.root, "--output", unit,
                            "--fraction", fraction, "--task", task,
                            "--seed", str(seed)], check=True)
        status("complete")
    except Exception as exc:
        status("failed", error=repr(exc)); raise


if __name__ == "__main__": main()
