#!/usr/bin/env python3
"""Run the shared CLOCS-comparable PyTorch linear-head protocol for ST-MEM."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

from prepare_features import CAMPAIGN, MODEL_NAME, TASKS


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); incoming = path.with_suffix(path.suffix + ".incoming")
    incoming.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 46, 55), required=True)
    parser.add_argument("--device", required=True); args = parser.parse_args()
    root = args.root.resolve(); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    shared_path = Path(__file__).resolve().parents[1] / "q1_ecgfounder_ecgjepa_comparison_20260917/run_probe.py"
    comparison_dir = str(shared_path.parent)
    if comparison_dir not in sys.path: sys.path.insert(1, comparison_dir)
    spec = importlib.util.spec_from_file_location("shared_linear_probe", shared_path)
    shared = importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)
    shared.CAMPAIGN = CAMPAIGN
    status = root / "results" / CAMPAIGN / f"seed-{args.seed}-status.json"; completed = 0
    for task in TASKS:
        for fraction in (0.01, 0.1, 1.0):
            atomic_json(status, {"state": "running", "seed": args.seed, "model": MODEL_NAME,
                "task": task, "fraction": fraction, "completed_units": completed,
                "total_units": 18, "device": str(device)})
            shared.train_unit(root, MODEL_NAME, task, fraction, args.seed, device); completed += 1
    value = {"state": "complete", "seed": args.seed, "completed_units": completed,
             "total_units": 18, "model": MODEL_NAME, "device": str(device)}
    atomic_json(status.with_name(f"seed-{args.seed}-complete.json"), value); atomic_json(status, value)


if __name__ == "__main__": main()
