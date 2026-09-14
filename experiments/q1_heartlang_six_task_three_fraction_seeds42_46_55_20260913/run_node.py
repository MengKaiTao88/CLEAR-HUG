#!/usr/bin/env python3
"""Single-GPU node queue for train/validation or gated formal test."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import atomic_json
from protocol import CAMPAIGN, FRACTIONS, MODELS, NODE_TASKS, NODES, SEEDS, TASKS, specs


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def status(path, state, stage=None, spec=None, error=None):
    atomic_json(path, {"state": state, "stage": stage, "spec": spec, "error": error,
                       "updated_at": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node", choices=tuple(NODES), required=True)
    parser.add_argument("--mode", choices=("canary", "train", "formal"), required=True)
    args = parser.parse_args()
    code = args.root / "src/CLEAR-HUG/experiments/q1_heartlang_six_task_three_fraction_seeds42_46_55_20260913"
    out = args.root / "results" / CAMPAIGN
    features = out / "features"
    gate = out / "global-pretest-gate.json"
    queue_status = out / f"{args.node}-{args.mode}-queue-status.json"
    selected = list(specs(args.node))
    if args.mode == "canary":
        model, fraction, task, seed = __import__("protocol").CANARIES[args.node]
        # Both encoders are exercised on the same canary spec.
        selected = [(name, fraction, task, seed) for name in MODELS]
    try:
        if args.mode in {"canary", "train"}:
            tasks = sorted({item[2] for item in selected})
            models = sorted({item[0] for item in selected})
            for task in tasks:
                for model in models:
                    for split in ("train", "val"):
                        if model == "heartlang":
                            spec_name = f"qrs:{task}:{split}"
                            status(queue_status, "running", "heartlang-qrs", spec_name)
                            run([sys.executable, code / "prepare_heartlang_qrs.py", "--root", args.root,
                                 "--task", task, "--split", split])
                        spec_name = f"features:{model}:{task}:{split}"
                        status(queue_status, "running", "feature-extraction", spec_name)
                        run([sys.executable, code / "extract_features.py", "--root", args.root,
                             "--model", model, "--task", task, "--split", split,
                             "--output", features / model / task / split])
            for model, fraction, task, seed in selected:
                spec_name = f"{model}:{fraction}:{task}:{seed}"
                status(queue_status, "running", "linear-probe", spec_name)
                unit = out / fraction / f"{task}-seed{seed}" / model
                run([sys.executable, code / "train_probe.py", "--features", features,
                     "--output", unit, "--model", model, "--fraction", fraction,
                     "--task", task, "--seed", seed])
        else:
            if not gate.exists():
                raise RuntimeError("global gate is absent")
            for task in NODE_TASKS[args.node]:
                for model in MODELS:
                    if model == "heartlang":
                        spec_name = f"qrs:{task}:test"
                        status(queue_status, "running", "formal-heartlang-qrs", spec_name)
                        run([sys.executable, code / "prepare_heartlang_qrs.py", "--root", args.root,
                             "--task", task, "--split", "test", "--gate", gate])
                    spec_name = f"test-features:{model}:{task}"
                    status(queue_status, "running", "formal-feature-extraction", spec_name)
                    run([sys.executable, code / "extract_features.py", "--root", args.root,
                         "--model", model, "--task", task, "--split", "test", "--gate", gate,
                         "--output", features / model / task / "test"])
            for model, fraction, task, seed in selected:
                spec_name = f"{model}:{fraction}:{task}:{seed}"
                status(queue_status, "running", "formal-test", spec_name)
                unit = out / fraction / f"{task}-seed{seed}" / model
                run([sys.executable, code / "formal_eval.py", "--features", features,
                     "--checkpoint", unit / "checkpoint-best.pth", "--gate", gate,
                     "--output", unit / "formal-test", "--model", model,
                     "--fraction", fraction, "--task", task, "--seed", seed])
        status(queue_status, "complete", args.mode)
    except Exception as exc:
        status(queue_status, "failed", args.mode, error=repr(exc))
        raise


if __name__ == "__main__":
    main()
