#!/usr/bin/env python3
"""Evaluate the six locked HiLAR checkpoints on validation at threshold 0.5.

This script deliberately uses only the validation feature arrays.  Macro F1 and
exact-match accuracy follow the repository's ``analyze_ecg_classification``
convention; macro AUPRC is threshold independent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from model import HILAKLRA
from protocol import CAMPAIGN, TASKS
from train import FeatureDataset, atomic_json, checkpoints, load_split, macro_metrics, predict


def evaluate(root: Path, task: str, device: torch.device) -> dict:
    val_global, val_lead, val_local, val_mask, val_labels, _, _ = load_split(
        root, task, "val"
    )
    baseline_state, hilak_state, mean, std, _ = checkpoints(root, task)
    dataset = FeatureDataset(
        val_global, val_lead, val_local, val_mask, val_labels, mean, std
    )

    output = root / "results" / CAMPAIGN / task
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model = HILAKLRA(val_labels.shape[1], baseline_state, hilak_state)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)

    probabilities = predict(model, dataset, device)
    labels = np.asarray(val_labels, dtype=np.int32)
    predictions = (probabilities > 0.5).astype(np.int32)
    macro_auroc, macro_auprc = macro_metrics(labels, probabilities)

    complete = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    if abs(macro_auroc - complete["best_val_macro_auroc"]) > 1e-7:
        raise RuntimeError(
            f"{task}: checkpoint AUROC {macro_auroc} does not match selected result "
            f"{complete['best_val_macro_auroc']}"
        )

    result = {
        "state": "complete",
        "task": task,
        "split": "validation",
        "seed": complete["seed"],
        "fraction": complete["fraction"],
        "records": len(labels),
        "classes": labels.shape[1],
        "checkpoint_epoch": checkpoint["epoch"],
        "threshold": 0.5,
        "threshold_selection": "fixed; not optimized on validation",
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(labels, predictions)),
        "labelwise_accuracy": float(np.mean(labels == predictions)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "test_data_loaded": False,
        "test_evaluations": 0,
    }
    atomic_json(output / "val_metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=TASKS, action="append")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tasks = tuple(args.task) if args.task else TASKS
    results = [evaluate(args.root.resolve(), task, device) for task in tasks]
    summary = {
        "state": "complete",
        "split": "validation",
        "threshold": 0.5,
        "tasks": results,
        "mean_macro_auprc": float(np.mean([x["macro_auprc"] for x in results])),
        "mean_macro_f1": float(np.mean([x["macro_f1"] for x in results])),
        "mean_exact_match_accuracy": float(
            np.mean([x["exact_match_accuracy"] for x in results])
        ),
        "mean_labelwise_accuracy": float(
            np.mean([x["labelwise_accuracy"] for x in results])
        ),
        "test_data_loaded": False,
        "test_evaluations": 0,
    }
    result_root = args.root.resolve() / "results" / CAMPAIGN
    atomic_json(result_root / "val_metrics_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
